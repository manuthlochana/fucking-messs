"""Autonomous site discovery + polite drip crawling (Step 1).

The operator supplies only a domain URL (e.g. ``https://tudo.lk``). This module
maps the entire catalogue and drip-scrapes it over days — no per-site selector
authoring.

Public API
----------
* :func:`ingest_domain` — discover product URLs and populate ``crawl_queue``.
* :func:`run_drip_worker` — polite, memory-bounded queue consumer.

Discovery strategy (no bs4/lxml/playwright — httpx + stdlib xml + regex only):
    1. Fetch ``robots.txt``; collect ``Sitemap:`` directives.
    2. Fetch each sitemap; recurse ``<sitemapindex>`` entries; harvest ``<loc>``.
    3. Keep only same-registered-domain, product-looking URLs.
    4. If sitemaps yield nothing, BFS-spider from the homepage, following
       same-domain links up to ``discovery_max_spider_pages``.
    5. Bulk-enqueue into ``crawl_queue`` (idempotent ON CONFLICT DO NOTHING).

Drip worker (polite / self-throttling / memory-bounded):
    * claims one URL at a time, <= ``drip_max_concurrency`` in flight;
    * randomized ``drip_min_delay_s``..``drip_max_delay_s`` jitter per fetch;
    * recycles the httpx client every ``drip_context_recycle_pages`` fetches;
    * pauses + ``gc.collect()`` when RSS exceeds ``drip_mem_soft_limit_mb``;
    * per-domain circuit breaker: ``circuit_breaker_threshold`` consecutive
      403/429/503 → pause the domain for ``circuit_breaker_cooldown_s``.

Never raises out of the worker loop — a bad page fails its queue row and the
loop continues.
"""

from __future__ import annotations

import asyncio
import gc
import random
import re
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

from config import Settings, settings as default_settings
from db import queries
from logging_utils import log

# --------------------------------------------------------------------------- #
# URL heuristics
# --------------------------------------------------------------------------- #
# Path/query fragments that strongly imply a single product page.
_PRODUCT_URL_RE = re.compile(
    r"(/product/|/products/|/p/|/item/|/items/|/buy/|/dp/|-p-|/shop/"
    r"|[?&]sku=|[?&]product_id=|[?&]pid=)",
    re.I,
)
# Extensions / paths that are never products — skipped during spidering.
_SKIP_URL_RE = re.compile(
    r"(\.(?:jpg|jpeg|png|gif|webp|svg|ico|css|js|pdf|zip|mp4|woff2?|ttf|xml|json)(?:$|\?)"
    r"|/cart|/checkout|/login|/register|/account|/wishlist|/wp-admin|/wp-login"
    r"|/cdn-cgi/|#|tel:|mailto:|javascript:)",
    re.I,
)
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#\s]+)["']""", re.I)
_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _registered_domain(url: str) -> str:
    """Best-effort eTLD+1 without the ``tldextract`` dependency.

    Good enough for same-site containment: ``www.shop.tudo.lk`` and ``tudo.lk``
    both collapse to ``tudo.lk``. Two-label public suffixes common in .lk
    (``com.lk``, ``co.uk`` …) are handled so we keep the label before them.
    """
    host = _host(url)
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    two_label_suffix = parts[-2] in {"com", "co", "net", "org", "gov", "ac", "edu"} and len(parts[-1]) == 2
    take = 3 if two_label_suffix else 2
    return ".".join(parts[-take:])


def _canon(url: str) -> str:
    """Strip fragment + trailing slash so dedup is stable."""
    p = urlsplit(url.strip())
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "").lower()
    netloc = host
    if p.port and not ((scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)):
        netloc = f"{host}:{p.port}"
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, p.query, ""))


def is_product_url(url: str) -> bool:
    """True when the URL looks like a single product/detail page."""
    if _SKIP_URL_RE.search(url):
        return False
    return bool(_PRODUCT_URL_RE.search(url))


def _normalize_domain_url(domain_url: str) -> str:
    """Accept ``tudo.lk`` or ``https://tudo.lk/x`` → scheme+host origin."""
    raw = domain_url.strip()
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    p = urlsplit(raw)
    host = (p.hostname or "").lower()
    netloc = f"{host}:{p.port}" if p.port else host
    return urlunsplit(((p.scheme or "https").lower(), netloc, "/", "", ""))


# --------------------------------------------------------------------------- #
# HTTP fetch helper (dependency-light: httpx only)
# --------------------------------------------------------------------------- #
def _default_headers(settings: Settings) -> Dict[str, str]:
    return {
        "User-Agent": settings.user_agent
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


async def _fetch(
    client,
    url: str,
) -> Tuple[Optional[int], str]:
    """GET a URL. Returns ``(status_code, body)``; ``(None, "")`` on transport error."""
    try:
        resp = await client.get(url)
        # ``resp.text`` decodes lazily; guard against gigantic non-HTML bodies.
        body = resp.text if len(resp.content) <= 5_000_000 else ""
        return resp.status_code, body
    except Exception as exc:  # transport / timeout / TLS
        log.debug(f"[Discovery] fetch error {url}: {exc!r}")
        return None, ""


# --------------------------------------------------------------------------- #
# robots.txt + sitemap parsing (stdlib + regex, no lxml)
# --------------------------------------------------------------------------- #
def parse_robots(text: str) -> Dict[str, List[str]]:
    """Extract ``Sitemap:`` directives and ``Disallow:`` rules from robots.txt."""
    sitemaps: List[str] = []
    disallows: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "sitemap" and val:
            sitemaps.append(val)
        elif key == "disallow" and val:
            disallows.append(val)
    return {"sitemaps": sitemaps, "disallows": disallows}


def extract_sitemap_locs(xml_text: str) -> Tuple[bool, List[str]]:
    """Return ``(is_index, locs)`` from sitemap XML via regex (namespace-agnostic)."""
    is_index = "<sitemapindex" in (xml_text or "").lower()
    locs = [m.strip() for m in _LOC_RE.findall(xml_text or "") if m.strip()]
    return is_index, locs


async def discover_via_sitemap(
    client,
    origin: str,
    robots_sitemaps: List[str],
    *,
    max_sitemaps: int = 100,
) -> Tuple[List[str], List[str]]:
    """Walk sitemap(s) breadth-first. Returns ``(product_urls, sitemaps_seen)``.

    Recurses ``<sitemapindex>`` children up to ``max_sitemaps`` total fetches so
    a hostile/looping sitemap cannot grow the crawl without bound.
    """
    base_dom = _registered_domain(origin)
    # Seed with robots-declared sitemaps, then the two conventional locations.
    to_visit: List[str] = list(robots_sitemaps) + [
        urljoin(origin, "/sitemap.xml"),
        urljoin(origin, "/sitemap_index.xml"),
    ]
    seen_sitemaps: Set[str] = set()
    product_urls: List[str] = []
    product_seen: Set[str] = set()
    fetched = 0

    while to_visit and fetched < max_sitemaps:
        sm = _canon(to_visit.pop(0))
        if sm in seen_sitemaps:
            continue
        seen_sitemaps.add(sm)
        status, body = await _fetch(client, sm)
        fetched += 1
        if status != 200 or not body:
            continue
        is_index, locs = extract_sitemap_locs(body)
        if is_index:
            for loc in locs:
                if _registered_domain(loc) == base_dom and _canon(loc) not in seen_sitemaps:
                    to_visit.append(loc)
        else:
            for loc in locs:
                if _registered_domain(loc) != base_dom:
                    continue
                cu = _canon(loc)
                if cu not in product_seen and is_product_url(cu):
                    product_seen.add(cu)
                    product_urls.append(cu)

    return product_urls, sorted(seen_sitemaps)


# --------------------------------------------------------------------------- #
# BFS link spidering (fallback when no usable sitemap exists)
# --------------------------------------------------------------------------- #
def extract_links(html: str, base_url: str) -> List[str]:
    """Resolve all same-page ``href`` values to absolute URLs (regex, no bs4)."""
    out: List[str] = []
    for href in _HREF_RE.findall(html or ""):
        try:
            absu = urljoin(base_url, href)
        except Exception:
            continue
        if absu.startswith(("http://", "https://")):
            out.append(absu)
    return out


async def spider_bfs(
    client,
    origin: str,
    *,
    max_pages: int,
    delay_range: Tuple[float, float] = (0.0, 0.0),
) -> List[str]:
    """BFS from the homepage, harvesting product-looking same-domain URLs.

    Politeness delay is applied between page fetches when ``delay_range`` is
    non-zero. Bounded by ``max_pages`` crawled pages.
    """
    base_dom = _registered_domain(origin)
    queue: List[str] = [_canon(origin)]
    visited: Set[str] = set()
    products: Set[str] = set()
    pages = 0

    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        status, body = await _fetch(client, url)
        pages += 1
        if status != 200 or not body:
            continue
        for link in extract_links(body, url):
            if _registered_domain(link) != base_dom:
                continue
            cu = _canon(link)
            if is_product_url(cu):
                products.add(cu)
            if cu not in visited and not _SKIP_URL_RE.search(cu) and len(visited) + len(queue) < max_pages * 4:
                queue.append(cu)
        lo, hi = delay_range
        if hi > 0:
            await asyncio.sleep(random.uniform(lo, hi))

    return sorted(products)


# --------------------------------------------------------------------------- #
# Public: ingest_domain — discover + populate crawl_queue
# --------------------------------------------------------------------------- #
async def ingest_domain(
    domain_url: str,
    pool: Optional[object],
    *,
    daily_page_limit: int = 1500,
    settings: Settings = default_settings,
) -> Dict[str, object]:
    """Discover every product URL for ``domain_url`` and enqueue it.

    Returns a summary dict: ``{domain, origin, method, discovered, enqueued,
    daily_page_limit, sitemaps}``. Safe in dry-run mode (``pool is None``): it
    still performs discovery and reports counts but enqueues nothing.
    """
    origin = _normalize_domain_url(domain_url)
    domain = _host(origin)
    log.header(f"KALA-BALANA — INGEST DOMAIN {domain}")

    try:
        import httpx
    except Exception as exc:  # pragma: no cover - httpx is a project dep
        log.error(f"httpx unavailable, cannot ingest: {exc}")
        return {"domain": domain, "origin": origin, "method": "none",
                "discovered": 0, "enqueued": 0, "error": str(exc)}

    method = "sitemap"
    sitemaps: List[str] = []
    async with httpx.AsyncClient(
        timeout=20.0, follow_redirects=True, headers=_default_headers(settings)
    ) as client:
        # 1. robots.txt
        _, robots_body = await _fetch(client, urljoin(origin, "/robots.txt"))
        robots = parse_robots(robots_body)
        log.info(f"robots.txt: {len(robots['sitemaps'])} sitemap directive(s), "
                 f"{len(robots['disallows'])} disallow rule(s)")

        # 2. sitemap walk
        product_urls, sitemaps = await discover_via_sitemap(
            client, origin, robots["sitemaps"]
        )
        log.info(f"sitemap discovery: {len(product_urls)} product URL(s) from "
                 f"{len(sitemaps)} sitemap(s)")

        # 3. spider fallback
        if not product_urls:
            method = "spider"
            log.warn("no product URLs from sitemaps — falling back to BFS spider")
            product_urls = await spider_bfs(
                client, origin,
                max_pages=settings.discovery_max_spider_pages,
                delay_range=(settings.drip_min_delay_s, settings.drip_max_delay_s),
            )
            log.info(f"spider discovery: {len(product_urls)} product URL(s)")

    # Respect the per-run daily budget cap on how many we enqueue at once.
    capped = product_urls[: max(1, daily_page_limit)] if daily_page_limit else product_urls
    enqueued = await queries.enqueue_crawl_urls(pool, domain=domain, urls=capped)

    log.success(
        f"ingest_domain({domain}) — discovered={len(product_urls)} "
        f"enqueued={enqueued} method={method} (limit={daily_page_limit})"
    )
    return {
        "domain": domain,
        "origin": origin,
        "method": method,
        "discovered": len(product_urls),
        "enqueued": enqueued,
        "daily_page_limit": daily_page_limit,
        "sitemaps": sitemaps,
    }


# --------------------------------------------------------------------------- #
# Drip worker: polite, memory-bounded crawl_queue consumer
# --------------------------------------------------------------------------- #
def _rss_mb() -> float:
    """Resident set size in MB (psutil if present, else ``resource``)."""
    try:  # pragma: no cover - optional dependency
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource
            import sys

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports ru_maxrss in KB; macOS/BSD report it in bytes.
            return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
        except Exception:
            return 0.0


class _CircuitBreaker:
    """Per-domain breaker: N consecutive block-statuses → pause for a cooldown."""

    def __init__(self, threshold: int, cooldown_s: int) -> None:
        self._threshold = max(1, threshold)
        self._cooldown_s = max(1, cooldown_s)
        self._fails: Dict[str, int] = {}
        self._paused_until: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_open(self, domain: str) -> bool:
        async with self._lock:
            until = self._paused_until.get(domain, 0.0)
            if until and time.monotonic() < until:
                return True
            if until:
                # Cooldown elapsed — reset.
                self._paused_until.pop(domain, None)
                self._fails[domain] = 0
            return False

    async def record(self, domain: str, blocked: bool) -> None:
        async with self._lock:
            if not blocked:
                self._fails[domain] = 0
                return
            self._fails[domain] = self._fails.get(domain, 0) + 1
            if self._fails[domain] >= self._threshold:
                self._paused_until[domain] = time.monotonic() + self._cooldown_s
                log.warn(
                    f"[Drip] circuit breaker OPEN for {domain} — "
                    f"{self._fails[domain]} consecutive blocks, pausing "
                    f"{self._cooldown_s}s"
                )


_BLOCK_STATUSES = {403, 429, 503}


def _resolve_html_extractor() -> Tuple[Optional[str], Optional[Callable]]:
    """Prefer the Step-2 zero-selector extractor; fall back to single-URL ingest.

    Returns ``(mode, fn)`` where mode is ``"html"`` (fn takes pre-fetched HTML)
    or ``"url"`` (fn re-fetches the URL itself), or ``(None, None)``.
    """
    try:
        from auto_extractor import extract_and_persist_html  # type: ignore

        return "html", extract_and_persist_html
    except Exception:
        pass
    try:
        from main import fetch_and_parse_product_page

        return "url", fetch_and_parse_product_page
    except Exception:
        return None, None


async def run_drip_worker(
    pool: Optional[object],
    *,
    domain: Optional[str] = None,
    settings: Settings = default_settings,
    llm_pool: Optional[object] = None,
    daily_page_limit: int = 1500,
    stop_event: Optional[asyncio.Event] = None,
    idle_grace_s: float = 15.0,
) -> Dict[str, int]:
    """Consume ``crawl_queue`` politely until drained, budget hit, or stopped.

    One shared circuit breaker across ``drip_max_concurrency`` worker coroutines;
    each keeps its own httpx client and recycles it every
    ``drip_context_recycle_pages`` fetches to bound memory. Returns
    ``{scraped, failed, skipped}`` counters.
    """
    if pool is None:
        log.warn("[Drip] no DB pool — drip worker is a no-op in dry-run mode.")
        return {"scraped": 0, "failed": 0, "skipped": 0}

    try:
        import httpx
    except Exception as exc:  # pragma: no cover
        log.error(f"[Drip] httpx unavailable: {exc}")
        return {"scraped": 0, "failed": 0, "skipped": 0}

    breaker = _CircuitBreaker(
        settings.circuit_breaker_threshold, settings.circuit_breaker_cooldown_s
    )
    ex_mode, ex_fn = _resolve_html_extractor()
    if ex_fn is None:
        log.warn("[Drip] no extractor available — pages will be marked done without extraction.")

    counters = {"scraped": 0, "failed": 0, "skipped": 0}
    budget = {"remaining": max(1, daily_page_limit)}
    state_lock = asyncio.Lock()
    n_workers = max(1, settings.drip_max_concurrency)
    delay = (settings.drip_min_delay_s, settings.drip_max_delay_s)
    recycle_n = max(1, settings.drip_context_recycle_pages)
    mem_limit = settings.drip_mem_soft_limit_mb
    headers = _default_headers(settings)

    log.header(f"KALA-BALANA — DRIP CRAWL{f' [{domain}]' if domain else ''}")
    log.info(
        f"  workers={n_workers} jitter={delay[0]}-{delay[1]}s "
        f"recycle@{recycle_n} mem_cap={mem_limit}MB budget={daily_page_limit}"
    )

    async def worker(wid: int) -> None:
        client = httpx.AsyncClient(timeout=25.0, follow_redirects=True, headers=headers)
        since_recycle = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                async with state_lock:
                    if budget["remaining"] <= 0:
                        return
                # Claim the next pending URL (SKIP LOCKED across workers).
                job = await queries.claim_next_crawl_url(pool, domain)
                if job is None:
                    # Nothing pending — wait briefly, then exit if still empty.
                    await asyncio.sleep(min(idle_grace_s, 5.0))
                    again = await queries.claim_next_crawl_url(pool, domain)
                    if again is None:
                        return
                    job = again

                url = job["url"]
                jdom = job.get("domain") or _host(url)

                # Circuit breaker gate.
                if await breaker.is_open(jdom):
                    await queries.fail_crawl_url(pool, url, "circuit_breaker_open")
                    async with state_lock:
                        counters["skipped"] += 1
                    continue

                # Memory ceiling — pause + GC before proceeding.
                rss = _rss_mb()
                if mem_limit and rss and rss > mem_limit:
                    log.warn(f"[Drip w{wid}] RSS {rss:.0f}MB > {mem_limit}MB — GC + pause")
                    gc.collect()
                    await asyncio.sleep(2.0)

                # Politeness jitter before every fetch.
                await asyncio.sleep(random.uniform(*delay))

                status, body = await _fetch(client, url)
                since_recycle += 1

                blocked = status in _BLOCK_STATUSES
                await breaker.record(jdom, blocked)

                if status != 200 or not body:
                    await queries.fail_crawl_url(pool, url, f"HTTP {status}")
                    await queries.record_domain_profile_result(pool, jdom, success=False)
                    async with state_lock:
                        counters["failed"] += 1
                else:
                    try:
                        if ex_mode == "html":
                            await ex_fn(url, body, pool=pool, llm_pool=llm_pool, settings=settings)  # type: ignore[misc]
                        elif ex_mode == "url":
                            await ex_fn(url, db_pool=pool, llm_pool=llm_pool)  # type: ignore[misc]
                        await queries.complete_crawl_url(pool, url)
                        await queries.record_domain_profile_result(pool, jdom, success=True)
                        async with state_lock:
                            counters["scraped"] += 1
                    except Exception as exc:
                        await queries.fail_crawl_url(pool, url, f"extract error: {exc}")
                        async with state_lock:
                            counters["failed"] += 1

                async with state_lock:
                    budget["remaining"] -= 1

                # Recycle the client to bound memory growth.
                if since_recycle >= recycle_n:
                    await client.aclose()
                    gc.collect()
                    client = httpx.AsyncClient(
                        timeout=25.0, follow_redirects=True, headers=headers
                    )
                    since_recycle = 0
                    log.debug(f"[Drip w{wid}] recycled HTTP client after {recycle_n} pages")
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    await asyncio.gather(*(worker(i) for i in range(n_workers)))
    log.success(
        f"[Drip] done — scraped={counters['scraped']} "
        f"failed={counters['failed']} skipped={counters['skipped']}"
    )
    return counters

