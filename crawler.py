"""Crawl4AI orchestration with anti-fragility guardrails.

Responsibilities:

* **Loop / trap detection** — a URL + content-hash "ring" (in-memory, or Redis
  when ``REDIS_URL`` is set) stops cyclic redirects and duplicate pages.
* **Memory ceiling** — one shared browser, max 2 tabs, images/CSS/fonts/media
  disabled both via Blink flags and a network route hook, contexts closed and
  ``gc.collect()`` run after each batch.
* **Gated pipeline** — every page flows fetch → hash → Sentry → branch, with a
  stealth re-fetch when a bot challenge is detected.
* **Structured extraction** — verified product pages are handed to
  :func:`auto_extractor.extract_and_persist_html`, the single zero-selector
  extractor (JSON-LD → cached domain strategy → LLM) shared with the drip worker
  and the CLI. This module owns fetch/classify/loop-guarding only; it no longer
  authors its own extraction prompt.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from auto_extractor import extract_and_persist_html
from config import Settings, settings as default_settings
from llm_pool import GeminiClient
from logging_utils import log
from schemas import (
    PageInspectionResult,
    PageTypeEnum,
    ScrapedProductItem,
)
from sentry import SmartSentry

# --------------------------------------------------------------------------- #
# Guarded Crawl4AI import — keep the rest of the package usable without it.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    _CRAWL4AI_ERR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover
    AsyncWebCrawler = BrowserConfig = CrawlerRunConfig = CacheMode = None  # type: ignore
    _CRAWL4AI_ERR = _exc


# Query params stripped during canonicalization (tracking/analytics junk).
_TRACKING_KEYS = {
    "gclid", "fbclid", "mc_cid", "mc_eid", "_ga", "ref", "ref_src",
    "igshid", "yclid", "msclkid", "spm", "scm",
}
# Path fragments that hint at a real product (used to prioritize child links).
_PRODUCT_HINTS = ("product", "products", "/p/", "/dp/", "item", "catalogue", "-p-", "buy")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# URL / content normalization helpers
# --------------------------------------------------------------------------- #
def canonicalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase host, drop fragment + tracking params."""
    p = urlsplit(url.strip())
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "").lower()
    netloc = host
    if p.port and not (
        (scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)
    ):
        netloc = f"{host}:{p.port}"

    pairs = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_KEYS and not k.lower().startswith("utm_")
    ]
    pairs.sort()
    query = urlencode(pairs)

    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, query, ""))


def content_hash(text: str) -> str:
    """Whitespace-normalized MD5 of page content, for cyclic-loop detection."""
    norm = _WS_RE.sub(" ", text or "").strip()
    return hashlib.md5(norm.encode("utf-8", "ignore")).hexdigest()


def registered_domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


# --------------------------------------------------------------------------- #
# Hash ring (in-memory or Redis-backed)
# --------------------------------------------------------------------------- #
class HashRing:
    """Tracks visited canonical URLs and content hashes.

    ``add_*`` returns ``True`` when the member is *newly* added (i.e. NOT seen
    before), so callers read it as "safe to proceed".
    """

    def __init__(self, settings: Settings = default_settings) -> None:
        self._settings = settings
        self._urls: set[str] = set()
        self._hashes: set[str] = set()
        self._redis = None
        if settings.redis_url:
            try:  # pragma: no cover - optional dependency
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(settings.redis_url)
                log.debug("HashRing using Redis backend.")
            except Exception as exc:  # pragma: no cover
                log.warn(f"Redis unavailable ({exc}); falling back to in-memory ring.")
                self._redis = None

    async def _add(self, kind: str, member: str) -> bool:
        if self._redis is not None:  # pragma: no cover - needs live Redis
            key = f"{self._settings.redis_namespace}:{kind}"
            added = await self._redis.sadd(key, member)
            return bool(added)
        store = self._urls if kind == "urls" else self._hashes
        if member in store:
            return False
        store.add(member)
        return True

    async def add_url(self, url: str) -> bool:
        return await self._add("urls", url)

    async def add_hash(self, digest: str) -> bool:
        return await self._add("hashes", digest)

    async def close(self) -> None:
        if self._redis is not None:  # pragma: no cover
            try:
                await self._redis.aclose()
            except AttributeError:
                await self._redis.close()


# --------------------------------------------------------------------------- #
# Pipeline result
# --------------------------------------------------------------------------- #
class PipelineStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    CATEGORY_EXPANDED = "CATEGORY_EXPANDED"
    CLASSIFIED_ONLY = "CLASSIFIED_ONLY"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    DUPLICATE = "DUPLICATE"
    FETCH_FAILED = "FETCH_FAILED"
    ERROR = "ERROR"
    # --- KALA-BALANA DB-aware variants ----------------------------------- #
    # Existing product: price/stock delta logged to price_history.
    DELTA_UPDATED = "DELTA_UPDATED"
    # New product: inserted into canonical_products, forensics queued.
    NEW_PRODUCT_QUEUED = "NEW_PRODUCT_QUEUED"


@dataclass
class PipelineResult:
    url: str
    canonical_url: str
    status: PipelineStatus
    page_type: Optional[PageTypeEnum] = None
    inspection: Optional[PageInspectionResult] = None
    item: Optional[ScrapedProductItem] = None
    child_links: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    used_stealth: bool = False
    decided_by_rules: bool = False
    elapsed_s: float = 0.0


# --------------------------------------------------------------------------- #
# The crawler
# --------------------------------------------------------------------------- #
class AntiFragileCrawler:
    """Orchestrates fetch → classify → extract with memory + loop guardrails."""

    _HEAVY_RESOURCES = {"image", "media", "font", "stylesheet"}

    def __init__(
        self,
        sentry: SmartSentry,
        llm: GeminiClient,
        settings: Settings = default_settings,
        ring: Optional[HashRing] = None,
        db_pool: Optional[object] = None,
        llm_pool: Optional[object] = None,
    ) -> None:
        self._sentry = sentry
        self._llm = llm
        self._settings = settings
        self._ring = ring or HashRing(settings)
        self._crawler = None  # type: ignore
        # Optional KALA-BALANA persistence layer (None → dry-run / backward compat).
        self._db_pool = db_pool
        self._llm_pool = llm_pool

    # ------------------------------------------------------------------ #
    # Browser / run configuration
    # ------------------------------------------------------------------ #
    def _browser_config(self):
        kwargs = dict(
            headless=self._settings.headless,
            text_mode=True,   # disables images at the Blink level
            light_mode=True,  # trims background features -> less RAM
            verbose=False,
            extra_args=self._settings.browser_args(),
        )
        if self._settings.proxy:
            kwargs["proxy"] = self._settings.proxy
        if self._settings.user_agent:
            kwargs["user_agent"] = self._settings.user_agent
        if BrowserConfig is not None:
            return BrowserConfig(**kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(**kwargs)

    def _run_config(self, stealth: bool):
        cache_mode = getattr(CacheMode, "BYPASS", None) if CacheMode is not None else None
        kwargs = dict(
            cache_mode=cache_mode,
            exclude_external_images=True,
            screenshot=False,
            remove_overlay_elements=True,
            word_count_threshold=1,
            verbose=False,
        )
        # Honeypot Rejection: Purge hidden/zero-dimension/off-viewport trap elements directly from DOM
        honeypot_js = """
        document.querySelectorAll('a, input, button, select, textarea, form, [href]').forEach(el => {
            try {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (
                    style.display === 'none' ||
                    style.opacity === '0' ||
                    style.visibility === 'hidden' ||
                    rect.width === 0 ||
                    rect.height === 0 ||
                    rect.top < -500 ||
                    rect.left < -500
                ) {
                    el.remove();
                }
            } catch (e) {}
        });
        """
        kwargs["js_code"] = honeypot_js

        if stealth:
            kwargs.update(
                magic=True,               # bundle of anti-bot evasions
                simulate_user=True,       # Crawl4AI applies randomized bezier mouse movements
                override_navigator=True,
                wait_until="networkidle",  # let a challenge settle/resolve
                page_timeout=self._settings.stealth_page_timeout_ms,
                delay_before_return_html=self._settings.stealth_settle_ms / 1000.0,
            )
        else:
            kwargs.update(
                wait_until="domcontentloaded",
                page_timeout=self._settings.page_timeout_ms,
            )
        if CrawlerRunConfig is not None:
            return CrawlerRunConfig(**kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(**kwargs)

    # ------------------------------------------------------------------ #
    # Resource-blocking hook (belt & suspenders with text_mode)
    # ------------------------------------------------------------------ #
    async def _abort_heavy(self, route) -> None:
        try:
            if route.request.resource_type in self._HEAVY_RESOURCES:
                await route.abort()
            else:
                await route.continue_()
        except Exception:  # pragma: no cover - route may already be handled
            try:
                await route.continue_()
            except Exception:
                pass

    async def _on_context(self, page=None, context=None, **_):
        target = context or page
        try:
            await target.route("**/*", self._abort_heavy)
        except Exception:  # pragma: no cover - signature/version differences
            try:
                await page.route("**/*", self._abort_heavy)
            except Exception:
                pass
        return page

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if _CRAWL4AI_ERR is not None:
            raise RuntimeError(
                "crawl4ai is not installed/importable. Install it with:\n"
                "    pip install crawl4ai\n"
                "    crawl4ai-setup   # installs the Playwright browser\n"
                f"Original import error: {_CRAWL4AI_ERR}"
            )
        self._crawler = AsyncWebCrawler(config=self._browser_config())
        # Register the network-level resource blocker if the hook API is present.
        try:
            self._crawler.crawler_strategy.set_hook("on_page_context_created", self._on_context)
        except Exception as exc:  # pragma: no cover
            log.debug(f"Could not register resource-block hook ({exc}); relying on text_mode.")
        await self._crawler.start()
        log.debug("Browser started (headless=%s, max_tabs=%s)." % (
            self._settings.headless, self._settings.max_concurrency))

    async def close(self) -> None:
        if self._crawler is not None:
            try:
                await self._crawler.close()
            finally:
                self._crawler = None
        await self._ring.close()
        gc.collect()  # reclaim browser/page buffers on the memory-tight VPS
        log.debug("Browser closed and gc.collect() run.")

    async def __aenter__(self) -> "AntiFragileCrawler":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Fetch + parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _markdown_str(result) -> str:
        md = getattr(result, "markdown", None)
        if md is None:
            return ""
        if isinstance(md, str):
            return md
        # Newer Crawl4AI returns a MarkdownGenerationResult object.
        for attr in ("fit_markdown", "raw_markdown"):
            val = getattr(md, attr, None)
            if val:
                return val
        return str(md)

    async def _fetch(self, url: str, stealth: bool):
        assert self._crawler is not None, "call start() before fetching"
        return await self._crawler.arun(url=url, config=self._run_config(stealth))

    def _extract_child_links(self, result, base_url: str) -> List[str]:
        """Pull same-domain child links from a category page, product-ish first."""
        links = getattr(result, "links", None) or {}
        internal = links.get("internal", []) if isinstance(links, dict) else []
        base_dom = registered_domain(base_url)
        base_canon = canonicalize_url(base_url)

        seen: set[str] = set()
        preferred: List[str] = []
        others: List[str] = []
        for entry in internal:
            href = entry.get("href") if isinstance(entry, dict) else entry
            if not href:
                continue
            try:
                canon = canonicalize_url(href)
            except Exception:
                continue
            if canon in seen or canon == base_canon:
                continue
            if registered_domain(canon) != base_dom:
                continue
            seen.add(canon)
            (preferred if any(h in canon.lower() for h in _PRODUCT_HINTS) else others).append(canon)

        ordered = preferred + others
        return ordered[: self._settings.max_children_per_category]

    # ------------------------------------------------------------------ #
    # The gated pipeline for one URL
    # ------------------------------------------------------------------ #
    async def process_url(self, url: str, depth: int = 0) -> PipelineResult:
        t0 = time.perf_counter()
        canonical = canonicalize_url(url)
        res = PipelineResult(url=url, canonical_url=canonical, status=PipelineStatus.ERROR)

        # -- Gate 0: URL dedup ring (cyclic redirect / re-queue guard) ------ #
        if not await self._ring.add_url(canonical):
            log.gate("RING", f"duplicate URL, skipping: {canonical}", ok=False)
            res.status = PipelineStatus.DUPLICATE
            res.notes.append("URL already visited")
            res.elapsed_s = time.perf_counter() - t0
            return res

        log.gate("FETCH", f"(d{depth}) {canonical}")
        try:
            result = await self._fetch(canonical, stealth=False)
        except Exception as exc:
            log.error(f"fetch raised for {canonical}: {exc}")
            res.status = PipelineStatus.FETCH_FAILED
            res.notes.append(f"fetch exception: {exc}")
            res.elapsed_s = time.perf_counter() - t0
            return res

        if not getattr(result, "success", False):
            err = getattr(result, "error_message", "unknown error")
            log.gate("FETCH", f"failed: {err}", ok=False)
            # Still let the Sentry look — a 403/404 body may be classifiable.

        html = getattr(result, "html", "") or getattr(result, "cleaned_html", "") or ""
        markdown = self._markdown_str(result)
        status_code = getattr(result, "status_code", None)
        used_stealth = False

        # -- Gate 1: content-hash ring (loop / mirror detection) ------------ #
        digest = content_hash(markdown or html)
        if not await self._ring.add_hash(digest):
            log.gate("HASH", f"duplicate content hash {digest[:10]}… (loop), skipping", ok=False)
            res.status = PipelineStatus.DUPLICATE
            res.notes.append(f"duplicate content hash {digest}")
            res.elapsed_s = time.perf_counter() - t0
            return res

        # -- Gate 2: Sentry inspection (rules -> LLM) ----------------------- #
        verdict, by_rules = await self._sentry.inspect(
            html=html, markdown=markdown, status_code=status_code
        )
        res.inspection = verdict
        res.decided_by_rules = by_rules
        res.page_type = verdict.page_type
        gate_src = "rules" if by_rules else "LLM"
        log.gate(
            "SENTRY",
            f"{verdict.page_type.value} (conf={verdict.confidence_score:.2f}, via {gate_src}) — "
            f"{verdict.reasoning}",
            ok=verdict.page_type not in (PageTypeEnum.BOT_CHALLENGE_CAPTCHA, PageTypeEnum.ERROR_404_PAGE),
        )

        # -- Gate 3: bot-challenge -> stealth fallback ---------------------- #
        if verdict.page_type == PageTypeEnum.BOT_CHALLENGE_CAPTCHA:
            log.gate("STEALTH", "challenge detected — retrying with stealth profile")
            try:
                result = await self._fetch(canonical, stealth=True)
                used_stealth = True
                html = getattr(result, "html", "") or getattr(result, "cleaned_html", "") or ""
                markdown = self._markdown_str(result)
                status_code = getattr(result, "status_code", None)
                await self._ring.add_hash(content_hash(markdown or html))
                verdict, by_rules = await self._sentry.inspect(
                    html=html, markdown=markdown, status_code=status_code
                )
                res.inspection = verdict
                res.decided_by_rules = by_rules
                res.page_type = verdict.page_type
                log.gate(
                    "STEALTH",
                    f"post-retry: {verdict.page_type.value} (conf={verdict.confidence_score:.2f})",
                    ok=verdict.page_type != PageTypeEnum.BOT_CHALLENGE_CAPTCHA,
                )
            except Exception as exc:
                log.error(f"stealth retry failed: {exc}")
                res.notes.append(f"stealth retry error: {exc}")

        res.used_stealth = used_stealth

        # -- Gate 4: branch on the (possibly updated) verdict --------------- #
        pt = verdict.page_type
        if pt == PageTypeEnum.BOT_CHALLENGE_CAPTCHA:
            res.status = PipelineStatus.BLOCKED
            res.notes.append("still challenged after stealth retry")

        elif pt == PageTypeEnum.CATEGORY_GRID:
            children = self._extract_child_links(result, canonical)
            res.child_links = children
            res.status = PipelineStatus.CATEGORY_EXPANDED
            log.gate("QUEUE", f"category grid → enqueuing {len(children)} child link(s)")

        elif pt in (PageTypeEnum.PRODUCT_PAGE, PageTypeEnum.OUT_OF_STOCK_PLACEHOLDER):
            # Single extraction authority: the zero-selector cascade shared with
            # the drip worker and CLI. It extracts AND persists atomically
            # (dry-run no-op when db_pool is None), enqueuing forensics for new
            # products. crawler.py no longer runs its own LLM-markdown path.
            outcome = await extract_and_persist_html(
                canonical, html,
                pool=self._db_pool, llm_pool=self._llm_pool, settings=self._settings,
            )
            res.notes.extend(outcome.get("notes", []))
            res.notes.append(f"autox_tier={outcome.get('tier')}")
            status = outcome.get("status")
            if status == "new_product_queued":
                res.status = PipelineStatus.NEW_PRODUCT_QUEUED
                log.gate("FORENSIC", f"new product queued (id={str(outcome.get('product_id'))[:8]}…) — "
                                     f"run `python worker.py` to process")
            elif status == "delta_updated":
                res.status = PipelineStatus.DELTA_UPDATED
            elif status in ("extracted_dry_run", "extracted"):
                res.status = (
                    PipelineStatus.OUT_OF_STOCK if outcome.get("in_stock") is False
                    else PipelineStatus.EXTRACTED
                )
            else:  # extract_failed / validation_failed / persist_failed
                res.status = PipelineStatus.CLASSIFIED_ONLY
                log.gate("EXTRACT", f"no item emitted ({status})", ok=False)
            price = outcome.get("price_lkr")
            if price is not None:
                log.gate("EXTRACT", f"LKR {price:,.2f} "
                                    f"({'in stock' if outcome.get('in_stock') else 'OUT OF STOCK'}) "
                                    f"via {outcome.get('tier')}")

        else:  # ERROR_404_PAGE / UNKNOWN_JUNK
            res.status = PipelineStatus.REJECTED
            res.notes.append(f"rejected as {pt.value}")

        res.elapsed_s = time.perf_counter() - t0
        return res

    # ------------------------------------------------------------------ #
    # Batch runner: bounded-concurrency BFS with category expansion
    # ------------------------------------------------------------------ #
    async def crawl_batch(self, seed_urls: List[str]) -> List[PipelineResult]:
        """Process seeds with <= max_concurrency tabs, expanding category grids.

        Bounded by ``max_pages`` (total) and ``max_depth`` (category recursion)
        so a hostile site cannot make the crawl grow without limit.
        """
        await self.start()
        results: List[PipelineResult] = []
        queue: asyncio.Queue = asyncio.Queue()
        for u in seed_urls:
            queue.put_nowait((u, 0))

        processed = 0
        lock = asyncio.Lock()

        async def worker(worker_id: int) -> None:
            nonlocal processed
            while True:
                url, depth = await queue.get()
                try:
                    async with lock:
                        over_budget = processed >= self._settings.max_pages
                    if over_budget:
                        continue  # drain the queue without doing work
                    result = await self.process_url(url, depth)
                    async with lock:
                        processed += 1
                    results.append(result)
                    if result.child_links and depth < self._settings.max_depth:
                        for child in result.child_links:
                            queue.put_nowait((child, depth + 1))
                except Exception as exc:  # keep the worker alive on any failure
                    log.error(f"[worker {worker_id}] unhandled error on {url}: {exc}")
                    results.append(
                        PipelineResult(
                            url=url,
                            canonical_url=canonicalize_url(url),
                            status=PipelineStatus.ERROR,
                            notes=[f"worker exception: {exc}"],
                        )
                    )
                finally:
                    queue.task_done()

        n_workers = max(1, self._settings.max_concurrency)
        workers = [asyncio.create_task(worker(i)) for i in range(n_workers)]
        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.close()

        if processed >= self._settings.max_pages:
            log.warn(f"Hit max_pages guard ({self._settings.max_pages}); remaining queue dropped.")
        return results
