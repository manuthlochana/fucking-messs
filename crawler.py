"""Crawl4AI orchestration with anti-fragility guardrails.

Responsibilities:

* **Loop / trap detection** — a URL + content-hash "ring" (in-memory, or Redis
  when ``REDIS_URL`` is set) stops cyclic redirects and duplicate pages.
* **Memory ceiling** — one shared browser, max 2 tabs, images/CSS/fonts/media
  disabled both via Blink flags and a network route hook, contexts closed and
  ``gc.collect()`` run after each batch.
* **Gated pipeline** — every page flows fetch → hash → Sentry → branch, with a
  stealth re-fetch when a bot challenge is detected.
* **Structured extraction** — verified product pages are parsed into a strictly
  validated :class:`ScrapedProductItem` by Gemini Flash.
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

from config import Settings, settings as default_settings
from llm_pool import GeminiClient
from logging_utils import log
from schemas import (
    PageInspectionResult,
    PageTypeEnum,
    ProductExtraction,
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


EXTRACTION_SYSTEM_INSTRUCTION = (
    "You extract structured e-commerce product data from the Markdown of a single "
    "Sri Lankan (LKR) product page. Rules: return every price as a plain number "
    "with no currency symbol, thousands separator, or spaces (e.g. 'Rs. 45,900' -> "
    "45900). Use null for anything genuinely unknown — never invent values. "
    "clean_title removes marketing noise (superlatives, emoji, 'FREE SHIPPING', "
    "shouting caps) but keeps brand, model numbers and capacities. in_stock is "
    "false for sold-out/unavailable items. Return only a few of the most important "
    "specifications."
)

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
        return BrowserConfig(**kwargs)

    def _run_config(self, stealth: bool):
        kwargs = dict(
            cache_mode=CacheMode.BYPASS,
            exclude_external_images=True,
            screenshot=False,
            remove_overlay_elements=True,
            word_count_threshold=1,
            verbose=False,
        )
        if stealth:
            kwargs.update(
                magic=True,               # bundle of anti-bot evasions
                simulate_user=True,
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
        return CrawlerRunConfig(**kwargs)

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
    # Structured extraction
    # ------------------------------------------------------------------ #
    async def extract_product(self, markdown: str, url: str) -> ScrapedProductItem:
        """Parse Markdown into a validated product item (raises on bad data)."""
        budget = min(len(markdown), self._settings.sentry_char_budget * 3)
        prompt = (
            f"Product URL: {url}\n"
            "Extract the product from this page content:\n"
            "----- BEGIN PAGE CONTENT -----\n"
            f"{markdown[:budget]}\n"
            "----- END PAGE CONTENT -----"
        )
        extraction = await self._llm.generate_structured(
            prompt,
            ProductExtraction,
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=1024,
        )
        # to_item() runs the strict validators (price finite & > 0, etc.).
        return extraction.to_item(url)

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
            try:
                item = await self.extract_product(markdown, canonical)
                if pt == PageTypeEnum.OUT_OF_STOCK_PLACEHOLDER:
                    item.in_stock = False
                res.item = item
                res.status = (
                    PipelineStatus.OUT_OF_STOCK if not item.in_stock else PipelineStatus.EXTRACTED
                )
                log.gate(
                    "EXTRACT",
                    f"'{item.clean_title}' — LKR {item.price_lkr:,.2f} "
                    f"({'in stock' if item.in_stock else 'OUT OF STOCK'})",
                )
                # -------------------------------------------------------- #
                # KALA-BALANA DB-aware fork (no-op when db_pool is None)
                # -------------------------------------------------------- #
                if self._db_pool is not None:
                    await self._persist_product(item, canonical, res)
            except Exception as exc:
                # Strict validation failed (e.g. no positive price) — keep the
                # classification but don't emit a bogus item.
                res.status = PipelineStatus.CLASSIFIED_ONLY
                res.notes.append(f"extraction/validation failed: {exc}")
                log.gate("EXTRACT", f"validation failed, no item emitted: {exc}", ok=False)

        else:  # ERROR_404_PAGE / UNKNOWN_JUNK
            res.status = PipelineStatus.REJECTED
            res.notes.append(f"rejected as {pt.value}")

        res.elapsed_s = time.perf_counter() - t0
        return res

    # ------------------------------------------------------------------ #
    # KALA-BALANA: DB persistence + forensics dispatcher
    # ------------------------------------------------------------------ #
    async def _persist_product(
        self,
        item: "ScrapedProductItem",
        listing_url: str,
        res: PipelineResult,
    ) -> None:
        """Persist a scraped product to PostgreSQL and dispatch forensics if new.

        Case A — Existing product (fingerprint found):
            Update listings.current_price_lkr and insert a price_history row.
            Sets res.status = DELTA_UPDATED.

        Case B — New product (fingerprint not found):
            Insert into canonical_products as 'discovered', upsert the merchant
            and listing rows, then launch run_forensic_pipeline as a background
            asyncio task.
            Sets res.status = NEW_PRODUCT_QUEUED.
        """
        from urllib.parse import urlsplit as _urlsplit

        from db import queries
        from normalizer import SpecNormalizer

        try:
            norm = SpecNormalizer()
            spec = norm.normalize(item.raw_title, brand_hint=item.brand)

            # Upsert merchant row from the listing URL's domain.
            domain = (_urlsplit(listing_url).netloc or "unknown").lower()
            merchant_id = await queries.upsert_merchant(
                self._db_pool,
                domain=domain,
                display_name=domain,
            )

            existing_id = await queries.lookup_by_fingerprint(
                self._db_pool, spec.spec_fingerprint
            )

            if existing_id:
                # ---- Case A: Delta check -------------------------------- #
                log.gate("DB", f"existing product (fp={spec.spec_fingerprint[:12]}…) — delta update")
                listing_id = await queries.upsert_listing(
                    self._db_pool,
                    product_id=existing_id,
                    merchant_id=merchant_id,
                    listing_url=listing_url,
                    price_lkr=item.price_lkr,
                    in_stock=item.in_stock,
                )
                if listing_id:
                    await queries.log_price_history(
                        self._db_pool,
                        listing_id=listing_id,
                        price_lkr=item.price_lkr,
                        in_stock=item.in_stock,
                    )
                res.status = PipelineStatus.DELTA_UPDATED
                res.notes.append(
                    f"delta: price=LKR {item.price_lkr:,.2f} "
                    f"in_stock={item.in_stock} fingerprint={spec.spec_fingerprint[:16]}…"
                )

            else:
                # ---- Case B: New product -------------------------------- #
                log.gate("DB", f"NEW product detected — inserting + queuing forensics")
                product_id = await queries.insert_canonical_product(
                    self._db_pool,
                    spec_fingerprint=spec.spec_fingerprint,
                    brand=spec.brand or item.brand,
                    model_family=spec.model_family,
                    sub_model=spec.sub_model,
                    storage_gb=spec.storage_gb,
                    ram_gb=spec.ram_gb,
                    region_code=spec.region_code,
                    raw_title=item.raw_title,
                    clean_title=item.clean_title,
                )
                if product_id:
                    listing_id = await queries.upsert_listing(
                        self._db_pool,
                        product_id=product_id,
                        merchant_id=merchant_id,
                        listing_url=listing_url,
                        price_lkr=item.price_lkr,
                        in_stock=item.in_stock,
                    )
                    if listing_id:
                        await queries.log_price_history(
                            self._db_pool,
                            listing_id=listing_id,
                            price_lkr=item.price_lkr,
                            in_stock=item.in_stock,
                        )
                    # Enqueue to the durable forensic_queue (consumed by the
                    # standalone --worker process) rather than spawning an
                    # unmanaged asyncio background task inside the crawl loop.
                    if product_id:
                        await queries.enqueue_forensic_job(
                            self._db_pool,
                            product_id=product_id,
                            merchant_id=merchant_id,
                            listing_url=listing_url,
                        )
                        log.gate(
                            "FORENSIC",
                            f"Job enqueued for {item.clean_title!r} "
                            f"(id={product_id[:8]}…) — run `python main.py --worker` to process",
                        )
                res.status = PipelineStatus.NEW_PRODUCT_QUEUED
                res.notes.append(
                    f"new: fingerprint={spec.spec_fingerprint[:16]}… "
                    f"brand={spec.brand} sub_model={spec.sub_model!r}"
                )

        except Exception as exc:
            # DB errors must never crash the crawl loop.
            log.warn(f"[DB persist] non-fatal error: {exc}")
            res.notes.append(f"db_persist error (non-fatal): {exc}")

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
