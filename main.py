"""KALA-BALANA — System CLI entry point.

Three execution modes:

    python main.py --crawl <url>   Bounded Crawl4AI crawl with anti-bot stealth.
                                   New products are enqueued to the durable
                                   forensic_queue table; the crawl process exits
                                   cleanly after the batch completes.

    python main.py --worker        Standalone serial forensic intelligence worker.
                                   Polls forensic_queue, claims one job at a time
                                   (Concurrency=1 to protect the 4GB VPS memory
                                   ceiling), runs the 5-phase 300s pipeline, then
                                   marks each job done/failed. Runs until SIGINT.

    python main.py --demo          End-to-end sandbox demonstration.
                                   Crawls a safe practice site (books.toscrape.com),
                                   prints the gate-by-gate trace, and — when DB is
                                   available — runs one forensic worker cycle inline
                                   with full trace logging.

Common flags:
    --db    Enable PostgreSQL persistence (requires DB_DSN env var).
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional

from config import settings
from logging_utils import log


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_llm_clients():
    """Return (GeminiClient, MultiKeyLLMPool|None)."""
    from llm_pool import GeminiClient, MultiKeyLLMPool

    llm = GeminiClient(settings)
    llm_pool: Optional[object] = None
    try:
        llm_pool = MultiKeyLLMPool.from_settings(settings)
        log.debug(f"LLM Pool ready:\n{llm_pool.slot_summary()}")  # type: ignore[attr-defined]
    except Exception as exc:
        log.warn(f"MultiKeyLLMPool init failed, forensics will be disabled: {exc}")
    return llm, llm_pool


async def _init_db():
    """Create the asyncpg pool and run idempotent migrations. Returns pool or None."""
    from db.pool import create_pool
    from db.migrations import run_migrations

    try:
        pool = await create_pool(
            settings.db_dsn,
            max_size=settings.db_pool_max_size,
            min_size=settings.db_pool_min_size,
        )
        if pool:
            await run_migrations(pool)
        return pool
    except Exception as exc:
        log.error(f"DB initialization failed: {exc}")
        log.warn("Continuing without DB (dry-run mode).")
        return None


async def _close_db(pool: Optional[object]) -> None:
    if pool is None:
        return
    try:
        from db.pool import close_pool
        await close_pool(pool)  # type: ignore[arg-type]
    except Exception:
        pass


def _print_summary(results: list) -> None:
    from crawler import PipelineStatus

    _STATUS_STYLE = {
        PipelineStatus.EXTRACTED:          log.success,
        PipelineStatus.OUT_OF_STOCK:       log.warn,
        PipelineStatus.CATEGORY_EXPANDED:  log.info,
        PipelineStatus.CLASSIFIED_ONLY:    log.warn,
        PipelineStatus.BLOCKED:            log.error,
        PipelineStatus.REJECTED:           log.warn,
        PipelineStatus.DUPLICATE:          log.info,
        PipelineStatus.FETCH_FAILED:       log.error,
        PipelineStatus.ERROR:              log.error,
        PipelineStatus.DELTA_UPDATED:      log.success,
        PipelineStatus.NEW_PRODUCT_QUEUED: log.success,
    }

    log.header("PIPELINE SUMMARY")
    for r in results:
        emit = _STATUS_STYLE.get(r.status, log.info)
        pt = r.page_type.value if r.page_type else "—"
        emit(f"{r.status.value:<20} [{pt}] {r.canonical_url}  ({r.elapsed_s:.1f}s)")
        if r.item:
            it = r.item
            log.info(
                f"    → {it.clean_title} | brand={it.brand or '—'} | "
                f"LKR {it.price_lkr:,.2f}"
                + (f" (was {it.original_price_lkr:,.2f})" if it.original_price_lkr else "")
                + f" | in_stock={it.in_stock}"
                + (f" | warranty={it.warranty_claimed}" if it.warranty_claimed else "")
            )
            if it.specs_snippet:
                specs = ", ".join(f"{k}={v}" for k, v in list(it.specs_snippet.items())[:5])
                log.info(f"      specs: {specs}")
        if r.child_links:
            log.info(f"    → expanded into {len(r.child_links)} child link(s)")
        for note in r.notes:
            log.debug(f"    note: {note}")

    tally: dict[str, int] = {}
    for r in results:
        tally[r.status.value] = tally.get(r.status.value, 0) + 1
    log.rule()
    log.info("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


async def _print_db_summary(pool: object) -> None:
    try:
        row = await pool.fetchrow(  # type: ignore[attr-defined]
            """
            SELECT
                (SELECT COUNT(*) FROM canonical_products)      AS products,
                (SELECT COUNT(*) FROM listings)                AS listings,
                (SELECT COUNT(*) FROM price_history)           AS price_rows,
                (SELECT COUNT(*) FROM defect_dossiers)         AS defects,
                (SELECT COUNT(*) FROM currency_arbitrage_logs) AS arb_logs,
                (SELECT COUNT(*) FROM forensic_queue WHERE status = 'pending') AS queue_pending,
                (SELECT COUNT(*) FROM forensic_queue WHERE status = 'done')    AS queue_done
            """
        )
        if row:
            log.header("KALA-BALANA DATABASE SUMMARY")
            log.info(f"  canonical_products  : {row['products']}")
            log.info(f"  listings            : {row['listings']}")
            log.info(f"  price_history rows  : {row['price_rows']}")
            log.info(f"  defect_dossiers     : {row['defects']}")
            log.info(f"  arbitrage_logs      : {row['arb_logs']}")
            log.info(f"  forensic_queue      : {row['queue_pending']} pending / {row['queue_done']} done")
    except Exception as exc:
        log.warn(f"Could not fetch DB summary: {exc}")


# ---------------------------------------------------------------------------
# Single-URL ingestion (Track A wiring)
# ---------------------------------------------------------------------------

# Bank/agent warranty phrases → warranty_tier enum. Deterministic, conservative:
# anything we cannot positively classify stays 'unstated'.
_WARRANTY_TIER_HINTS = (
    ("official_agent", ("official warranty", "agent warranty", "authorized",
                         "authorised", "brand warranty", "manufacturer warranty")),
    ("shop_inhouse",   ("shop warranty", "seller warranty", "store warranty",
                         "in-house", "inhouse")),
    ("checking",       ("warranty checking", "being verified", "to be confirmed")),
)


def _map_warranty_tier(warranty_claimed: Optional[str]) -> str:
    """Map a free-text warranty claim to the ``warranty_tier`` enum (best-effort)."""
    if not warranty_claimed:
        return "unstated"
    low = warranty_claimed.lower()
    for tier, needles in _WARRANTY_TIER_HINTS:
        if any(n in low for n in needles):
            return tier
    return "unstated"


async def _extract_headline_fields(
    url: str,
    html: str,
    llm: Optional[object],
) -> Optional["object"]:
    """Return a validated ``ScrapedProductItem`` for the page.

    Prefers the LLM structured extractor (same contract as
    ``crawler.AntiFragileCrawler.extract_product``); on failure or when no LLM
    is configured, falls back to deterministic regex anchors so ingestion still
    yields a row in dry-run / offline mode. Returns ``None`` if even the
    deterministic path cannot recover a usable price.
    """
    import re

    from schemas import ScrapedProductItem

    # --- Preferred path: LLM structured extraction ------------------------- #
    if llm is not None:
        try:
            from crawler import EXTRACTION_SYSTEM_INSTRUCTION
            from schemas import ProductExtraction

            budget = min(len(html), settings.sentry_char_budget * 3)
            prompt = (
                f"Product URL: {url}\n"
                "Extract the product from this page content:\n"
                "----- BEGIN PAGE CONTENT -----\n"
                f"{html[:budget]}\n"
                "----- END PAGE CONTENT -----"
            )
            extraction = await llm.generate_structured(  # type: ignore[attr-defined]
                prompt,
                ProductExtraction,
                system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=1024,
            )
            return extraction.to_item(url)
        except Exception as exc:
            log.warn(f"[Ingest] LLM extraction failed ({exc}); using deterministic fallback.")

    # --- Deterministic fallback (mirrors the delta-poll regex anchors) ----- #
    price: Optional[float] = None
    json_ld_m = re.search(r'"price"\s*:\s*["\']?([\d,]+(?:\.\d{2})?)["\']?', html, re.I)
    if json_ld_m:
        try:
            price = float(json_ld_m.group(1).replace(",", "")) or None
        except ValueError:
            price = None
    if price is None:
        lkr_m = re.search(r'(?:Rs\.?|LKR)\s*([\d,]{3,}(?:\.\d{2})?)', html, re.I)
        if lkr_m:
            try:
                price = float(lkr_m.group(1).replace(",", "")) or None
            except ValueError:
                price = None
    if not price or price <= 0:
        return None

    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    raw_title = (title_m.group(1).strip() if title_m else url)[:300] or url
    oos = re.search(r"\b(out of stock|sold out|unavailable|discontinued)\b", html, re.I)
    in_stock_hit = re.search(r"\b(in stock|add to cart|buy now)\b", html, re.I)
    in_stock = not (oos and not in_stock_hit)

    try:
        return ScrapedProductItem(
            raw_title=raw_title,
            clean_title=raw_title,
            brand=None,
            price_lkr=price,
            in_stock=in_stock,
        )
    except Exception:
        return None


async def fetch_and_parse_product_page(
    url: str,
    db_pool: Optional[object] = None,
    llm_pool: Optional[object] = None,
    llm: Optional[object] = None,
    *,
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    """Fetch one product page, extract, normalize, and persist atomically.

    This is the standalone single-URL ingestion entrypoint that Track A refers
    to (see ``extractor.py``'s module docstring). Flow:

        fetch DOM (httpx)
          → resolve domain profile (site_profiles.loader.get_profile)
          → Track A commerce signals (extractor.extract_track_a)
          → headline fields (LLM structured extraction, else deterministic regex)
          → spec normalization (normalizer.SpecNormalizer)
          → atomic Postgres upsert + durable forensic_queue enqueue.

    Durable queue state lives in Postgres ``forensic_queue`` (the crawler's
    HashRing is the only Redis-backed structure and covers URL dedup, not job
    state). Never raises — returns a result dict describing the outcome.
    """
    import re
    from urllib.parse import urlsplit

    result: Dict[str, Any] = {"url": url, "status": "error", "notes": []}

    # 1. Fetch the DOM.
    headers = {
        "User-Agent": settings.user_agent
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        import httpx

        async with httpx.AsyncClient(
            timeout=timeout_s, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            result["status"] = "fetch_failed"
            result["notes"].append(f"HTTP {resp.status_code}")
            return result
        html = resp.text
    except Exception as exc:
        result["status"] = "fetch_failed"
        result["notes"].append(f"fetch exception: {exc}")
        log.warn(f"[Ingest] fetch failed for {url}: {exc}")
        return result

    # 2. Resolve the site profile (never fatal — falls back to the generic one).
    profile = None
    try:
        from site_profiles.loader import get_profile

        profile = get_profile(url)
        result["profile"] = getattr(profile, "name", None)
    except Exception as exc:
        result["notes"].append(f"profile load skipped: {exc}")

    # 3. Headline fields (LLM structured, else deterministic).
    item = await _extract_headline_fields(url, html, llm)
    if item is None:
        result["status"] = "extract_failed"
        result["notes"].append("no usable product price recovered")
        return result

    # 4. Track A commerce signals (BNPL markup, 2-of-3 stock, promos, surcharges).
    try:
        from extractor import extract_track_a

        track_a = extract_track_a(html, cash_price=item.price_lkr, profile=profile)
        result["track_a"] = {
            "bnpl_plans": len(track_a.bnpl_plans),
            "bank_promos": len(track_a.bank_promos),
            "surcharges": len(track_a.surcharges),
            "stock_verdict": track_a.stock.in_stock,
            "stock_confidence": track_a.stock.confidence,
        }
        # A 2-of-3-verified stock verdict overrides the headline in_stock flag.
        if track_a.stock.agreement_reached and track_a.stock.in_stock is not None:
            item.in_stock = bool(track_a.stock.in_stock)
    except Exception as exc:
        track_a = None
        result["notes"].append(f"track_a skipped: {exc}")

    # 5. Normalize the title into a canonical spec fingerprint.
    from normalizer import SpecNormalizer

    spec = SpecNormalizer().normalize(item.raw_title, brand_hint=item.brand)
    result["fingerprint"] = spec.spec_fingerprint[:16]

    # 6. Dry-run mode: nothing to persist.
    if db_pool is None:
        result["status"] = "extracted_dry_run"
        result["price_lkr"] = item.price_lkr
        result["in_stock"] = item.in_stock
        log.info(
            f"[Ingest] (dry-run) {item.clean_title} — LKR {item.price_lkr:,.2f} "
            f"in_stock={item.in_stock} fp={spec.spec_fingerprint[:12]}…"
        )
        return result

    # 7. Atomic persistence + durable forensic enqueue.
    try:
        from db import queries

        domain = (urlsplit(url).netloc or "unknown").lower()
        warranty_tier = _map_warranty_tier(getattr(item, "warranty_claimed", None))

        async with db_pool.acquire() as conn:  # type: ignore[attr-defined]
            async with conn.transaction():
                merchant_id = await queries.upsert_merchant(
                    conn, domain=domain, display_name=domain
                )
                existing_id = await queries.lookup_by_fingerprint(
                    conn, spec.spec_fingerprint
                )
                product_id = existing_id
                if not product_id:
                    product_id = await queries.insert_canonical_product(
                        conn,
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

                listing_id = None
                if product_id:
                    listing_id = await queries.upsert_listing(
                        conn,
                        product_id=product_id,
                        merchant_id=merchant_id,
                        listing_url=url,
                        price_lkr=item.price_lkr,
                        warranty_tier=warranty_tier,
                        in_stock=item.in_stock,
                    )
                    if listing_id:
                        await queries.log_price_history(
                            conn,
                            listing_id=listing_id,
                            price_lkr=item.price_lkr,
                            in_stock=item.in_stock,
                        )

                if existing_id:
                    result["status"] = "delta_updated"
                elif product_id:
                    # New product → enqueue durable forensic job (Track B input).
                    await queries.enqueue_forensic_job(
                        conn,
                        product_id=product_id,
                        merchant_id=merchant_id,
                        listing_url=url,
                    )
                    result["status"] = "new_product_queued"

        result["product_id"] = product_id
        result["price_lkr"] = item.price_lkr
        result["in_stock"] = item.in_stock
        log.success(
            f"[Ingest] {result['status']}: {item.clean_title} — "
            f"LKR {item.price_lkr:,.2f} (fp={spec.spec_fingerprint[:12]}…)"
        )
    except Exception as exc:
        result["status"] = "persist_failed"
        result["notes"].append(f"db persist error: {exc}")
        log.warn(f"[Ingest] persist failed for {url}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Mode: --delta-poll
# ---------------------------------------------------------------------------
async def cmd_delta_poll(use_db: bool) -> None:
    """Nightly fast-delta poller for price/stock updates."""
    log.header("KALA-BALANA — DELTA POLL MODE")
    if not use_db:
        log.error("Delta poll requires database (--db). Exiting.")
        return
        
    db_pool = await _init_db()
    if not db_pool:
        return
        
    try:
        from db import queries
        import httpx
        import re
        from schemas import ProductExtraction
        llm, llm_pool = _build_llm_clients()
        from sentry import SmartSentry
        sentry = SmartSentry(llm, settings)

        log.info("Fetching existing active listings for nightly delta polling...")
        rows = await db_pool.fetch(
            "SELECT id, product_id, listing_url, current_price_lkr, in_stock FROM listings"
        )
        log.info(f"Found {len(rows)} listings to poll.")
        
        headers = {
            "User-Agent": settings.user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

        updated_count = 0
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            for row in rows:
                url = row["listing_url"]
                lid = row["id"]
                current_price = float(row["current_price_lkr"] or 0.0)
                old_stock = bool(row["in_stock"])
                
                try:
                    resp = await client.get(url)
                    new_stock = old_stock
                    new_price = None

                    if resp.status_code == 404:
                        new_stock = False
                    elif resp.status_code == 200:
                        html = resp.text

                        # 1. Stock status check
                        oos_match = re.search(r"\b(out of stock|sold out|unavailable|discontinued|no stock)\b", html, re.I)
                        in_stock_match = re.search(r"\b(in stock|add to cart|buy now|available)\b", html, re.I)
                        if oos_match and not in_stock_match:
                            new_stock = False
                        elif in_stock_match:
                            new_stock = True

                        # 2. Price extraction: regex anchors (JSON-LD, microdata, currency prefixes)
                        # JSON-LD offer price
                        json_ld_m = re.search(r'"price"\s*:\s*["\']?([\d,]+(?:\.\d{2})?)["\']?', html, re.I)
                        if json_ld_m:
                            try:
                                p_val = float(json_ld_m.group(1).replace(",", ""))
                                if p_val > 0:
                                    new_price = p_val
                            except ValueError:
                                pass

                        # DOM price tag (Rs. / LKR)
                        if new_price is None:
                            lkr_m = re.search(r'(?:Rs\.?|LKR)\s*([\d,]{3,}(?:\.\d{2})?)', html, re.I)
                            if lkr_m:
                                try:
                                    p_val = float(lkr_m.group(1).replace(",", ""))
                                    if p_val > 0:
                                        new_price = p_val
                                except ValueError:
                                    pass

                        # If regex is ambiguous and LLM is available, use Sentry / structured extraction on truncated HTML
                        if new_price is None and llm is not None:
                            try:
                                snippet = html[:4000]
                                ext = await sentry.extract_product_item(url, snippet)
                                if ext and ext.price_lkr > 0:
                                    new_price = ext.price_lkr
                                    new_stock = ext.in_stock
                            except Exception:
                                pass

                    # 3. Determine if price or stock changed
                    price_changed = (new_price is not None) and (abs(new_price - current_price) > 0.01)
                    stock_changed = (new_stock != old_stock)

                    if price_changed or stock_changed:
                        effective_price = new_price if new_price is not None else current_price
                        await db_pool.execute(
                            """
                            UPDATE listings
                            SET current_price_lkr = $2, in_stock = $3, updated_at = NOW()
                            WHERE id = $1::uuid
                            """,
                            lid, effective_price, new_stock
                        )
                        await queries.log_price_history(
                            db_pool,
                            listing_id=str(lid),
                            price_lkr=effective_price,
                            in_stock=new_stock,
                        )
                        updated_count += 1
                        log.info(f"[Delta Poll] {url} -> LKR {current_price} => {effective_price} | in_stock: {old_stock} => {new_stock}")

                except Exception as exc:
                    log.warn(f"[Delta Poll] Failed to poll {url}: {exc}")
                    
        log.success(f"Delta polling complete: {updated_count}/{len(rows)} listings updated.")
    finally:
        await _close_db(db_pool)

# ---------------------------------------------------------------------------
# Mode: --crawl
# ---------------------------------------------------------------------------

async def cmd_crawl(urls: List[str], use_db: bool) -> None:
    """Crawl one or more URLs, persist new products, enqueue forensic jobs."""
    from crawler import AntiFragileCrawler
    from sentry import SmartSentry

    log.header("KALA-BALANA — CRAWL MODE")
    log.info(
        f"  model={settings.gemini_model}  "
        f"max_concurrency={settings.max_concurrency}  "
        f"max_pages={settings.max_pages}  "
        f"max_depth={settings.max_depth}  "
        f"db={'enabled' if use_db else 'disabled (dry-run)'}"
    )

    try:
        settings.require_api_key()
    except RuntimeError as exc:
        log.error(str(exc))
        return

    llm, llm_pool = _build_llm_clients()
    sentry = SmartSentry(llm, settings)
    db_pool = await _init_db() if use_db else None

    crawler = AntiFragileCrawler(
        sentry, llm, settings,
        db_pool=db_pool,
        llm_pool=llm_pool,
    )

    log.info(f"seed URLs ({len(urls)}):")
    for u in urls:
        log.debug(f"  {u}")
    log.rule()

    results = await crawler.crawl_batch(urls)
    _print_summary(results)

    if db_pool is not None:
        await _print_db_summary(db_pool)

    await _close_db(db_pool)


# ---------------------------------------------------------------------------
# Mode: --worker
# ---------------------------------------------------------------------------

async def cmd_worker(use_db: bool) -> None:
    """Run the serial forensic worker until interrupted (SIGINT/SIGTERM)."""
    from forensic_worker import run_worker_loop

    if not use_db:
        log.error("--worker requires --db (a PostgreSQL connection is mandatory for the queue).")
        return

    try:
        settings.require_api_key()
    except RuntimeError as exc:
        log.error(str(exc))
        return

    _, llm_pool = _build_llm_clients()
    if llm_pool is None:
        log.error("Cannot start worker: no LLM API keys available.")
        return

    db_pool = await _init_db()
    if db_pool is None:
        log.error("Cannot start worker: DB connection failed.")
        return

    try:
        await run_worker_loop(db_pool, llm_pool, settings)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warn("[Worker] Interrupted — shutting down gracefully.")
    finally:
        await _close_db(db_pool)


# ---------------------------------------------------------------------------
# Mode: --demo
# ---------------------------------------------------------------------------

_DEMO_TARGETS: List[tuple[str, str]] = [
    ("active product", "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"),
    ("category grid",  "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"),
    ("404 error page", "https://books.toscrape.com/catalogue/this-page-does-not-exist_9999/index.html"),
]


async def _drain_one_forensic_job(db_pool, llm_pool) -> None:
    """Pull and run one forensic job inline (for demo mode only)."""
    from db import queries
    from forensic_worker import ForensicContext, run_forensic_pipeline
    from normalizer import NormalizedSpec
    from schemas import ScrapedProductItem as _SPI

    job = await queries.claim_next_forensic_job(db_pool)
    if job is None:
        return

    row = await db_pool.fetchrow(
        "SELECT raw_title, clean_title, brand, model_family, sub_model, "
        "storage_gb, ram_gb, region_code FROM canonical_products WHERE id=$1::uuid",
        job["product_id"],
    )
    if row is None:
        await queries.fail_forensic_job(db_pool, job["id"], "not_found")
        return

    spec = NormalizedSpec.from_parts(
        brand=row["brand"] or "",
        model_family=row["model_family"] or "",
        sub_model=row["sub_model"] or "",
        ram_gb=row["ram_gb"],
        storage_gb=row["storage_gb"],
        region_code=row["region_code"] or "",
    )
    scraped_item = _SPI(
        raw_title=row["raw_title"],
        clean_title=row["clean_title"],
        brand=row["brand"],
        price_lkr=1.0,
        in_stock=True,
        product_url=job["listing_url"],
    )
    ctx = ForensicContext(
        product_id=job["product_id"],
        spec=spec,
        scraped_item=scraped_item,
        listing_url=job["listing_url"],
        merchant_id=job.get("merchant_id"),
    )
    report = await asyncio.wait_for(
        run_forensic_pipeline(ctx, db_pool, llm_pool, settings),
        timeout=310.0,
    )
    await queries.complete_forensic_job(db_pool, job["id"])
    log.success(f"[Demo] Forensic pipeline status: {report.final_status}")


async def cmd_demo(use_db: bool) -> None:
    """End-to-end sandbox demonstration with full gate trace."""
    from crawler import AntiFragileCrawler
    from sentry import SmartSentry

    log.header("KALA-BALANA — DEMO MODE (books.toscrape.com sandbox)")
    log.info("  Targets: 1 product page + 1 category grid + 1 deliberate 404")

    try:
        settings.require_api_key()
    except RuntimeError as exc:
        log.error(str(exc))
        return

    llm, llm_pool = _build_llm_clients()
    sentry = SmartSentry(llm, settings)
    db_pool = await _init_db() if use_db else None

    seed_urls = [u for _, u in _DEMO_TARGETS]
    for label, url in _DEMO_TARGETS:
        log.debug(f"  {label}: {url}")

    crawler = AntiFragileCrawler(
        sentry, llm, settings,
        db_pool=db_pool,
        llm_pool=llm_pool,
    )

    results = await crawler.crawl_batch(seed_urls)
    _print_summary(results)

    if db_pool is not None and llm_pool is not None:
        pending_count = 0
        try:
            row = await db_pool.fetchrow(
                "SELECT COUNT(*) AS n FROM forensic_queue WHERE status = 'pending'"
            )
            pending_count = row["n"] if row else 0
        except Exception:
            pass

        if pending_count > 0:
            log.info(f"\n[Demo] {pending_count} forensic job(s) queued — running inline…")
            try:
                await asyncio.wait_for(
                    _drain_one_forensic_job(db_pool, llm_pool),
                    timeout=315.0,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                log.warn(f"[Demo] Forensic inline run ended: {exc}")

        await _print_db_summary(db_pool)

    await _close_db(db_pool)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    use_db = "--db" in args
    args_no_db = [a for a in args if a != "--db"]

    if not args_no_db:
        log.info("No mode flag — defaulting to --demo. "
                 "Use --crawl <url>, --worker --db, or --demo.")
        try:
            asyncio.run(cmd_demo(use_db=use_db))
        except KeyboardInterrupt:
            log.warn("Interrupted by user.")
        return

    mode = args_no_db[0]

    if mode == "--crawl":
        urls = args_no_db[1:]
        if not urls:
            log.error("--crawl requires at least one URL.")
            log.error("  Usage: python main.py --crawl <url> [url…] [--db]")
            sys.exit(1)
        try:
            asyncio.run(cmd_crawl(urls, use_db=use_db))
        except KeyboardInterrupt:
            log.warn("Crawl interrupted by user.")

    elif mode == "--worker":
        try:
            asyncio.run(cmd_worker(use_db=use_db))
        except KeyboardInterrupt:
            log.warn("Worker interrupted by user.")

    elif mode == "--demo":
        try:
            asyncio.run(cmd_demo(use_db=use_db))
        except KeyboardInterrupt:
            log.warn("Demo interrupted by user.")
            
    elif mode == "--delta-poll":
        try:
            asyncio.run(cmd_delta_poll(use_db=use_db))
        except KeyboardInterrupt:
            log.warn("Delta poll interrupted by user.")

    elif mode == "--dashboard":
        port = 8000
        if "--port" in args:
            try:
                p_idx = args.index("--port")
                if p_idx + 1 < len(args):
                    port = int(args[p_idx + 1])
            except Exception:
                pass
        try:
            from dashboard.app import start_dashboard
            start_dashboard(host="0.0.0.0", port=port)
        except KeyboardInterrupt:
            log.warn("Dashboard stopped by user.")

    else:
        # Backward compatibility: bare URLs treated as --crawl targets.
        urls = [a for a in args_no_db if not a.startswith("--")]
        if urls:
            try:
                asyncio.run(cmd_crawl(urls, use_db=use_db))
            except KeyboardInterrupt:
                log.warn("Interrupted by user.")
        else:
            log.error(
                f"Unknown mode: {mode!r}\n"
                "Usage:\n"
                "  python main.py --crawl <url> [url…] [--db]\n"
                "  python main.py --worker --db\n"
                "  python main.py --demo [--db]\n"
                "  python main.py --dashboard [--port 8000]\n"
            )
            sys.exit(1)



if __name__ == "__main__":
    main()

