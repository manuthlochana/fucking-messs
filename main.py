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
from typing import List, Optional

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

