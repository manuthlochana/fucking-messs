"""End-to-end demonstration of the Smart Sentry scraping pipeline.

Runs a small batch of URLs through the full engine and prints a colored,
gate-by-gate trace followed by a summary table.

Usage:
    python main.py                            # demo crawl, no DB
    python main.py <url1> <url2> ...         # scrape your own targets
    python main.py --db                       # same demo but with PostgreSQL
    python main.py --db --forensic-demo <url> # full KALA-BALANA pipeline

The default URLs point at https://books.toscrape.com — a sandbox *built for
scraping practice* — so the demo is runnable and ToS-friendly out of the box:
a product page, a category grid, and a deliberate 404 to exercise the error
gate. Swap in your own LKR store URLs (active product / out-of-stock / category)
via the command line. (The demo site prices are in GBP; the numeric value is
still parsed into ``price_lkr`` — the field name assumes an LKR store.)
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

from config import settings
from crawler import AntiFragileCrawler, PipelineResult, PipelineStatus
from llm import GeminiClient
from logging_utils import log
from sentry import SmartSentry

# (label, url) — swap these for real LKR store targets as needed.
DEFAULT_TARGETS: List[tuple[str, str]] = [
    ("active product", "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"),
    ("category grid", "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"),
    ("error / 404 page", "https://books.toscrape.com/catalogue/this-page-does-not-exist_9999/index.html"),
]

_STATUS_STYLE = {
    PipelineStatus.EXTRACTED: log.success,
    PipelineStatus.OUT_OF_STOCK: log.warn,
    PipelineStatus.CATEGORY_EXPANDED: log.info,
    PipelineStatus.CLASSIFIED_ONLY: log.warn,
    PipelineStatus.BLOCKED: log.error,
    PipelineStatus.REJECTED: log.warn,
    PipelineStatus.DUPLICATE: log.info,
    PipelineStatus.FETCH_FAILED: log.error,
    PipelineStatus.ERROR: log.error,
    PipelineStatus.DELTA_UPDATED: log.success,
    PipelineStatus.NEW_PRODUCT_QUEUED: log.success,
}


def _print_summary(results: List[PipelineResult]) -> None:
    log.header("PIPELINE SUMMARY")
    for r in results:
        emit = _STATUS_STYLE.get(r.status, log.info)
        pt = r.page_type.value if r.page_type else "—"
        emit(f"{r.status.value:<17} [{pt}] {r.canonical_url}  ({r.elapsed_s:.1f}s)")
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

    # Aggregate counts.
    tally: dict[str, int] = {}
    for r in results:
        tally[r.status.value] = tally.get(r.status.value, 0) + 1
    log.rule()
    log.info("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


async def _print_db_summary(pool: object) -> None:
    """Print a brief summary of the KALA-BALANA database state."""
    try:
        row = await pool.fetchrow(  # type: ignore[attr-defined]
            """
            SELECT
                (SELECT COUNT(*) FROM canonical_products) AS products,
                (SELECT COUNT(*) FROM listings)           AS listings,
                (SELECT COUNT(*) FROM price_history)      AS price_rows,
                (SELECT COUNT(*) FROM defect_dossiers)    AS defects,
                (SELECT COUNT(*) FROM currency_arbitrage_logs) AS arb_logs
            """
        )
        if row:
            log.header("KALA-BALANA DATABASE SUMMARY")
            log.info(f"  canonical_products : {row['products']}")
            log.info(f"  listings           : {row['listings']}")
            log.info(f"  price_history rows : {row['price_rows']}")
            log.info(f"  defect_dossiers    : {row['defects']}")
            log.info(f"  arbitrage_logs     : {row['arb_logs']}")
    except Exception as exc:
        log.warn(f"Could not fetch DB summary: {exc}")


async def run(
    urls: List[str],
    use_db: bool = False,
    forensic_demo: bool = False,
) -> None:
    log.header("KALA-BALANA AUTONOMOUS FORENSIC INTELLIGENCE ENGINE")
    log.info(
        f"model={settings.gemini_model}  max_concurrency={settings.max_concurrency}  "
        f"max_pages={settings.max_pages}  max_depth={settings.max_depth}  "
        f"db={'enabled' if use_db else 'disabled (dry-run)'}"
    )

    # Fail fast with a friendly message if the key is missing.
    try:
        settings.require_api_key()
    except RuntimeError as exc:
        log.error(str(exc))
        return

    # Build LLM clients.
    llm = GeminiClient(settings)

    # Build the multi-key pool (used for forensics + DB mode).
    llm_pool = None
    try:
        from llm import MultiKeyLLMPool
        llm_pool = MultiKeyLLMPool.from_settings(settings)
        log.debug(f"LLM Pool ready:\n{llm_pool.slot_summary()}")
    except Exception as exc:
        log.warn(f"MultiKeyLLMPool init failed, forensics will be disabled: {exc}")

    sentry = SmartSentry(llm, settings)

    # Initialize DB pool and run migrations if requested.
    db_pool = None
    if use_db:
        try:
            from db.pool import create_pool
            from db.migrations import run_migrations
            db_pool = await create_pool(
                settings.db_dsn,
                max_size=settings.db_pool_max_size,
                min_size=settings.db_pool_min_size,
            )
            if db_pool:
                await run_migrations(db_pool)
        except Exception as exc:
            log.error(f"DB initialization failed: {exc}")
            log.warn("Continuing without DB (dry-run mode).")
            db_pool = None

    crawler = AntiFragileCrawler(
        sentry, llm, settings,
        db_pool=db_pool,
        llm_pool=llm_pool,
    )

    log.info(f"seed URLs: {len(urls)}")
    for u in urls:
        log.debug(f"  seed: {u}")
    log.rule()

    if forensic_demo and db_pool is not None:
        log.info("FORENSIC DEMO MODE: crawling then waiting up to 300s for forensic pipeline…")

    results = await crawler.crawl_batch(urls)
    _print_summary(results)

    # In forensic demo mode, wait for background forensic tasks to complete.
    if forensic_demo and db_pool is not None:
        pending = [t for t in asyncio.all_tasks() if t.get_name().startswith("forensic_")]
        if pending:
            log.info(f"Waiting for {len(pending)} forensic pipeline task(s)…")
            done, _ = await asyncio.wait(pending, timeout=310)
            log.success(f"Forensic tasks finished: {len(done)}/{len(pending)}")
        await _print_db_summary(db_pool)

    # Graceful DB pool teardown.
    if db_pool is not None:
        try:
            from db.pool import close_pool
            await close_pool(db_pool)
        except Exception:
            pass


def main() -> None:
    args = sys.argv[1:]

    use_db = "--db" in args
    forensic_demo = "--forensic-demo" in args

    # Strip flags from the URL list.
    urls = [a for a in args if not a.startswith("--")]

    if not urls:
        log.info("No URLs given — using built-in demo targets (books.toscrape.com sandbox).")
        for label, url in DEFAULT_TARGETS:
            log.debug(f"  {label}: {url}")
        urls = [u for _, u in DEFAULT_TARGETS]

    try:
        asyncio.run(run(urls, use_db=use_db, forensic_demo=forensic_demo))
    except KeyboardInterrupt:
        log.warn("interrupted by user")


if __name__ == "__main__":
    main()
