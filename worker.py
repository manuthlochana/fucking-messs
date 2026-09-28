"""KALA-BALANA autonomous background worker — the crawl_queue consumer.

This is the process that makes the data actually flow:

    Dashboard / --ingest-domain  ──enqueue──▶  crawl_queue (Postgres)
                                                     │
                                              ┌──────┴───────┐
                                              ▼              ▼
                                   THIS WORKER's two concurrent loops
                                              │              │
                          crawl consumer ─────┘              └───── forensic consumer
                     (browser render + extract)              (run_worker_loop drains
                                              │               forensic_queue)
                                              ▼
                        canonical_products / listings / price_history
                                              │
                                              └──enqueue──▶ forensic_queue

Loop 1 — crawl consumer (``_crawl_consumer``):
    * claims the oldest pending ``crawl_queue`` row (``FOR UPDATE SKIP LOCKED``);
    * renders it in a REAL headless browser via ``AntiFragileCrawler`` — Crawl4AI
      executes the page's JavaScript so SPA/React/Next.js product pages and their
      dynamically-injected links resolve (raw HTTP would see an empty shell);
    * classifies with the Sentry, then either expands a category grid (feeding the
      dynamically-discovered child links back into ``crawl_queue``) or extracts +
      persists a product via ``auto_extractor`` (which enqueues forensics for new
      products);
    * sleeps a polite 5–15 s between pages; a per-domain circuit breaker pauses a
      host after repeated 403/429/503-style blocks.

Loop 2 — forensic consumer (``forensic_worker.run_worker_loop``):
    * drains ``forensic_queue`` one job at a time (Track B intelligence).

Degradation: if Crawl4AI is not installed, the crawl consumer falls back to the
static httpx drip worker (``discovery.run_drip_worker``) — it still drains the
queue but cannot render JavaScript, so SPA catalogs will under-extract. Install
crawl4ai (``pip install crawl4ai && crawl4ai-setup``) for full JS rendering.

Run it with::

    python worker.py            # or: python main.py --worker --db
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from config import settings
from logging_utils import log

# The worker honours a 5–15 s polite delay between page fetches per the operator
# contract, regardless of how DRIP_*_DELAY_S is tuned lower for the static path.
POLITE_MIN_S = 5.0
POLITE_MAX_S = 15.0


def _browser_available() -> bool:
    """True when Crawl4AI can drive a headless browser for JS rendering."""
    try:
        from crawler import _CRAWL4AI_ERR  # type: ignore

        return _CRAWL4AI_ERR is None
    except Exception:
        return False


async def _crawl_consumer(
    db_pool: object,
    llm: object,
    llm_pool: Optional[object],
    *,
    stop_event: asyncio.Event,
    poll_idle_s: float = 10.0,
) -> None:
    """Browser-backed ``crawl_queue`` consumer (Loop 1). Runs until ``stop_event``."""
    from crawler import AntiFragileCrawler, PipelineStatus
    from db import queries
    from discovery import _CircuitBreaker, _host
    from sentry import SmartSentry

    lo = max(POLITE_MIN_S, settings.drip_min_delay_s)
    hi = max(POLITE_MAX_S, settings.drip_max_delay_s)
    breaker = _CircuitBreaker(
        settings.circuit_breaker_threshold, settings.circuit_breaker_cooldown_s
    )

    sentry = SmartSentry(llm, settings)
    crawler = AntiFragileCrawler(
        sentry, llm, settings, db_pool=db_pool, llm_pool=llm_pool
    )
    await crawler.start()  # opens the shared headless browser
    log.header("KALA-BALANA CRAWL WORKER — headless browser queue consumer online")
    log.info(f"  polite jitter={lo:.0f}-{hi:.0f}s | max_depth={settings.max_depth}")

    # Statuses that mean "this URL did not yield a good page" → fail the row.
    _FAIL_STATUSES = {
        PipelineStatus.FETCH_FAILED,
        PipelineStatus.ERROR,
        PipelineStatus.BLOCKED,
    }

    try:
        while not stop_event.is_set():
            try:
                job = await queries.claim_next_crawl_url(db_pool)
            except Exception as exc:
                log.warn(f"[CrawlWorker] claim error (retrying): {exc}")
                await _sleep_or_stop(stop_event, poll_idle_s)
                continue

            if job is None:
                # Queue empty — keep the browser warm and re-poll.
                await _sleep_or_stop(stop_event, poll_idle_s)
                continue

            url = job["url"]
            jdom = job.get("domain") or _host(url)
            depth = int(job.get("depth") or 0)

            if await breaker.is_open(jdom):
                await queries.fail_crawl_url(db_pool, url, "circuit_breaker_open")
                continue

            # Politeness: 5–15 s jitter before touching the site.
            await asyncio.sleep(random.uniform(lo, hi))

            try:
                result = await crawler.process_url(url, depth=depth)
            except Exception as exc:
                log.warn(f"[CrawlWorker] process_url failed for {url}: {exc}")
                await queries.fail_crawl_url(db_pool, url, f"process error: {exc}")
                await queries.record_domain_profile_result(db_pool, jdom, success=False)
                continue

            await breaker.record(jdom, result.status == PipelineStatus.BLOCKED)

            # Feed dynamically-discovered category child links back into the queue.
            if result.child_links and depth < settings.max_depth:
                try:
                    added = await queries.enqueue_crawl_urls(
                        db_pool, domain=jdom, urls=result.child_links, depth=depth + 1
                    )
                    if added:
                        log.info(f"[CrawlWorker] +{added} child URL(s) discovered from {url}")
                except Exception as exc:
                    log.warn(f"[CrawlWorker] enqueue children failed: {exc}")

            # Finalize the queue row.
            if result.status in _FAIL_STATUSES:
                note = "; ".join(result.notes[:3]) if result.notes else ""
                await queries.fail_crawl_url(db_pool, url, f"{result.status.value}: {note}")
                await queries.record_domain_profile_result(db_pool, jdom, success=False)
            else:
                await queries.complete_crawl_url(db_pool, url)
                await queries.record_domain_profile_result(
                    db_pool, jdom,
                    success=result.status != PipelineStatus.CLASSIFIED_ONLY,
                )
    finally:
        await crawler.close()
        log.info("[CrawlWorker] browser closed; crawl consumer stopped.")


async def _crawl_consumer_static(
    db_pool: object,
    llm_pool: Optional[object],
    *,
    stop_event: asyncio.Event,
) -> None:
    """Fallback crawl consumer when Crawl4AI is absent (no JS rendering)."""
    from discovery import run_drip_worker

    log.warn(
        "[CrawlWorker] Crawl4AI not installed — using the STATIC httpx drip "
        "consumer. SPA/React product pages and JS-injected links will NOT "
        "render. Install crawl4ai for full browser rendering."
    )
    while not stop_event.is_set():
        await run_drip_worker(
            db_pool,
            settings=settings,
            llm_pool=llm_pool,
            daily_page_limit=settings.drip_daily_page_limit,
            stop_event=stop_event,
        )
        if stop_event.is_set():
            break
        # Queue drained — idle before polling for freshly-enqueued URLs.
        await _sleep_or_stop(stop_event, 30.0)


async def _sleep_or_stop(stop_event: asyncio.Event, timeout_s: float) -> None:
    """Sleep up to ``timeout_s``, returning early if ``stop_event`` is set."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        pass


async def run_worker(*, poll_interval_s: float = 10.0) -> None:
    """Boot the DB pool + LLM clients and run both consumer loops concurrently.

    Blocks until interrupted (SIGINT/SIGTERM → KeyboardInterrupt). A missing LLM
    pool is tolerated (extraction/forensics degrade), but a DB connection is
    mandatory — the whole point of this process is to consume the DB-backed
    ``crawl_queue`` and ``forensic_queue``.
    """
    from forensic_worker import run_worker_loop

    # main.py owns the shared pool/LLM bootstrap helpers.
    from main import _build_llm_clients, _init_db, _close_db

    try:
        settings.require_api_key()
    except RuntimeError as exc:
        log.error(str(exc))
        return

    llm, llm_pool = _build_llm_clients()

    db_pool = await _init_db()
    if db_pool is None:
        log.error("Cannot start worker: DB connection failed. crawl_queue lives in Postgres.")
        return

    stop_event = asyncio.Event()

    if _browser_available():
        crawl_task = asyncio.create_task(
            _crawl_consumer(db_pool, llm, llm_pool, stop_event=stop_event,
                            poll_idle_s=poll_interval_s),
            name="crawl-consumer",
        )
    else:
        crawl_task = asyncio.create_task(
            _crawl_consumer_static(db_pool, llm_pool, stop_event=stop_event),
            name="crawl-consumer-static",
        )

    forensic_task = asyncio.create_task(
        run_worker_loop(db_pool, llm_pool, settings, poll_interval_s=poll_interval_s),
        name="forensic-consumer",
    )

    log.header("KALA-BALANA WORKER — crawl + forensic consumers running (Ctrl-C to stop)")
    try:
        await asyncio.gather(crawl_task, forensic_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.warn("[Worker] Interrupted — shutting down gracefully.")
    finally:
        stop_event.set()
        for task in (crawl_task, forensic_task):
            task.cancel()
        await asyncio.gather(crawl_task, forensic_task, return_exceptions=True)
        await _close_db(db_pool)
        log.info("[Worker] Stopped.")


def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        log.warn("[Worker] Interrupted at top level — bye.")


if __name__ == "__main__":
    main()
