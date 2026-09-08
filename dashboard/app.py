"""KALA-BALANA Operational & Forensic Admin Dashboard.

Built with FastAPI + Jinja2 + Tailwind CSS (via CDN).
Zero Node/React build bloat. Lightweight and memory-efficient.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import settings
from db import queries
from db.pool import close_pool, create_pool
from logging_utils import log

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Shared DB pool for the dashboard process
db_pool: Optional[object] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        db_pool = await create_pool(
            settings.db_dsn,
            max_size=settings.db_pool_max_size,
            min_size=settings.db_pool_min_size,
        )
        if db_pool:
            from db.migrations import run_migrations
            await run_migrations(db_pool)
            log.info("[Dashboard] DB connection established.")
    except Exception as exc:
        log.warn(f"[Dashboard] DB initialization failed (dry-run mode): {exc}")
        db_pool = None
    yield
    if db_pool:
        await close_pool(db_pool)
        log.info("[Dashboard] DB connection closed.")


app = FastAPI(
    title="KALA-BALANA Hardware Intelligence Dashboard",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def overview_page(request: Request):
    stats = await queries.get_dashboard_stats(db_pool)
    recent_products = await queries.get_catalog_products(db_pool, limit=6)
    recent_jobs = await queries.get_forensic_queue_jobs(db_pool, limit=6)
    active_job = next((j for j in recent_jobs if j.get("status") == "claimed"), None)

    return templates.TemplateResponse(
        request=request,
        name="overview.html",
        context={
            "active_page": "overview",
            "stats": stats,
            "recent_products": recent_products,
            "recent_jobs": recent_jobs,
            "active_job": active_job,
        },
    )


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request, message: Optional[str] = None):
    return templates.TemplateResponse(
        request=request,
        name="ingest.html",
        context={
            "active_page": "ingest",
            "message": message,
        },
    )


@app.post("/ingest")
async def ingest_urls(
    background_tasks: BackgroundTasks,
    urls: str = Form(...),
):
    url_list = [u.strip() for u in urls.splitlines() if u.strip() and not u.strip().startswith("#")]
    if not url_list:
        raise HTTPException(status_code=400, detail="No valid URLs submitted.")

    # Run crawl in background task so the web request returns immediately
    async def _run_crawl_bg(targets: list[str]):
        from crawler import AntiFragileCrawler
        from llm_pool import GeminiClient, MultiKeyLLMPool
        from sentry import SmartSentry

        try:
            llm = GeminiClient(settings)
            llm_pool = MultiKeyLLMPool.from_settings(settings)
            sentry = SmartSentry(llm, settings)
            crawler = AntiFragileCrawler(sentry, llm, settings, db_pool=db_pool, llm_pool=llm_pool)
            await crawler.crawl_batch(targets)
        except Exception as exc:
            log.error(f"[Dashboard Ingest] Background crawl failed: {exc}")

    background_tasks.add_task(_run_crawl_bg, url_list)
    return RedirectResponse(
        url=f"/ingest?message=Crawl+launched+for+{len(url_list)}+URL(s).+Discovered+products+will+be+enqueued+automatically.",
        status_code=303,
    )


@app.get("/catalog", response_class=HTMLResponse)
async def catalog_page(request: Request, q: Optional[str] = None):
    products = await queries.get_catalog_products(db_pool, limit=100, query=q)
    return templates.TemplateResponse(
        request=request,
        name="catalog.html",
        context={
            "active_page": "catalog",
            "products": products,
            "query": q,
        },
    )


@app.get("/product/{product_id}", response_class=HTMLResponse)
async def product_detail_page(request: Request, product_id: str):
    data = await queries.get_product_detail_with_dossier(db_pool, product_id)
    if not data:
        raise HTTPException(status_code=404, detail="Product not found.")

    return templates.TemplateResponse(
        request=request,
        name="product_detail.html",
        context={
            "active_page": "catalog",
            "product": data["product"],
            "attributes": data["attributes"],
            "listings": data["listings"],
            "defect_dossier": data["defect_dossier"],
            "arbitrage_logs": data["arbitrage_logs"],
            "teardowns": data["teardowns"],
        },
    )


@app.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request, status: Optional[str] = None):
    jobs = await queries.get_forensic_queue_jobs(db_pool, limit=100, status=status)
    return templates.TemplateResponse(
        request=request,
        name="queue.html",
        context={
            "active_page": "queue",
            "jobs": jobs,
            "current_status": status,
        },
    )



@app.post("/queue/retry/{job_id}")
async def retry_queue_job(job_id: int):
    await queries.retry_failed_forensic_job(db_pool, job_id)
    return RedirectResponse(url="/queue", status_code=303)


@app.get("/api/stats")
async def api_stats():
    stats = await queries.get_dashboard_stats(db_pool)
    return stats


def start_dashboard(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch the dashboard server via uvicorn."""
    import uvicorn
    log.header(f"KALA-BALANA DASHBOARD running on http://{host}:{port}")
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=False)
