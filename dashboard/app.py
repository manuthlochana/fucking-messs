"""KALA-BALANA Operational & Forensic Admin Dashboard.

Built with FastAPI + Jinja2 + Tailwind CSS (via CDN).
Zero Node/React build bloat. Lightweight and memory-efficient.

Security
--------
When ``ADMIN_DASHBOARD_KEY`` is set, every route requires HTTP Basic auth
(username ``ADMIN_DASHBOARD_USER``, default ``admin``; password = the key) and
the state-changing POST endpoints additionally require a CSRF token equal to the
key, supplied as the ``csrf_token`` form field or the ``X-CSRF-Token`` header.
When the key is unset the dashboard runs UNAUTHENTICATED (a loud warning is
logged) — intended only for a trusted localhost session.
"""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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

# ---------------------------------------------------------------------------
# Authentication & CSRF
# ---------------------------------------------------------------------------
_basic = HTTPBasic(auto_error=False)


def require_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic),
) -> Optional[str]:
    """HTTP Basic gate. No-op (open) when no admin key is configured.

    Uses constant-time comparison to avoid leaking the key via timing.
    """
    key = settings.admin_dashboard_key
    if not key:
        # Dashboard is open — acceptable only for trusted localhost use.
        return None
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, settings.admin_dashboard_user)
    pass_ok = secrets.compare_digest(credentials.password, key)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def require_csrf(
    csrf_token: Optional[str] = Form(default=None),
    x_csrf_token: Optional[str] = Header(default=None),
) -> None:
    """Reject state-changing requests lacking a valid CSRF token.

    The token equals the admin key and may arrive as the ``csrf_token`` form
    field or the ``X-CSRF-Token`` header. Skipped entirely when no key is set.
    """
    key = settings.admin_dashboard_key
    if not key:
        return
    supplied = csrf_token or x_csrf_token or ""
    if not secrets.compare_digest(supplied, key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing or invalid CSRF token.",
        )


def _csrf_value() -> str:
    """Token value to embed in server-rendered forms (empty when auth off)."""
    return settings.admin_dashboard_key or ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    if not settings.admin_dashboard_key:
        log.warn(
            "[Dashboard] ADMIN_DASHBOARD_KEY is not set — dashboard is UNAUTHENTICATED. "
            "Set it before exposing the dashboard beyond localhost."
        )
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
    # Every route requires auth when a key is configured.
    dependencies=[Depends(require_auth)],
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
            "csrf_token": _csrf_value(),
        },
    )


@app.post("/ingest", dependencies=[Depends(require_csrf)])
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
            "csrf_token": _csrf_value(),
        },
    )



@app.post("/queue/retry/{job_id}", dependencies=[Depends(require_csrf)])
async def retry_queue_job(job_id: int):
    await queries.retry_failed_forensic_job(db_pool, job_id)
    return RedirectResponse(url="/queue", status_code=303)


@app.get("/api/stats")
async def api_stats():
    stats = await queries.get_dashboard_stats(db_pool)
    return stats


# ---------------------------------------------------------------------------
# Autonomous crawl monitor + domain ingestion (Step 6)
# ---------------------------------------------------------------------------

@app.get("/crawl", response_class=HTMLResponse)
async def crawl_monitor_page(request: Request, message: Optional[str] = None):
    domains = await queries.list_crawl_domains(db_pool)
    return templates.TemplateResponse(
        request=request,
        name="crawl.html",
        context={
            "active_page": "crawl",
            "domains": domains,
            "message": message,
            "csrf_token": _csrf_value(),
        },
    )


@app.post("/api/domains/add", dependencies=[Depends(require_csrf)])
async def add_domain(
    background_tasks: BackgroundTasks,
    domain: str = Form(...),
    daily_page_limit: int = Form(1500),
):
    """Kick off autonomous catalog discovery for a domain, then drip-crawl it."""
    domain = domain.strip()
    if not domain:
        raise HTTPException(status_code=400, detail="Domain URL is required.")

    async def _run_ingest_bg(url: str, limit: int):
        import discovery
        from llm_pool import MultiKeyLLMPool

        try:
            summary = await discovery.ingest_domain(
                url, db_pool, daily_page_limit=limit, settings=settings
            )
            log.info(
                f"[Dashboard] Ingested {summary.get('domain')}: "
                f"discovered={summary.get('discovered')} enqueued={summary.get('enqueued')}"
            )
            llm_pool = None
            try:
                llm_pool = MultiKeyLLMPool.from_settings(settings)
            except Exception:
                pass
            await discovery.run_drip_worker(
                db_pool, domain=summary.get("domain"), settings=settings,
                llm_pool=llm_pool, daily_page_limit=limit,
            )
        except Exception as exc:
            log.error(f"[Dashboard] Domain ingest failed for {url}: {exc}")

    background_tasks.add_task(_run_ingest_bg, domain, daily_page_limit)
    return RedirectResponse(
        url=f"/crawl?message=Discovery+launched+for+{domain}.+The+drip+crawler+will+scrape+the+catalog+over+the+coming+hours.",
        status_code=303,
    )


@app.post("/api/chat")
async def api_chat(payload: Dict[str, Any]):
    """Advisor RAG endpoint — grounded shopping advice over completed dossiers."""
    query = (payload.get("query") or payload.get("message") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="A 'query' field is required.")
    product_id = payload.get("product_id")

    from advisor import advise
    from llm_pool import MultiKeyLLMPool

    llm_pool = None
    try:
        llm_pool = MultiKeyLLMPool.from_settings(settings)
    except Exception:
        pass

    rec = await advise(
        query, db_pool=db_pool, llm_pool=llm_pool, settings=settings,
        product_id=product_id,
    )
    return rec.model_dump()


def start_dashboard(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch the dashboard server via uvicorn."""
    import uvicorn
    log.header(f"KALA-BALANA DASHBOARD running on http://{host}:{port}")
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=False)
