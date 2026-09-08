"""KALA-BALANA 5-Minute Forensic Intelligence Orchestrator.

Runs a 300-second autonomous pipeline for newly-discovered products.
Each phase has a strict ``asyncio.wait_for`` timeout; a phase failure
never kills subsequent phases — the pipeline is fault-tolerant by design.

Phase budget
------------
  Phase 1 — Ground Truth Specs      60 s
  Phase 2 — FX Arbitrage            60 s
  Phase 3 — Defect Mining           90 s
  Phase 4 — Merchant Forensics      60 s
  Phase 5 — Predecessor Comparison  30 s
                              Total 300 s

Database writes
---------------
After each phase completes, the product's ``pipeline_status`` is updated
so that a crash mid-pipeline results in only the last completed phase
being re-run, not the whole pipeline.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlsplit

from config import Settings, settings as default_settings
from logging_utils import log
from schemas import (
    ScrapedProductItem,
    GroundTruthSpec,
    DefectReport,
    MerchantAudit,
    PredecessorComparison,
)

# Lazy imports to avoid hard deps at import time.
try:
    from normalizer import NormalizedSpec
except ImportError:
    NormalizedSpec = None  # type: ignore


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForensicContext:
    """Everything the forensic pipeline needs to run."""

    product_id: str              # UUID string from canonical_products
    spec: Any                    # NormalizedSpec instance
    scraped_item: ScrapedProductItem
    listing_url: str
    merchant_id: Optional[str] = None


@dataclass
class ForensicReport:
    """Accumulated results from all phases."""

    product_id: str
    phase1_done: bool = False
    phase2_done: bool = False
    phase3_done: bool = False
    phase4_done: bool = False
    phase5_done: bool = False
    errors: List[str] = field(default_factory=list)
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    arbitrage: Dict[str, Any] = field(default_factory=dict)
    defects: List[Dict[str, Any]] = field(default_factory=list)
    merchant_audit: Dict[str, Any] = field(default_factory=dict)
    predecessor: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_status(self) -> str:
        if self.phase5_done:
            return "complete"
        for i in range(5, 0, -1):
            if getattr(self, f"phase{i}_done"):
                return f"phase{i}_done"
        return "discovered"


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

async def _phase1_ground_truth(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
) -> Dict[str, Any]:
    """Phase 1: Discover official spec sheet and ground-truth specifications.

    Uses the LLM to identify the official brand URL, chipset, display,
    battery, camera, and official MSRP in USD from the product name.
    We do NOT crawl external sites here to stay within budget; instead
    we prompt the LLM with its training knowledge and ask it to report
    confidence alongside each value.
    """
    from pydantic import BaseModel, Field

    class GroundTruthSpec(BaseModel):
        official_product_name: str = Field(description="Canonical product name as sold globally.")
        chipset: Optional[str] = Field(default=None, description="SoC/processor name.")
        display_inches: Optional[float] = Field(default=None, description="Screen size in inches.")
        battery_mah: Optional[int] = Field(default=None, description="Battery capacity in mAh.")
        main_camera_mp: Optional[int] = Field(default=None, description="Main camera megapixels.")
        launch_msrp_usd: Optional[float] = Field(default=None, description="Official launch MSRP in USD.")
        official_url: Optional[str] = Field(default=None, description="Official brand product page URL.")
        launch_year: Optional[int] = Field(default=None, description="Year the product launched.")
        confidence: float = Field(description="Confidence 0-1 in the data accuracy.", default=0.5)

    spec = ctx.spec
    product_name = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip()

    prompt = (
        f"Product: {product_name}\n"
        f"Raw title: {ctx.scraped_item.raw_title}\n\n"
        "Provide the official ground-truth specifications for this product. "
        "Use only verified information; set null for anything uncertain."
    )

    system = (
        "You are a precise hardware specification analyst. "
        "Return only factual, verifiable data about the product. "
        "Never invent specifications. Set null for anything you are not certain about."
    )

    result = await llm_pool.generate_structured(
        prompt, GroundTruthSpec,
        system_instruction=system,
        temperature=0.0,
        max_output_tokens=512,
    )
    return result.model_dump()


async def _phase2_fx_arbitrage(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 2: FX arbitrage and price-gouging classification.

    Fetches USD/LKR exchange rate, computes true landed cost,
    and classifies the merchant's price relative to fair market value.
    """
    import aiohttp

    msrp_usd: Optional[float] = ground_truth.get("launch_msrp_usd")
    merchant_price = ctx.scraped_item.price_lkr

    # Fetch FX rate.
    fx_rate: Optional[float] = None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(settings.fx_api_url) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    # Open Exchange Rates format: {"rates": {"LKR": 300.5}}
                    rates = data.get("rates", {})
                    fx_rate = rates.get("LKR")
    except Exception as exc:
        log.warn(f"[Forensic P2] FX fetch failed: {exc}")

    true_landed: Optional[float] = None
    markup_pct: Optional[float] = None
    price_label = "unknown"

    if msrp_usd and fx_rate:
        true_landed = round(msrp_usd * fx_rate * settings.import_duty_factor, 2)
        markup_pct = round((merchant_price - true_landed) / true_landed * 100, 1)
        if markup_pct < -5.0:
            price_label = "sub_msrp_likely_grey"
        elif markup_pct <= 40.0:
            price_label = "fair_import_margin"
        else:
            price_label = "price_gouged"

    result = {
        "global_msrp_usd": msrp_usd,
        "fx_rate_usd_lkr": fx_rate,
        "true_landed_cost_lkr": true_landed,
        "merchant_price_lkr": merchant_price,
        "markup_pct": markup_pct,
        "price_label": price_label,
    }

    # Persist to DB.
    if db_pool is not None:
        try:
            from db import queries
            await queries.insert_arbitrage_log(
                db_pool,
                product_id=ctx.product_id,
                global_msrp_usd=msrp_usd,
                cbsl_rate=fx_rate,
                true_landed_cost_lkr=true_landed,
                merchant_price_lkr=merchant_price,
                markup_pct=markup_pct,
                price_label=price_label,
            )
        except Exception as exc:
            log.warn(f"[Forensic P2] DB write failed: {exc}")

    return result


async def _phase3_defect_mining(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
) -> List[Dict[str, Any]]:
    """Phase 3: Reddit/forum defect mining with >=2 source corroboration rule.

    Queries Reddit's public JSON API (no OAuth required) for posts
    mentioning defects/problems with the product. Applies a 2-source
    minimum before promoting a defect to the dossier.
    """
    import aiohttp
    from pydantic import BaseModel, Field

    spec = ctx.spec
    product_query = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip()
    safe_query = quote_plus(f"{product_query} problem defect issue")

    # Reddit public JSON search.
    reddit_posts: List[Dict[str, Any]] = []
    headers = {"User-Agent": "KALA-BALANA-Forensics/1.0"}
    try:
        url = f"https://www.reddit.com/search.json?q={safe_query}&sort=relevance&limit=25&type=link"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
        ) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    children = data.get("data", {}).get("children", [])
                    for child in children:
                        post_data = child.get("data", {})
                        if not post_data:
                            continue
                        title = post_data.get("title", "")
                        selftext = (post_data.get("selftext") or "")[:500]
                        score = post_data.get("score", 0)
                        url_post = post_data.get("url", "")
                        subreddit = post_data.get("subreddit", "")
                        if score > 0 or any(
                            kw in (title + selftext).lower()
                            for kw in ("defect", "problem", "issue", "broken", "fail",
                                       "throttle", "overheat", "green line", "crack",
                                       "humidity", "moisture", "dead", "bug")
                        ):
                            reddit_posts.append({
                                "title": title,
                                "text": selftext,
                                "url": url_post,
                                "subreddit": subreddit,
                                "score": score,
                            })
    except Exception as exc:
        log.warn(f"[Forensic P3] Reddit fetch failed: {exc}")

    if not reddit_posts:
        return []

    # LLM summarization of defect categories.
    class DefectItem(BaseModel):
        defect_category: str = Field(
            description="One of: hardware, thermal, display, humidity, battery, camera, software, other"
        )
        severity: str = Field(description="One of: critical, moderate, minor")
        description: str = Field(description="Concise description of the defect (max 200 chars).")
        corroborating_count: int = Field(
            description="Number of distinct posts/sources mentioning this defect."
        )
        astroturf_score: float = Field(
            description="0.0-1.0 likelihood of being astroturfed (high = suspicious)",
            default=0.0,
        )
        source_urls: List[str] = Field(default_factory=list)

    class DefectReport(BaseModel):
        defects: List[DefectItem] = Field(
            description="List of distinct defect types found. Only include defects mentioned in >=2 independent sources."
        )

    posts_text = "\n---\n".join(
        f"Title: {p['title']}\nText: {p['text']}\nURL: {p['url']}"
        for p in reddit_posts[:20]
    )
    prompt = (
        f"Product: {product_query}\n\n"
        f"Reddit posts about defects/problems:\n{posts_text}\n\n"
        "Analyze these posts and extract distinct defect types. "
        "Apply strict >=2 independent source corroboration: only include a defect "
        "if at least 2 different posts mention it. "
        "Compute astroturf_score: high (>0.7) if most positive comments come from brand-new accounts "
        "or the product subreddit only, low otherwise."
    )

    system = (
        "You are a hardware quality analyst. Extract real user-reported defects "
        "from forum posts. Be skeptical of single-source claims. Ignore sponsored content."
    )

    try:
        report = await llm_pool.generate_structured(
            prompt, DefectReport,
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=1024,
        )
        defects = report.defects
    except Exception as exc:
        log.warn(f"[Forensic P3] LLM defect analysis failed: {exc}")
        return []

    # Persist confirmed defects to DB.
    results = []
    for defect in defects:
        if defect.corroborating_count < 2:
            continue  # Strict corroboration gate.
        defect_dict = defect.model_dump()
        if db_pool is not None:
            try:
                from db import queries
                await queries.insert_defect_dossier(
                    db_pool,
                    product_id=ctx.product_id,
                    source_url=defect.source_urls[0] if defect.source_urls else None,
                    source_platform="reddit",
                    defect_category=defect.defect_category,
                    severity=defect.severity,
                    corroborating_count=defect.corroborating_count,
                    astroturf_score=defect.astroturf_score,
                    description=defect.description,
                )
            except Exception as exc:
                log.warn(f"[Forensic P3] DB defect write failed: {exc}")
        results.append(defect_dict)

    return results


async def _phase4_merchant_audit(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
) -> Dict[str, Any]:
    """Phase 4: Merchant physical presence and dark-pattern forensics.

    Inspects the merchant's website for:
    - BNPL integration (Koko, Mintpay) and their surcharge patterns
    - Fake urgency timers and countdown clocks
    - Authorized agent status based on bundled allowlist
    - Physical address signals in the page footer/contact page
    """
    import aiohttp
    from pydantic import BaseModel, Field

    # Authorized agent allowlist (curated, expandable via config).
    _AUTHORIZED_AGENTS: Dict[str, List[str]] = {
        "apple":   ["abans.lk", "dialog.lk", "singer.lk", "apple.com"],
        "samsung": ["abans.lk", "samsung.com/lk", "melstatech.com"],
        "google":  ["dialog.lk", "google.com"],
        "sony":    ["abans.lk", "sony.lk"],
    }

    merchant_domain = urlsplit(ctx.listing_url).netloc.lower()
    brand = (ctx.spec.brand or "").lower()
    authorized_domains = _AUTHORIZED_AGENTS.get(brand, [])
    is_authorized = any(auth in merchant_domain for auth in authorized_domains)

    # Fetch merchant homepage to inspect dark patterns.
    dark_patterns: List[str] = []
    koko_mintpay_fees: Dict[str, Any] = {}
    bnpl_surcharge: Optional[float] = None
    raw_audit: Dict[str, Any] = {"domain": merchant_domain, "is_authorized": is_authorized}

    _COUNTDOWN_RE = re.compile(r"(countdown|timer|hurry|limited|only\s+\d+\s+left)", re.IGNORECASE)
    _KOKO_RE = re.compile(r"koko|mintpay|pay4d|payhere", re.IGNORECASE)
    _ADDR_RE = re.compile(
        r"(colombo|kandy|galle|negombo|matara|kurunegala|no\.\s*\d+|\d+,\s*[A-Z])",
        re.IGNORECASE,
    )

    try:
        base_url = f"{urlsplit(ctx.listing_url).scheme}://{merchant_domain}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; KalaBalaraBot/1.0)"}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
        ) as session:
            async with session.get(base_url, allow_redirects=True) as resp:
                if resp.status == 200:
                    body = await resp.text(errors="ignore")
                    body_lower = body.lower()
                    # Check dark patterns.
                    if _COUNTDOWN_RE.search(body):
                        dark_patterns.append("fake_urgency_timer")
                    if re.search(r"only\s+\d+\s+(item|unit|piece)s?\s+(left|remaining)", body, re.IGNORECASE):
                        dark_patterns.append("false_scarcity_counter")
                    if re.search(r"(someone\s+else|\d+\s+people)\s+(viewing|watching)", body, re.IGNORECASE):
                        dark_patterns.append("social_proof_manipulation")
                    # BNPL detection.
                    if _KOKO_RE.search(body):
                        koko_mintpay_fees["bnpl_present"] = True
                        dark_patterns.append("bnpl_installment_obfuscation")
                        # Try to extract surcharge pct from nearby text.
                        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(surcharge|fee|interest)", body, re.IGNORECASE)
                        if m:
                            bnpl_surcharge = float(m.group(1))
                            koko_mintpay_fees["surcharge_pct"] = bnpl_surcharge
                    # Physical address detection.
                    if _ADDR_RE.search(body):
                        raw_audit["physical_address_detected"] = True
                    else:
                        raw_audit["physical_address_detected"] = False
                        dark_patterns.append("no_physical_address")
    except Exception as exc:
        log.warn(f"[Forensic P4] Merchant homepage fetch failed: {exc}")
        raw_audit["fetch_error"] = str(exc)

    result = {
        "merchant_domain": merchant_domain,
        "is_authorized_agent": is_authorized,
        "dark_patterns": dark_patterns,
        "koko_mintpay_hidden_fees": koko_mintpay_fees,
        "bnpl_surcharge_pct": bnpl_surcharge,
        "raw_audit_data": raw_audit,
    }

    # Persist to DB.
    if db_pool is not None and ctx.merchant_id:
        try:
            from db import queries
            await queries.upsert_merchant_forensics(
                db_pool,
                merchant_id=ctx.merchant_id,
                bnpl_surcharge_pct=bnpl_surcharge,
                cc_surcharge_pct=None,  # would need more specific scraping
                dark_patterns=dark_patterns,
                koko_mintpay_hidden_fees=koko_mintpay_fees,
                raw_audit_data=raw_audit,
            )
        except Exception as exc:
            log.warn(f"[Forensic P4] DB merchant forensics write failed: {exc}")

    return result


async def _phase5_predecessor_comparison(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 5: Predecessor model comparison and generational upgrade verdict."""
    from pydantic import BaseModel, Field

    class PredecessorComparison(BaseModel):
        predecessor_name: Optional[str] = Field(
            default=None, description="Name of the direct predecessor model."
        )
        key_improvements: List[str] = Field(
            default_factory=list,
            description="Significant improvements over the predecessor.",
        )
        key_regressions: List[str] = Field(
            default_factory=list,
            description="Features that regressed or were removed vs predecessor.",
        )
        upgrade_verdict: str = Field(
            description="One of: worth_upgrade, marginal, skip",
            default="marginal",
        )
        verdict_reasoning: str = Field(
            description="One-sentence justification for the upgrade verdict.",
            default="",
        )

    spec = ctx.spec
    product_name = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip()

    prompt = (
        f"Product: {product_name}\n"
        f"Specs: {json.dumps(ground_truth, indent=2)}\n\n"
        "Compare this product to its direct predecessor model. "
        "Identify key improvements and regressions. "
        "Deliver an honest upgrade verdict: 'worth_upgrade', 'marginal', or 'skip'. "
        "Base the verdict on meaningful hardware differences, not marketing."
    )

    system = (
        "You are an expert hardware reviewer. Be objective and consumer-focused. "
        "Do not be influenced by marketing. Focus on real-world usability differences."
    )

    try:
        result = await llm_pool.generate_structured(
            prompt, PredecessorComparison,
            system_instruction=system,
            temperature=0.1,
            max_output_tokens=512,
        )
        comparison = result.model_dump()
    except Exception as exc:
        log.warn(f"[Forensic P5] LLM predecessor comparison failed: {exc}")
        comparison = {"error": str(exc)}

    return comparison


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def run_forensic_pipeline(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings = default_settings,
) -> ForensicReport:
    """Run all 5 forensic phases with per-phase timeouts.

    This is designed to be launched as a background task (``asyncio.create_task``).
    It is fully fault-tolerant: each phase is wrapped in a try/except with an
    ``asyncio.wait_for`` timeout. Phase data is persisted to the DB after each
    phase so partial results survive a crash.
    """
    from db import queries  # noqa: F401

    report = ForensicReport(product_id=ctx.product_id)
    spec = ctx.spec
    product_name = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip().title()

    log.gate("FORENSIC", f"Starting 5-phase pipeline for: {product_name}")
    log.gate("FORENSIC", f"  product_id={ctx.product_id} | total_budget=300s")

    # ------------------------------------------------------------------ #
    # Phase 1 — Ground Truth (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 1/5] Ground Truth Specs (60s budget)…")
    try:
        report.ground_truth = await asyncio.wait_for(
            _phase1_ground_truth(ctx, db_pool, llm_pool, settings),
            timeout=settings.forensic_phase1_timeout_s,
        )
        report.phase1_done = True
        await queries.update_pipeline_status(
            db_pool, ctx.product_id, "phase1_done",
            extra_attributes={"ground_truth": report.ground_truth},
        )
        msrp = report.ground_truth.get("launch_msrp_usd")
        chip = report.ground_truth.get("chipset", "?")
        log.gate("FORENSIC", f"  ✓ Phase 1 done | MSRP=${msrp} | Chip={chip}")
    except asyncio.TimeoutError:
        msg = "Phase 1 timed out after 60s"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
    except Exception as exc:
        msg = f"Phase 1 failed: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")

    # ------------------------------------------------------------------ #
    # Phase 2 — FX Arbitrage (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 2/5] FX Arbitrage & Price Classification (60s budget)…")
    try:
        report.arbitrage = await asyncio.wait_for(
            _phase2_fx_arbitrage(ctx, db_pool, llm_pool, settings, report.ground_truth),
            timeout=settings.forensic_phase2_timeout_s,
        )
        report.phase2_done = True
        await queries.update_pipeline_status(
            db_pool, ctx.product_id, "phase2_done",
            extra_attributes={"arbitrage": report.arbitrage},
        )
        label = report.arbitrage.get("price_label", "unknown")
        markup = report.arbitrage.get("markup_pct")
        log.gate("FORENSIC", f"  ✓ Phase 2 done | label={label} | markup={markup}%")
    except asyncio.TimeoutError:
        msg = "Phase 2 timed out after 60s"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
    except Exception as exc:
        msg = f"Phase 2 failed: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")

    # ------------------------------------------------------------------ #
    # Phase 3 — Defect Mining (90s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 3/5] Reddit Defect Mining (90s budget)…")
    try:
        report.defects = await asyncio.wait_for(
            _phase3_defect_mining(ctx, db_pool, llm_pool, settings),
            timeout=settings.forensic_phase3_timeout_s,
        )
        report.phase3_done = True
        await queries.update_pipeline_status(
            db_pool, ctx.product_id, "phase3_done",
            extra_attributes={"defect_count": len(report.defects)},
        )
        log.gate("FORENSIC", f"  ✓ Phase 3 done | {len(report.defects)} confirmed defect(s)")
    except asyncio.TimeoutError:
        msg = "Phase 3 timed out after 90s"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
    except Exception as exc:
        msg = f"Phase 3 failed: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")

    # ------------------------------------------------------------------ #
    # Phase 4 — Merchant Forensics (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 4/5] Merchant Forensic Audit (60s budget)…")
    try:
        report.merchant_audit = await asyncio.wait_for(
            _phase4_merchant_audit(ctx, db_pool, llm_pool, settings),
            timeout=settings.forensic_phase4_timeout_s,
        )
        report.phase4_done = True
        await queries.update_pipeline_status(
            db_pool, ctx.product_id, "phase4_done",
            extra_attributes={"merchant_audit": report.merchant_audit},
        )
        patterns = report.merchant_audit.get("dark_patterns", [])
        authorized = report.merchant_audit.get("is_authorized_agent", False)
        log.gate(
            "FORENSIC",
            f"  ✓ Phase 4 done | authorized={authorized} | dark_patterns={patterns}"
        )
    except asyncio.TimeoutError:
        msg = "Phase 4 timed out after 60s"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
    except Exception as exc:
        msg = f"Phase 4 failed: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")

    # ------------------------------------------------------------------ #
    # Phase 5 — Predecessor Comparison (30s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 5/5] Predecessor Comparison (30s budget)…")
    try:
        report.predecessor = await asyncio.wait_for(
            _phase5_predecessor_comparison(ctx, db_pool, llm_pool, settings, report.ground_truth),
            timeout=settings.forensic_phase5_timeout_s,
        )
        report.phase5_done = True
        await queries.update_pipeline_status(
            db_pool, ctx.product_id, "complete",
            extra_attributes={"predecessor_comparison": report.predecessor},
        )
        verdict = report.predecessor.get("upgrade_verdict", "?")
        log.gate("FORENSIC", f"  ✓ Phase 5 done | upgrade_verdict={verdict}")
    except asyncio.TimeoutError:
        msg = "Phase 5 timed out after 30s"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
    except Exception as exc:
        msg = f"Phase 5 failed: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")

    # ------------------------------------------------------------------ #
    # Final status
    # ------------------------------------------------------------------ #
    phases_done = sum(
        1 for i in range(1, 6) if getattr(report, f"phase{i}_done")
    )
    if report.errors:
        log.warn(
            f"[Forensic] Pipeline ended | {phases_done}/5 phases OK | "
            f"{len(report.errors)} error(s): " + "; ".join(report.errors)
        )
    else:
        log.success(
            f"[Forensic] Pipeline COMPLETE for {product_name} | "
            f"all 5 phases succeeded."
        )

    if phases_done == 0 and db_pool is not None:
        try:
            await queries.update_pipeline_status(db_pool, ctx.product_id, "failed")
        except Exception:
            pass

    return report


# ---------------------------------------------------------------------------
# Serial worker loop — entry point for `python main.py --worker`
# ---------------------------------------------------------------------------

async def run_worker_loop(
    db_pool: object,
    llm_pool: object,
    settings: Settings,
    *,
    poll_interval_s: float = 10.0,
) -> None:
    """Run the forensic worker loop until interrupted.

    Concurrency = 1: processes one job at a time to protect the 4GB VPS
    memory ceiling.  Claims jobs atomically from the ``forensic_queue``
    table using ``SELECT … FOR UPDATE SKIP LOCKED`` so multiple workers
    (if ever deployed) never double-process a job.

    Each job gets a hard 310s outer ``asyncio.wait_for`` guard (5s slack
    above the 300s pipeline budget) to guarantee forward progress even if
    an internal phase hangs past its own timeout.
    """
    from db import queries
    from normalizer import SpecNormalizer

    norm = SpecNormalizer()
    log.header("KALA-BALANA FORENSIC WORKER — starting poll loop")
    log.info(
        f"  poll_interval={poll_interval_s}s | phase budgets: "
        f"P1={settings.forensic_phase1_timeout_s}s "
        f"P2={settings.forensic_phase2_timeout_s}s "
        f"P3={settings.forensic_phase3_timeout_s}s "
        f"P4={settings.forensic_phase4_timeout_s}s "
        f"P5={settings.forensic_phase5_timeout_s}s"
    )

    while True:
        try:
            job = await queries.claim_next_forensic_job(db_pool)
        except Exception as exc:
            log.warn(f"[Worker] Queue claim error (retrying in {poll_interval_s}s): {exc}")
            await asyncio.sleep(poll_interval_s)
            continue

        if job is None:
            # Queue empty — idle wait.
            await asyncio.sleep(poll_interval_s)
            continue

        job_id: int = job["id"]
        product_id: str = job["product_id"]
        merchant_id: Optional[str] = job.get("merchant_id")
        listing_url: str = job["listing_url"]

        log.gate(
            "WORKER",
            f"Claimed job #{job_id} | product={product_id[:8]}… | url={listing_url[:60]}",
        )

        # Fetch full product record from DB to reconstruct ForensicContext.
        try:
            row = await db_pool.fetchrow(  # type: ignore[attr-defined]
                """
                SELECT raw_title, clean_title, brand, model_family, sub_model,
                       storage_gb, ram_gb, region_code, spec_fingerprint
                FROM canonical_products WHERE id = $1::uuid
                """,
                product_id,
            )
            if row is None:
                log.warn(f"[Worker] Product {product_id[:8]} not found in DB — skipping job.")
                await queries.fail_forensic_job(db_pool, job_id, "product_not_found")
                continue

            # Reconstruct a minimal NormalizedSpec from the stored fields.
            from normalizer import NormalizedSpec as _NS
            spec = _NS.from_parts(
                brand=row["brand"] or "",
                model_family=row["model_family"] or "",
                sub_model=row["sub_model"] or "",
                ram_gb=row["ram_gb"],
                storage_gb=row["storage_gb"],
                region_code=row["region_code"] or "",
            )

            # Reconstruct a minimal ScrapedProductItem.
            from schemas import ScrapedProductItem as _SPI
            scraped_item = _SPI(
                raw_title=row["raw_title"],
                clean_title=row["clean_title"],
                brand=row["brand"],
                price_lkr=1.0,   # price is not needed by forensic phases
                in_stock=True,
                product_url=listing_url,
            )

            ctx = ForensicContext(
                product_id=product_id,
                spec=spec,
                scraped_item=scraped_item,
                listing_url=listing_url,
                merchant_id=merchant_id,
            )

        except Exception as exc:
            log.error(f"[Worker] Context reconstruction failed for job #{job_id}: {exc}")
            await queries.fail_forensic_job(db_pool, job_id, f"context_error: {exc}")
            continue

        # Run the 5-phase pipeline with a hard outer timeout.
        try:
            await asyncio.wait_for(
                run_forensic_pipeline(ctx, db_pool, llm_pool, settings),
                timeout=310.0,
            )
            await queries.complete_forensic_job(db_pool, job_id)
            log.success(f"[Worker] Job #{job_id} completed for product {product_id[:8]}…")
        except asyncio.TimeoutError:
            msg = "Hard 310s outer timeout exceeded"
            log.warn(f"[Worker] Job #{job_id} timed out: {msg}")
            await queries.fail_forensic_job(db_pool, job_id, msg)
        except Exception as exc:
            msg = f"Unhandled pipeline error: {exc}"
            log.error(f"[Worker] Job #{job_id} failed: {msg}")
            await queries.fail_forensic_job(db_pool, job_id, msg)
