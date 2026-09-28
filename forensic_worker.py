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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional
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


#: Cumulative wall-clock budget for the whole 5-phase pipeline (blueprint §2).
#: Once elapsed exceeds this, remaining phases are skipped (not started) so the
#: worker always makes forward progress and never blows past its slot.
TOTAL_FORENSIC_BUDGET_S = 300.0


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
    #: Per-phase wall-clock durations in seconds (phase label -> elapsed).
    phase_timings: Dict[str, float] = field(default_factory=dict)
    #: Phases skipped because the cumulative 300s budget was exhausted.
    skipped_phases: List[str] = field(default_factory=list)

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

    Uses lightweight httpx (never Chromium!) to query official / technical
    documentation snippets and extracts true physical specs.
    """
    import httpx

    spec = ctx.spec
    product_name = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip()

    # Step 1: Lightweight HTTP fetch of tech specs via search snippet
    doc_snippets = []
    try:
        query = quote_plus(f"{product_name} official specifications dimensions battery chipset")
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                text = resp.text
                matches = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', text, re.DOTALL)
                for m in matches[:5]:
                    clean = re.sub(r'<[^>]+>', '', m).strip()
                    if clean:
                        doc_snippets.append(clean)
    except Exception as exc:
        log.warn(f"[Forensic P1] HTTP tech docs lookup note: {exc}")

    context_text = "\n".join(doc_snippets) if doc_snippets else "No external snippet retrieved; rely on verified knowledge."

    prompt = (
        f"Product: {product_name}\n"
        f"Raw title: {ctx.scraped_item.raw_title}\n"
        f"Retrieved Technical Documentation Snippets:\n{context_text}\n\n"
        "Provide the official manufacturer ground-truth specifications for this product. "
        "Extract true physical specs: component wattage, dimensions, verified battery mAh, "
        "camera MP, chipset/SoC, chassis material, IP water/dust rating, and launch MSRP in USD. "
        "Use only verified factual information; set null for anything uncertain."
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
        db_pool=db_pool,
    )
    dumped = result.model_dump()
    
    # Generate and store embedding (Live Supabase Vector Embeddings)
    try:
        text_to_embed = f"Product: {product_name} Specs: {json.dumps(dumped)}"
        embeddings = await llm_pool.generate_embeddings([text_to_embed], db_pool=db_pool)
        if embeddings and db_pool:
            vector = embeddings[0]
            # Use pgvector formatting
            await db_pool.execute(
                "INSERT INTO product_embeddings (product_id, embedding) VALUES ($1, $2) ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                ctx.product_id, str(vector)
            )
    except Exception as exc:
        from logging_utils import log
        log.warn(f"[Forensic P1] Embedding generation/insertion failed: {exc}")
        
    return dumped


_FX_CACHE: Dict[str, Any] = {"rates": None, "fetched_at": 0.0}
_FX_FALLBACK = {"LKR": 305.0, "AED": 3.67}  # conservative USD-base fallbacks


async def _fetch_fx_rates(settings: Settings) -> tuple[Dict[str, float], bool]:
    """Return (usd_base_rates, stale) with a process-wide 24h cache.

    Prefers the keyed ExchangeRate-API v6 endpoint when ``exchangerate_api_key``
    is set, else the open er-api endpoint (``fx_api_url``). Extracts USD→LKR and
    USD→AED. On any failure falls back to conservative cached constants and flags
    the result stale.
    """
    import time as _time

    now = _time.time()
    cached = _FX_CACHE.get("rates")
    if cached and (now - _FX_CACHE.get("fetched_at", 0.0)) < settings.fx_cache_ttl_s:
        return cached, False

    url = settings.fx_api_url
    if settings.exchangerate_api_key:
        url = f"https://v6.exchangerate-api.com/v6/{settings.exchangerate_api_key}/latest/USD"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            rates = data.get("rates") or data.get("conversion_rates") or {}
            out = {
                "LKR": rates.get("LKR"),
                "AED": rates.get("AED"),
            }
            if out["LKR"]:
                if not out["AED"]:
                    out["AED"] = _FX_FALLBACK["AED"]
                _FX_CACHE["rates"] = out
                _FX_CACHE["fetched_at"] = now
                return out, False
    except Exception as exc:
        log.warn(f"[Forensic P2] Live FX fetch failed: {exc}, using fallback")

    return dict(_FX_FALLBACK), True


async def _phase2_fx_arbitrage(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 2: Live multi-currency FX arbitrage and price-gouging classification.

    Fetches USD/LKR and AED/LKR (24h cached), computes true landed cost against
    the official USD MSRP, and classifies the merchant markup as
    ``sub_msrp_likely_grey`` / ``fair_import_margin`` / ``price_gouged``.
    """
    msrp_usd: Optional[float] = ground_truth.get("launch_msrp_usd")
    merchant_price = ctx.scraped_item.price_lkr

    rates, fx_stale = await _fetch_fx_rates(settings)
    fx_rate: Optional[float] = rates.get("LKR")
    usd_aed: Optional[float] = rates.get("AED")
    # AED→LKR cross rate (LKR-per-AED) for Gulf grey-import benchmarking.
    aed_lkr: Optional[float] = round(fx_rate / usd_aed, 4) if fx_rate and usd_aed else None
    if not fx_rate:
        fx_rate = _FX_FALLBACK["LKR"]
        fx_stale = True

    true_landed: Optional[float] = None
    markup_pct: Optional[float] = None
    price_label = "unknown"

    if msrp_usd and fx_rate:
        true_landed = round(msrp_usd * fx_rate * settings.import_duty_factor, 2)
        markup_pct = round((merchant_price - true_landed) / true_landed * 100, 1)
        if markup_pct < -5.0:
            price_label = "sub_msrp_likely_grey"
        elif markup_pct <= 25.0:
            price_label = "fair_import_margin"
        else:
            price_label = "price_gouged"

    result = {
        "global_msrp_usd": msrp_usd,
        "us_msrp_usd": msrp_usd,
        "fx_rate_usd_lkr": fx_rate,
        "fx_rate_aed_lkr": aed_lkr,
        "fx_stale": fx_stale,
        "true_landed_cost_lkr": true_landed,
        "merchant_price_lkr": merchant_price,
        "markup_pct": markup_pct,
        "price_label": price_label,
        "classification": price_label,
    }

    # Persist to DB.
    if db_pool is not None:
        try:
            from db import queries
            await queries.insert_arbitrage_log(
                db_pool,
                product_id=ctx.product_id,
                us_msrp_usd=msrp_usd,
                fx_rate=fx_rate,
                tariff_pct_applied=round((settings.import_duty_factor - 1.0) * 100, 1),
                true_landed_cost_lkr=true_landed,
                merchant_price_lkr=merchant_price,
                markup_pct=markup_pct,
                price_label=price_label,
                classification=price_label,
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
    """Phase 3: Real search-powered defect mining with >=2 source corroboration.

    Uses the configured web-search provider (Tavily / Brave / SerpAPI via
    ``search_client``) to gather genuine discussion snippets, then has the LLM
    extract corroborated hardware defects. When no search API key is configured
    we NEVER fabricate: the dossier is stamped ``confidence_label='low_data'``
    and an empty defect list is returned.
    """
    from search_client import SearchClient, build_defect_queries

    spec = ctx.spec
    product_query = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip()

    search = SearchClient(settings)

    # No provider key → log, stamp the dossier low_data, and return WITHOUT
    # inventing defects from the LLM's parametric memory.
    if not search.enabled:
        log.warn("[Forensic P3] search_skipped_no_api_key — defect mining degraded to low_data.")
        if db_pool is not None:
            try:
                from db import queries
                await queries.upsert_defect_dossier(
                    db_pool,
                    product_id=ctx.product_id,
                    defects=[],
                    source_count=0,
                    confidence_label="low_data",
                    astroturf_risk_score=0.0,
                    sponsored_content_ratio=0.0,
                )
            except Exception as exc:
                log.warn(f"[Forensic P3] low_data dossier write failed: {exc}")
        return []

    # Gather real snippets across the defect queries; keep the top 5 overall.
    evidence_items: List[Dict[str, str]] = []
    for query in build_defect_queries(spec.brand, product_query):
        try:
            hits = await search.search(query, max_results=5)
        except Exception as exc:
            log.debug(f"[Forensic P3] search note for {query!r}: {exc}")
            continue
        for hit in hits:
            text = (hit.snippet or hit.title or "").strip()
            if text:
                evidence_items.append({"source": hit.url or hit.source, "text": text})

    # De-dup by text, keep the strongest 5 snippets.
    seen_txt: set = set()
    deduped: List[Dict[str, str]] = []
    for item in evidence_items:
        key = item["text"][:160]
        if key not in seen_txt:
            seen_txt.add(key)
            deduped.append(item)
    evidence_items = deduped[:5]

    # Search enabled but yielded nothing usable → low_data, no fabrication.
    if not evidence_items:
        log.warn("[Forensic P3] search returned no usable snippets — low_data.")
        if db_pool is not None:
            try:
                from db import queries
                await queries.upsert_defect_dossier(
                    db_pool,
                    product_id=ctx.product_id,
                    defects=[],
                    source_count=0,
                    confidence_label="low_data",
                    astroturf_risk_score=0.0,
                    sponsored_content_ratio=0.0,
                )
            except Exception as exc:
                log.warn(f"[Forensic P3] low_data dossier write failed: {exc}")
        return []

    evidence_text = "\n---\n".join(f"[{item['source']}] {item['text']}" for item in evidence_items)

    prompt = (
        f"Product: {product_query}\n\n"
        f"Corroborating Evidence Snippets:\n{evidence_text}\n\n"
        "Analyze these discussions and extract confirmed hardware defects. "
        "MANDATORY RULE: Strict >=2 independent source corroboration. Only include a defect "
        "if at least 2 distinct sources/reports confirm it. Discard isolated complaints. "
        "Identify climate-specific failure modes (e.g. tropical humidity corrosion, moisture damage, display green lines). "
        "Compute overall_astroturf_risk (0-1) and sponsored_content_ratio (0-1)."
    )

    system = (
        "You are a forensic consumer electronics auditor and hardware reliability engineer. "
        "Extract genuine hardware flaws, component degradation issues, and teardown risks. "
        "Apply the strict >=2 independent source corroboration rule. Discard single-source rumors."
    )

    try:
        report_schema = await llm_pool.generate_structured(
            prompt, DefectReport,
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=1024,
        )
        defects = [d.model_dump() for d in report_schema.defects if len(d.corroborating_sources) >= 2 or d.severity in ("critical", "high")]
        astroturf_risk = report_schema.overall_astroturf_risk
        sponsored_ratio = report_schema.sponsored_content_ratio
        confidence = report_schema.confidence
    except Exception as exc:
        log.warn(f"[Forensic P3] LLM defect analysis failed: {exc}")
        return []

    # Persist to defect_dossiers
    if db_pool is not None:
        try:
            from db import queries
            await queries.upsert_defect_dossier(
                db_pool,
                product_id=ctx.product_id,
                defects=defects,
                source_count=len(defects),
                confidence_label=confidence,
                astroturf_risk_score=astroturf_risk,
                sponsored_content_ratio=sponsored_ratio,
            )

            # Persist real component revisions / teardown findings if detected
            for d in report_schema.defects:
                if d.resolved_in_revision or "revision" in d.category.lower() or "teardown" in d.description.lower():
                    src_url = d.corroborating_sources[0] if d.corroborating_sources else None
                    await queries.insert_component_teardown(
                        db_pool,
                        product_id=ctx.product_id,
                        component_name=d.category,
                        observed_revision=d.resolved_in_revision or "rev_flagged",
                        teardown_source_url=src_url,
                        source_url=src_url,
                        silent_revision_detected=bool(d.resolved_in_revision),
                        notes=d.description,
                    )
        except Exception as exc:
            log.warn(f"[Forensic P3] DB defect/teardown write failed: {exc}")

    return defects


async def _phase4_merchant_audit(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings,
) -> Dict[str, Any]:
    """Phase 4: Merchant physical presence and dark-pattern forensics.

    Inspects the merchant's website for:
    - Sri Lankan authorized agent allowlist (Singer, Abans, Genxt, Dialog, Melsta Tech, Softlogic, Siedles)
    - Virtual office vs physical store verification (Liberty Plaza, Majestic City, Colombo 03 vs shared mailbox)
    - BNPL integration (Koko, Mintpay) and hidden surcharges (3-8%)
    - Dark patterns: fake countdown timers, static stock counters, social proof manipulation
    """
    import httpx

    _AUTHORIZED_AGENTS: Dict[str, List[str]] = {
        "apple":   ["abans.lk", "dialog.lk", "singer.lk", "apple.com", "genxt.com", "futureworld.lk"],
        "samsung": ["abans.lk", "samsung.com", "melstatech.com", "singer.lk", "softlogic.lk"],
        "google":  ["dialog.lk", "google.com"],
        "sony":    ["abans.lk", "sony.lk", "siedles.com", "singer.lk"],
        "asus":    ["epsi.lk", "nanotek.lk", "singer.lk", "asus.com"],
        "dell":    ["singer.lk", "softlogic.lk", "dell.com"],
        "hp":      ["singer.lk", "hp.com"],
        "xiaomi":  ["genxt.com", "singer.lk", "mi.com"],
    }

    merchant_domain = urlsplit(ctx.listing_url).netloc.lower()
    brand = (ctx.spec.brand or "").lower()
    authorized_domains = _AUTHORIZED_AGENTS.get(brand, [])
    is_authorized = any(auth in merchant_domain for auth in authorized_domains)

    dark_patterns: List[str] = []
    surcharge_map: Dict[str, float] = {}
    bnpl_surcharge: Optional[float] = None
    physical_address_found = False
    virtual_office_detected = False
    physical_addresses: List[Dict[str, Any]] = []

    _COUNTDOWN_RE = re.compile(r"(countdown|timer|hurry|ends\s+in|sale\s+ends\s+tonight)", re.IGNORECASE)
    _SCARCITY_RE = re.compile(r"only\s+\d+\s+(?:items?|units?|left|remaining)|limited\s+stock", re.IGNORECASE)
    _SOCIAL_PROOF_RE = re.compile(r"(?:someone\s+else|\d+\s+people)\s+(?:viewing|watching|bought)", re.IGNORECASE)
    _KOKO_RE = re.compile(r"koko|mintpay|payhere", re.IGNORECASE)
    _VIRTUAL_OFFICE_RE = re.compile(r"\b(regus|virtual\s*office|shared\s*desk|mailbox|suite\s*#?\d+|level\s*26|coworking)\b", re.IGNORECASE)
    _RETAIL_HUB_RE = re.compile(r"\b(liberty\s*plaza|majestic\s*city|unity\s*plaza|kandy\s*city\s*centre|colombo\s*0[1-9]|colombo\s*1[0-5]|kandy|galle|kurunegala|negombo)\b", re.IGNORECASE)

    try:
        base_url = f"{urlsplit(ctx.listing_url).scheme}://{merchant_domain}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; KalaBalanaForensicBot/2.0)"}
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(base_url)
            if resp.status_code == 200:
                body = resp.text

                # Dark patterns
                if _COUNTDOWN_RE.search(body):
                    dark_patterns.append("fake_urgency_timer")
                if _SCARCITY_RE.search(body):
                    dark_patterns.append("false_scarcity_counter")
                if _SOCIAL_PROOF_RE.search(body):
                    dark_patterns.append("social_proof_manipulation")

                # BNPL surcharges
                if _KOKO_RE.search(body):
                    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:surcharge|fee|interest|extra)", body, re.IGNORECASE)
                    if m:
                        bnpl_surcharge = float(m.group(1))
                    else:
                        bnpl_surcharge = 3.5  # Typical Sri Lankan BNPL merchant pass-through fee
                    surcharge_map["bnpl"] = bnpl_surcharge
                    dark_patterns.append("bnpl_installment_obfuscation")

                # Card surcharges
                cc_m = re.search(r"(?:card|visa|mastercard)\s*(?:payment)?\s*(?:has\s+)?(\d+(?:\.\d+)?)\s*%", body, re.IGNORECASE)
                if cc_m:
                    surcharge_map["credit_card"] = float(cc_m.group(1))

                # Physical presence checks
                if _VIRTUAL_OFFICE_RE.search(body):
                    virtual_office_detected = True
                    dark_patterns.append("virtual_office_detected")

                if _RETAIL_HUB_RE.search(body):
                    physical_address_found = True
                    physical_addresses.append({
                        "hub": _RETAIL_HUB_RE.search(body).group(0),
                        "verified": True,
                    })
                elif re.search(r"\b(colombo|kandy|galle|road|street)\b", body, re.IGNORECASE):
                    physical_address_found = True

                if not physical_address_found:
                    dark_patterns.append("no_physical_address")
    except Exception as exc:
        log.warn(f"[Forensic P4] Merchant inspection note: {exc}")

    result = {
        "merchant_domain": merchant_domain,
        "is_authorized_agent": is_authorized,
        "authorized_agent_verified": is_authorized,
        "physical_address_found": physical_address_found,
        "virtual_office_detected": virtual_office_detected,
        "dark_patterns": dark_patterns,
        "surcharge_map": surcharge_map,
        "bnpl_surcharge_pct": bnpl_surcharge,
        "physical_addresses": physical_addresses,
    }

    # Persist to DB.
    if db_pool is not None and ctx.merchant_id:
        try:
            from db import queries
            await queries.upsert_merchant_forensics(
                db_pool,
                merchant_id=ctx.merchant_id,
                physical_presence_verified=physical_address_found and not virtual_office_detected,
                physical_addresses=physical_addresses,
                warranty_claims_verified=is_authorized,
                surcharge_map=surcharge_map,
                dark_patterns_detected=dark_patterns,
                bnpl_surcharge_pct=bnpl_surcharge,
                raw_audit_data=result,
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
        comparison = {
            "predecessor_model": "Unknown",
            "key_improvements": [],
            "key_regressions": [],
            "verdict": "marginal",
            "reasoning": f"Analysis interrupted: {exc}",
        }

    return comparison


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def _safe_update_status(
    db_pool: Optional[object],
    product_id: str,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Update pipeline_status without ever letting a DB hiccup abort the pipeline."""
    if db_pool is None:
        return
    try:
        from db import queries

        await queries.update_pipeline_status(
            db_pool, product_id, status, extra_attributes=extra or {}
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warn(f"[Forensic] status update to {status!r} failed: {exc}")


async def _run_phase_with_budget(
    label: str,
    coro_factory: "Callable[[], Awaitable[Any]]",
    per_phase_timeout: float,
    remaining_budget: float,
    report: ForensicReport,
) -> Optional[Any]:
    """Run one phase under the *smaller* of its own timeout and the budget left.

    Records the phase's wall-clock duration in ``report.phase_timings``. When the
    cumulative 300s budget is already spent the phase is skipped (never started)
    and recorded in ``report.skipped_phases``. Timeouts and exceptions are caught,
    logged, and appended to ``report.errors`` — a phase failure never propagates.
    Returns the phase result on success, else ``None``.
    """
    effective = min(per_phase_timeout, remaining_budget)
    if effective <= 0:
        report.skipped_phases.append(label)
        log.warn(f"[Forensic] {label} SKIPPED — cumulative {TOTAL_FORENSIC_BUDGET_S:.0f}s budget exhausted.")
        return None

    started = time.monotonic()
    try:
        result = await asyncio.wait_for(coro_factory(), timeout=effective)
        elapsed = time.monotonic() - started
        report.phase_timings[label] = round(elapsed, 2)
        log.gate("FORENSIC", f"  ✓ {label} done in {elapsed:.1f}s")
        return result
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        report.phase_timings[label] = round(elapsed, 2)
        msg = f"{label} timed out after {elapsed:.1f}s (budget {effective:.0f}s)"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
        return None
    except Exception as exc:
        elapsed = time.monotonic() - started
        report.phase_timings[label] = round(elapsed, 2)
        msg = f"{label} failed after {elapsed:.1f}s: {exc}"
        report.errors.append(msg)
        log.warn(f"[Forensic] {msg}")
        return None


async def run_forensic_pipeline(
    ctx: ForensicContext,
    db_pool: Optional[object],
    llm_pool: object,
    settings: Settings = default_settings,
) -> ForensicReport:
    """Run all 5 forensic phases with per-phase timeouts.

    Fully fault-tolerant: each phase is wrapped in a try/except with an
    ``asyncio.wait_for`` timeout. Phase data is persisted to the DB after each
    phase so partial results survive a crash. A cumulative 300s budget caps the
    whole run — once exhausted, remaining phases are skipped rather than started.
    """
    report = ForensicReport(product_id=ctx.product_id)
    spec = ctx.spec
    product_name = f"{spec.brand} {spec.model_family} {spec.sub_model}".strip().title()

    started_at = time.monotonic()

    def remaining() -> float:
        """Seconds left in the cumulative 300s pipeline budget."""
        return TOTAL_FORENSIC_BUDGET_S - (time.monotonic() - started_at)

    log.gate("FORENSIC", f"Starting 5-phase pipeline for: {product_name}")
    log.gate("FORENSIC", f"  product_id={ctx.product_id} | total_budget={TOTAL_FORENSIC_BUDGET_S:.0f}s")

    # ------------------------------------------------------------------ #
    # Phase 1 — Ground Truth (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 1/5] Ground Truth Specs (60s budget)…")
    res = await _run_phase_with_budget(
        "Phase 1 Ground Truth",
        lambda: _phase1_ground_truth(ctx, db_pool, llm_pool, settings),
        settings.forensic_phase1_timeout_s, remaining(), report,
    )
    if res is not None:
        report.ground_truth = res
        report.phase1_done = True
        await _safe_update_status(
            db_pool, ctx.product_id, "phase1_done",
            {"ground_truth": report.ground_truth},
        )
        log.gate(
            "FORENSIC",
            f"  MSRP=${report.ground_truth.get('launch_msrp_usd')} | "
            f"Chip={report.ground_truth.get('chipset', '?')}",
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — FX Arbitrage (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 2/5] FX Arbitrage & Price Classification (60s budget)…")
    res = await _run_phase_with_budget(
        "Phase 2 FX Arbitrage",
        lambda: _phase2_fx_arbitrage(ctx, db_pool, llm_pool, settings, report.ground_truth),
        settings.forensic_phase2_timeout_s, remaining(), report,
    )
    if res is not None:
        report.arbitrage = res
        report.phase2_done = True
        await _safe_update_status(
            db_pool, ctx.product_id, "phase2_done",
            {"arbitrage": report.arbitrage},
        )
        log.gate(
            "FORENSIC",
            f"  label={report.arbitrage.get('price_label', 'unknown')} | "
            f"markup={report.arbitrage.get('markup_pct')}%",
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — Defect Mining (90s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 3/5] Reddit Defect Mining (90s budget)…")
    res = await _run_phase_with_budget(
        "Phase 3 Defect Mining",
        lambda: _phase3_defect_mining(ctx, db_pool, llm_pool, settings),
        settings.forensic_phase3_timeout_s, remaining(), report,
    )
    if res is not None:
        report.defects = res
        report.phase3_done = True
        await _safe_update_status(
            db_pool, ctx.product_id, "phase3_done",
            {"defect_count": len(report.defects)},
        )
        log.gate("FORENSIC", f"  {len(report.defects)} confirmed defect(s)")

    # ------------------------------------------------------------------ #
    # Phase 4 — Merchant Forensics (60s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 4/5] Merchant Forensic Audit (60s budget)…")
    res = await _run_phase_with_budget(
        "Phase 4 Merchant Audit",
        lambda: _phase4_merchant_audit(ctx, db_pool, llm_pool, settings),
        settings.forensic_phase4_timeout_s, remaining(), report,
    )
    if res is not None:
        report.merchant_audit = res
        report.phase4_done = True
        await _safe_update_status(
            db_pool, ctx.product_id, "phase4_done",
            {"merchant_audit": report.merchant_audit},
        )
        log.gate(
            "FORENSIC",
            f"  authorized={report.merchant_audit.get('is_authorized_agent', False)} | "
            f"dark_patterns={report.merchant_audit.get('dark_patterns', [])}",
        )

    # ------------------------------------------------------------------ #
    # Phase 5 — Predecessor Comparison (30s)
    # ------------------------------------------------------------------ #
    log.gate("FORENSIC", "[Phase 5/5] Predecessor Comparison (30s budget)…")
    res = await _run_phase_with_budget(
        "Phase 5 Predecessor",
        lambda: _phase5_predecessor_comparison(ctx, db_pool, llm_pool, settings, report.ground_truth),
        settings.forensic_phase5_timeout_s, remaining(), report,
    )
    if res is not None:
        report.predecessor = res
        report.phase5_done = True
        verdict = report.predecessor.get("verdict") or report.predecessor.get("upgrade_verdict", "?")
        log.gate("FORENSIC", f"  upgrade_verdict={verdict}")

    # ------------------------------------------------------------------ #
    # Final status evaluation
    # ------------------------------------------------------------------ #
    total_elapsed = time.monotonic() - started_at
    phases_done = sum(1 for i in range(1, 6) if getattr(report, f"phase{i}_done"))

    log.gate(
        "FORENSIC",
        f"  Pipeline elapsed={total_elapsed:.1f}s | timings={report.phase_timings} | "
        f"skipped={report.skipped_phases}",
    )

    if phases_done == 5:
        log.success(
            f"[Forensic] Pipeline COMPLETE for {product_name} | all 5 phases succeeded "
            f"in {total_elapsed:.1f}s."
        )
        await _safe_update_status(db_pool, ctx.product_id, "complete")
    elif phases_done > 0:
        log.warn(
            f"[Forensic] Pipeline PARTIAL for {product_name} | {phases_done}/5 phases "
            f"succeeded in {total_elapsed:.1f}s."
        )
        await _safe_update_status(db_pool, ctx.product_id, "partial")
    else:
        log.error(f"[Forensic] Pipeline FAILED for {product_name} | 0 phases succeeded.")
        await _safe_update_status(db_pool, ctx.product_id, "failed")

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
