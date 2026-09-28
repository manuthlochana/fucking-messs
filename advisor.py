"""Conversational RAG shopping advisor (blueprint Part B).

The advisor wraps the "Veteran Human Hardware Auditor" persona (222222.md §B1)
around a retrieval step that pulls *completed* forensic dossiers out of Postgres
and injects them into the model context using the §B2 template. It returns a
structured :class:`AdvisorRecommendation` rather than free text, so the dashboard
/ API can render warranty warnings, defect flags and price assessments as
first-class fields.

Design constraints honoured here:

* **Only complete intelligence is cited.** Retrieval filters to
  ``pipeline_status='complete'`` so a half-analysed product never drives advice.
* **No fabrication.** When the DB is empty / dry-run, we do *not* invent data —
  the advisor returns an explicit "insufficient data" recommendation.
* **Graceful degradation.** With no LLM pool available, a deterministic summary
  is produced from the retrieved rows so the endpoint still works offline.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import Settings, settings as default_settings
from logging_utils import log

# ---------------------------------------------------------------------------
# Persona system prompt (222222.md §B1, condensed to the operative rules).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are KALA-BALANA, a veteran hardware auditor and market buyer with 15+ years
of hands-on teardown, benchmarking, and local retail experience in Sri Lanka.
You are NOT a marketing assistant. You speak the way a trusted, brutally candid
senior technician speaks to a friend about to spend real money.

Non-negotiable rules:
1. STRIP MARKETING LANGUAGE — translate hype into concrete, falsifiable claims.
2. ANCHOR TO GLOBAL BASELINE PRICING — state US MSRP converted at the given FX
   rate and the resulting import/retail markup % before judging a local price.
   Assume ~15-20% typical import+duty overhead and state that assumption.
3. WARRANTY RISK OVERRIDES RAW PRICE — never recommend the cheapest listing if
   its warranty tier is unverified without explicitly flagging the tradeoff.
4. DEFECTS ARE SEVERITY-CALIBRATED, NOT ALARMIST — cite (a) how many independent
   sources corroborate, (b) whether it affects stock config, (c) whether it was
   fixed in a later revision. Never present one forum thread as a recall.
5. GENERATIONAL-JUMP HONESTY — say plainly when a premium over the predecessor
   is not justified for the user's use case.
6. NEVER FABRICATE A SPEC, PRICE, OR REVIEW SOURCE. If the context lacks a data
   point, say what is missing and what you'd need to verify it.
7. PLAIN, DIRECT, PEER-TO-PEER LANGUAGE. Give the specific answer, then nuance.
8. DISCLOSE CONFIDENCE / DATA FRESHNESS in a closing one-line note.

You exist to save the user from: (1) overpaying vs true global value, (2) a
warranty trap, (3) a known hardware flaw discovered after the return window.
"""


class AdvisorRecommendation(BaseModel):
    """Structured shopping advice returned to the caller/UI."""

    verdict: str = Field(
        default="insufficient_data",
        description="One of: buy, buy_with_caveats, hold, avoid, consider_alternative, insufficient_data.",
    )
    headline: str = Field(description="One-line bottom-line answer for the user.")
    price_assessment: str = Field(
        default="",
        description="Local price vs global baseline + markup %, with import-overhead assumption stated.",
    )
    best_listing: Optional[str] = Field(
        default=None,
        description="Merchant + price of the recommended listing, or null if none advised.",
    )
    warranty_warning: Optional[str] = Field(
        default=None,
        description="Explicit warranty-tradeoff warning when the cheapest listing is unverified.",
    )
    defect_warnings: List[str] = Field(
        default_factory=list,
        description="Severity-calibrated defect notes, each citing corroborating source count.",
    )
    confidence_note: str = Field(
        default="",
        description="Closing confidence / data-freshness disclosure.",
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_complete_products(
    db_pool: Optional[object],
    query: Optional[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return catalog rows for products whose forensic pipeline is COMPLETE.

    Filtering to ``pipeline_status='complete'`` is the blueprint's guard against
    citing half-analysed intelligence. Returns [] in dry-run mode (no DB).
    """
    if db_pool is None:
        return []
    from db import queries

    # Over-fetch then filter, so a text query still surfaces enough completes.
    rows = await queries.get_catalog_products(db_pool, limit=limit * 4, offset=0, query=query)
    complete = [r for r in rows if str(r.get("pipeline_status")) == "complete"]
    return complete[:limit]


async def load_dossier_bundle(
    db_pool: Optional[object],
    product_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the full product + listings + dossier + arbitrage bundle for context."""
    if db_pool is None:
        return None
    from db import queries

    return await queries.get_product_detail_with_dossier(db_pool, product_id)


# ---------------------------------------------------------------------------
# Context assembly (222222.md §B2 template)
# ---------------------------------------------------------------------------

def _esc(val: Any) -> str:
    """XML-attribute-safe string for a scalar value (never raises)."""
    if val is None:
        return ""
    return html.escape(str(val), quote=True)


def build_rag_context(bundle: Dict[str, Any], settings: Settings = default_settings) -> str:
    """Render a product bundle into the §B2 ``<product_context>`` block.

    Only data actually present in the bundle is emitted — missing fields are
    left blank rather than invented, so the persona's no-fabrication rule holds.
    """
    product = bundle.get("product", {}) or {}
    listings = bundle.get("listings", []) or []
    dossier = bundle.get("dossier") or {}
    arbitrage = (bundle.get("arbitrage") or bundle.get("arbitrage_logs") or [])
    latest_fx = arbitrage[0] if arbitrage else {}

    pid = _esc(product.get("id"))
    name = _esc(product.get("clean_title") or product.get("raw_title"))
    us_msrp = _esc(latest_fx.get("us_msrp_usd"))
    fx_rate = _esc(latest_fx.get("fx_rate"))

    lines: List[str] = ["<product_context>"]
    lines.append(f'  <canonical_product id="{pid}" name="{name}">')
    lines.append(f'    <global_baseline currency="USD" msrp="{us_msrp}" source="arbitrage_log"/>')
    lines.append(f'    <fx_rate pair="USD/LKR" rate="{fx_rate}" source="Central Bank"/>')
    lines.append("  </canonical_product>")

    lines.append(f'  <local_listings count="{len(listings)}">')
    for lst in listings:
        lines.append(
            '    <listing merchant="{m}" price_lkr="{p}" warranty_tier="{w}" '
            'warranty_verified="{v}" in_stock="{s}" bnpl_surcharge_pct="{b}"/>'.format(
                m=_esc(lst.get("merchant_name") or lst.get("merchant_domain")),
                p=_esc(lst.get("current_price_lkr")),
                w=_esc(lst.get("warranty_tier")),
                v=_esc(lst.get("is_verified_agent")),
                s=_esc(lst.get("in_stock")),
                b=_esc(lst.get("bnpl_surcharge_pct")),
            )
        )
    lines.append("  </local_listings>")

    lines.append(f'  <forensic_dossier generated="{_esc(dossier.get("generated_at"))}">')
    lines.append(
        '    <sentiment_confidence astroturf_risk="{a}" sponsored_content_ratio="{s}" '
        'confidence="{c}"/>'.format(
            a=_esc(dossier.get("astroturf_risk_score")),
            s=_esc(dossier.get("sponsored_content_ratio")),
            c=_esc(dossier.get("confidence_label")),
        )
    )
    lines.append("    <defects>")
    for d in (dossier.get("defects") or []):
        if not isinstance(d, dict):
            continue
        srcs = d.get("corroborating_sources") or []
        lines.append(
            '      <defect category="{cat}" severity="{sev}" corroborating_sources="{n}" '
            'affects_stock_config="{stock}" fixed_in_revision="{fix}"/>'.format(
                cat=_esc(d.get("category")),
                sev=_esc(d.get("severity") or d.get("severity_classification")),
                n=_esc(len(srcs) if isinstance(srcs, list) else d.get("source_count")),
                stock=_esc(d.get("affects_stock_config")),
                fix=_esc(d.get("resolved_in_revision")),
            )
        )
    lines.append("    </defects>")
    lines.append("  </forensic_dossier>")
    lines.append("</product_context>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic offline fallback
# ---------------------------------------------------------------------------

def _offline_recommendation(
    bundles: List[Dict[str, Any]],
    user_query: str,
) -> AdvisorRecommendation:
    """Build advice from retrieved rows without an LLM (dry-run / no-key mode)."""
    if not bundles:
        return AdvisorRecommendation(
            verdict="insufficient_data",
            headline="No completed forensic dossier matches that query yet.",
            confidence_note="No LLM used; no complete intelligence available to cite.",
        )
    bundle = bundles[0]
    product = bundle.get("product", {}) or {}
    listings = sorted(
        bundle.get("listings", []) or [],
        key=lambda x: (x.get("current_price_lkr") or float("inf")),
    )
    name = product.get("clean_title") or product.get("raw_title") or "the product"
    cheapest = listings[0] if listings else None
    dossier = bundle.get("dossier") or {}
    defects = [
        f"{d.get('category', 'issue')} (severity {d.get('severity', '?')}, "
        f"{len(d.get('corroborating_sources') or [])} sources)"
        for d in (dossier.get("defects") or [])
        if isinstance(d, dict)
    ]

    warranty_warning = None
    best = None
    if cheapest:
        best = f"{cheapest.get('merchant_name') or cheapest.get('merchant_domain')} " \
               f"@ Rs.{cheapest.get('current_price_lkr')}"
        if not cheapest.get("is_verified_agent"):
            warranty_warning = (
                f"Cheapest listing ({best}) is warranty tier "
                f"'{cheapest.get('warranty_tier')}' — unverified agent coverage."
            )

    return AdvisorRecommendation(
        verdict="buy_with_caveats" if cheapest else "hold",
        headline=f"{name}: {len(listings)} listing(s) tracked; see caveats below.",
        price_assessment="LLM offline — global-baseline comparison not generated. "
        "Listing prices shown as-is from tracked merchants.",
        best_listing=best,
        warranty_warning=warranty_warning,
        defect_warnings=defects,
        confidence_note="Deterministic offline summary (no LLM); "
        f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def advise(
    user_query: str,
    db_pool: Optional[object] = None,
    llm_pool: Optional[object] = None,
    settings: Settings = default_settings,
    *,
    product_id: Optional[str] = None,
    max_products: int = 3,
) -> AdvisorRecommendation:
    """Answer a shopping question grounded in completed forensic dossiers.

    Retrieval → context assembly (§B2) → persona-wrapped structured LLM call.
    Falls back to a deterministic summary when no LLM pool is configured, and
    returns an explicit ``insufficient_data`` verdict when nothing is retrieved.
    """
    # 1. Retrieve grounding.
    if product_id:
        one = await load_dossier_bundle(db_pool, product_id)
        bundles = [one] if one else []
    else:
        rows = await retrieve_complete_products(db_pool, user_query, limit=max_products)
        bundles = []
        for r in rows:
            b = await load_dossier_bundle(db_pool, r["id"])
            if b:
                bundles.append(b)

    # 2. Nothing to cite → never fabricate.
    if not bundles:
        return AdvisorRecommendation(
            verdict="insufficient_data",
            headline="I don't have a completed forensic dossier for that yet.",
            confidence_note="Retrieval returned no products with pipeline_status='complete'.",
        )

    # 3. No LLM → deterministic offline advice.
    if llm_pool is None:
        return _offline_recommendation(bundles, user_query)

    # 4. Assemble the persona-wrapped prompt and call the LLM pool.
    context_blocks = "\n\n".join(build_rag_context(b, settings) for b in bundles)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{context_blocks}\n\n"
        f"<user_query>{user_query}</user_query>\n\n"
        "Respond ONLY with the structured recommendation fields. Cite defect "
        "source counts; anchor price to the global baseline and state your "
        "import-overhead assumption; flag any warranty tradeoff explicitly."
    )
    try:
        result = await llm_pool.generate_structured(prompt, AdvisorRecommendation)
        if isinstance(result, AdvisorRecommendation):
            return result
        # Some pools return a dict — coerce defensively.
        return AdvisorRecommendation(**result)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - network/LLM failure
        log.warn(f"[Advisor] LLM call failed ({exc}); falling back to offline summary.")
        return _offline_recommendation(bundles, user_query)


