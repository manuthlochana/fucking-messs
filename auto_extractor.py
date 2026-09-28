"""Zero-selector self-healing extraction (Step 2).

A 3-tier cascade that needs no per-site selector authoring and degrades
gracefully in a ``bs4``/``lxml``/Playwright-free environment (regex + stdlib
``json`` only):

    Tier 1  Structured metadata — Schema.org JSON-LD, Microdata, OpenGraph.
            Zero LLM calls; covers most Shopify/WooCommerce/Magento storefronts.
    Tier 2  Cached extraction strategy from ``domain_profiles`` (learned once,
            reused for the whole domain). Strategy tokens understood here:
            ``jsonld`` · ``og`` · ``microdata`` · ``re:<pattern>``.
    Tier 3  LLM value+strategy inference over a sanitized DOM skeleton. The
            inferred fields are validated (price > 0) and the winning strategy
            is cached back to ``domain_profiles`` so the domain self-heals.

Track-A commerce signals (BNPL markup vs cash price, and the 3-signal
button/schema/text stock check) are layered on via :func:`extractor.extract_track_a`.

Public API
----------
* :func:`extract_fields`          — pure extraction → :class:`ExtractionOutcome`.
* :func:`extract_and_persist_html` — extract + atomic Postgres upsert + forensic
                                     enqueue (the drip worker's entrypoint).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from config import Settings, settings as default_settings
from db import queries
from logging_utils import log

# --------------------------------------------------------------------------- #
# Extraction outcome
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionOutcome:
    """What a tier recovered, plus provenance for logging/caching."""

    title: Optional[str] = None
    brand: Optional[str] = None
    price_lkr: Optional[float] = None
    original_price_lkr: Optional[float] = None
    in_stock: Optional[bool] = None
    warranty: Optional[str] = None
    tier: str = "none"  # jsonld | microdata | opengraph | cached | llm | none
    # Strategy tokens worth caching to domain_profiles for the next page.
    strategy: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.price_lkr and self.price_lkr > 0 and self.title)


# --------------------------------------------------------------------------- #
# Shared parsing helpers
# --------------------------------------------------------------------------- #
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|svg|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_JSONLD_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
_AVAIL_IN = ("instock", "in_stock", "limitedavailability", "onlineonly", "preorder")
_AVAIL_OUT = ("outofstock", "out_of_stock", "soldout", "discontinued", "backorder")


def _to_price(val: Any) -> Optional[float]:
    """Coerce '1,299.00' / 'Rs. 45900' / 45900 → float, or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        p = float(val)
        return p if p > 0 else None
    m = re.search(r"([\d,]+(?:\.\d+)?)", str(val).replace(" ", ""))
    if not m:
        return None
    try:
        p = float(m.group(1).replace(",", ""))
        return p if p > 0 else None
    except ValueError:
        return None


def _availability_bool(val: Any) -> Optional[bool]:
    """Map a schema.org availability string to a tri-state bool."""
    if val is None:
        return None
    s = str(val).lower().replace("http://", "").replace("https://", "").replace("schema.org/", "")
    if any(k in s for k in _AVAIL_OUT):
        return False
    if any(k in s for k in _AVAIL_IN):
        return True
    return None


# --------------------------------------------------------------------------- #
# Tier 1a — Schema.org JSON-LD
# --------------------------------------------------------------------------- #
def _flatten_jsonld(data: Any) -> List[dict]:
    """Return every dict node from a JSON-LD blob (handles lists + @graph)."""
    out: List[dict] = []
    if isinstance(data, list):
        for d in data:
            out.extend(_flatten_jsonld(d))
    elif isinstance(data, dict):
        out.append(data)
        if "@graph" in data:
            out.extend(_flatten_jsonld(data["@graph"]))
    return out


def _is_product_node(node: dict) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(str(x).lower() == "product" for x in types if x)


def parse_jsonld(html: str) -> Optional[ExtractionOutcome]:
    """Extract a product from embedded Schema.org JSON-LD, or None."""
    for m in _JSONLD_RE.finditer(html or ""):
        blob = m.group(1).strip()
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except Exception:
            continue
        for node in _flatten_jsonld(data):
            if not _is_product_node(node):
                continue
            offers = node.get("offers")
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            offers = offers if isinstance(offers, dict) else {}
            price = _to_price(offers.get("price") or offers.get("lowPrice"))
            if price is None:
                continue
            brand = node.get("brand")
            if isinstance(brand, dict):
                brand = brand.get("name")
            return ExtractionOutcome(
                title=str(node.get("name") or "").strip() or None,
                brand=str(brand).strip() if brand else None,
                price_lkr=price,
                in_stock=_availability_bool(offers.get("availability")),
                tier="jsonld",
                strategy={"price_selector": "jsonld", "title_selector": "jsonld",
                          "stock_selector": "jsonld", "extraction_tier": "jsonld"},
            )
    return None


# --------------------------------------------------------------------------- #
# Tier 1b — OpenGraph / Microdata (regex, no DOM engine)
# --------------------------------------------------------------------------- #
def _meta_content(html: str, *keys: str) -> Optional[str]:
    """First ``<meta ... content="...">`` whose property/name/itemprop matches."""
    for key in keys:
        pat = (
            r'<meta[^>]+(?:property|name|itemprop)\s*=\s*["\']'
            + re.escape(key)
            + r'["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']'
        )
        m = re.search(pat, html or "", re.I)
        if not m:
            # attribute order may be reversed (content before property)
            pat2 = (
                r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name|itemprop)\s*=\s*["\']'
                + re.escape(key)
                + r'["\']'
            )
            m = re.search(pat2, html or "", re.I)
        if m:
            return m.group(1).strip()
    return None


def parse_opengraph(html: str) -> Optional[ExtractionOutcome]:
    price = _to_price(
        _meta_content(html, "product:price:amount", "og:price:amount", "og:product:price:amount")
    )
    if price is None:
        return None
    title = _meta_content(html, "og:title")
    avail = _meta_content(html, "product:availability", "og:availability")
    return ExtractionOutcome(
        title=(title or "").strip() or None,
        price_lkr=price,
        in_stock=_availability_bool(avail),
        tier="opengraph",
        strategy={"price_selector": "og", "title_selector": "og",
                  "stock_selector": "og", "extraction_tier": "opengraph"},
    )


def parse_microdata(html: str) -> Optional[ExtractionOutcome]:
    price = _to_price(_meta_content(html, "price"))
    if price is None:
        # itemprop=price may sit on a span with content= rather than a <meta>.
        m = re.search(
            r'itemprop\s*=\s*["\']price["\'][^>]*\bcontent\s*=\s*["\']([\d,.]+)["\']',
            html or "", re.I,
        )
        price = _to_price(m.group(1)) if m else None
    if price is None:
        return None
    title = _meta_content(html, "name")
    return ExtractionOutcome(
        title=(title or "").strip() or None,
        price_lkr=price,
        in_stock=_availability_bool(_meta_content(html, "availability")),
        tier="microdata",
        strategy={"price_selector": "microdata", "extraction_tier": "microdata"},
    )


def _title_from_html(html: str) -> Optional[str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.I | re.S)
    if not m:
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html or "", re.I | re.S)
    if not m:
        return None
    txt = _WS_RE.sub(" ", _TAG_RE.sub("", m.group(1))).strip()
    return txt[:300] or None


# --------------------------------------------------------------------------- #
# Tier 2 — cached domain strategy
# --------------------------------------------------------------------------- #
def apply_cached_strategy(html: str, profile: Optional[dict]) -> Optional[ExtractionOutcome]:
    """Re-run the strategy previously learned for this domain (0 LLM calls).

    Strategy tokens understood: ``jsonld`` · ``og`` · ``microdata`` ·
    ``re:<pattern>`` (a regex whose first group captures the price digits).
    """
    if not profile:
        return None
    price_sel = (profile.get("price_selector") or "").strip()
    tier = (profile.get("extraction_tier") or "").strip()

    if price_sel == "jsonld" or tier == "jsonld":
        oc = parse_jsonld(html)
    elif price_sel == "og" or tier == "opengraph":
        oc = parse_opengraph(html)
    elif price_sel == "microdata" or tier == "microdata":
        oc = parse_microdata(html)
    elif price_sel.startswith("re:"):
        m = re.search(price_sel[3:], html or "", re.I | re.S)
        price = _to_price(m.group(1)) if (m and m.groups()) else None
        if price is None:
            return None
        oc = ExtractionOutcome(title=_title_from_html(html), price_lkr=price)
    else:
        return None

    if oc and oc.price_lkr:
        oc.tier = "cached"
        if not oc.title:
            oc.title = _title_from_html(html)
    return oc


# --------------------------------------------------------------------------- #
# Tier 3 — LLM value + reusable-regex inference over a sanitized skeleton
# --------------------------------------------------------------------------- #
def sanitize_skeleton(html: str, budget: int) -> str:
    """Strip scripts/styles, collapse whitespace, and truncate to ``budget``.

    Retains tag attributes + prices so the LLM can infer a reusable price regex.
    """
    stripped = _SCRIPT_STYLE_RE.sub(" ", html or "")
    stripped = _WS_RE.sub(" ", stripped)
    return stripped[: max(500, budget)]


_LLM_SYSTEM = (
    "You extract one Sri Lankan (LKR) e-commerce product from raw page HTML. "
    "Return price_lkr as a plain number (strip 'Rs.', 'LKR', commas). Use null "
    "for genuinely unknown fields; never invent values. clean_title strips "
    "marketing noise but keeps brand, model numbers and capacities. in_stock is "
    "false for sold-out/unavailable items. For price_regex, return a Python "
    "regular expression with exactly ONE capturing group that isolates the price "
    "digits on THIS site's pages (e.g. r'\"price\"\\\\s*:\\\\s*\"([\\\\d.,]+)\"'), "
    "or null if unsure — it will be cached and reused for the whole domain."
)


def _build_inferred_model():
    """Build the Tier-3 Pydantic schema lazily (pydantic is a runtime dep)."""
    from pydantic import BaseModel, Field

    class InferredExtraction(BaseModel):
        raw_title: str = Field(description="Product title exactly as shown.")
        clean_title: str = Field(description="Title with marketing noise removed.")
        brand: Optional[str] = Field(default=None)
        price_lkr: float = Field(description="Current selling price, plain number.")
        in_stock: bool = Field(default=True)
        warranty_claimed: Optional[str] = Field(default=None)
        price_regex: Optional[str] = Field(
            default=None,
            description="Reusable Python regex with one capture group for the price digits.",
        )

    return InferredExtraction


async def infer_via_llm(
    html: str,
    url: str,
    llm_pool: Optional[object],
    settings: Settings,
) -> Optional[ExtractionOutcome]:
    """Tier 3: ask the LLM for values + a reusable price regex; validate both."""
    if llm_pool is None:
        return None
    model = _build_inferred_model()
    skeleton = sanitize_skeleton(html, settings.sentry_char_budget * 3)
    prompt = (
        f"Product URL: {url}\n"
        "Extract the product and infer a reusable price regex from this HTML:\n"
        "----- BEGIN HTML -----\n"
        f"{skeleton}\n"
        "----- END HTML -----"
    )
    try:
        inferred = await llm_pool.generate_structured(  # type: ignore[attr-defined]
            prompt, model,
            system_instruction=_LLM_SYSTEM,
            temperature=0.0,
            max_output_tokens=1024,
        )
    except Exception as exc:
        log.warn(f"[AutoX] LLM inference failed for {url}: {exc}")
        return None

    price = _to_price(getattr(inferred, "price_lkr", None))
    if price is None:
        return None

    strategy: Dict[str, str] = {"extraction_tier": "llm"}
    # Validate the LLM's regex against the page before trusting it for caching.
    rx = getattr(inferred, "price_regex", None)
    if rx:
        try:
            m = re.search(rx, html, re.I | re.S)
            if m and m.groups():
                rx_price = _to_price(m.group(1))
                if rx_price and abs(rx_price - price) <= max(1.0, price * 0.02):
                    strategy = {"price_selector": f"re:{rx}", "extraction_tier": "llm_regex"}
        except re.error:
            pass

    return ExtractionOutcome(
        title=(getattr(inferred, "raw_title", None) or "").strip() or None,
        brand=(getattr(inferred, "brand", None) or None),
        price_lkr=price,
        in_stock=bool(getattr(inferred, "in_stock", True)),
        warranty=getattr(inferred, "warranty_claimed", None),
        tier="llm",
        strategy=strategy,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _host(url: str) -> str:
    return (urlsplit(url).netloc or "unknown").lower()


_PROFILE_KEYS = {
    "title_selector", "price_selector", "stock_selector",
    "specs_table_selector", "extraction_tier",
}


async def extract_fields(
    html: str,
    url: str,
    *,
    pool: Optional[object] = None,
    llm_pool: Optional[object] = None,
    settings: Settings = default_settings,
) -> ExtractionOutcome:
    """Run the 3-tier cascade + Track-A signals; cache the winning strategy.

    Tier 1 (JSON-LD → OpenGraph → Microdata) → Tier 2 (cached domain strategy)
    → Tier 3 (LLM). Never raises.
    """
    domain = _host(url)

    # -- Tier 1: structured metadata (0 LLM) ------------------------------- #
    oc = parse_jsonld(html) or parse_opengraph(html) or parse_microdata(html)

    # -- Tier 2: cached domain strategy ------------------------------------ #
    if oc is None or not oc.price_lkr:
        profile = await queries.get_domain_profile(pool, domain)
        cached = apply_cached_strategy(html, profile)
        if cached and cached.price_lkr:
            oc = cached

    # -- Tier 3: LLM value + reusable-regex inference ---------------------- #
    if oc is None or not oc.price_lkr:
        llm_oc = await infer_via_llm(html, url, llm_pool, settings)
        if llm_oc is not None:
            oc = llm_oc

    if oc is None:
        oc = ExtractionOutcome(tier="none", notes=["no price recovered"])
    if not oc.title:
        oc.title = _title_from_html(html)

    # -- Track A: BNPL markup + 3-signal (button/schema/text) stock -------- #
    try:
        from extractor import extract_track_a

        ta = extract_track_a(html, cash_price=oc.price_lkr)
        if ta.stock.agreement_reached and ta.stock.in_stock is not None:
            oc.in_stock = bool(ta.stock.in_stock)
        elif oc.in_stock is None:
            oc.in_stock = True  # default purchasable when no negative signal
        oc.notes.append(
            f"bnpl_plans={len(ta.bnpl_plans)} promos={len(ta.bank_promos)} "
            f"surcharges={len(ta.surcharges)} stock_conf={ta.stock.confidence}"
        )
    except Exception as exc:
        if oc.in_stock is None:
            oc.in_stock = True
        oc.notes.append(f"track_a skipped: {exc}")

    # -- Self-heal: cache the learned strategy for the domain -------------- #
    if oc.usable and oc.strategy and pool is not None:
        try:
            await queries.upsert_domain_profile(
                pool, domain=domain,
                **{k: v for k, v in oc.strategy.items() if k in _PROFILE_KEYS},
            )
        except Exception as exc:
            log.debug(f"[AutoX] profile cache skipped for {domain}: {exc!r}")

    log.gate("AUTOX", f"tier={oc.tier} price={oc.price_lkr} in_stock={oc.in_stock} — {url}",
             ok=oc.usable)
    return oc


async def extract_and_persist_html(
    url: str,
    html: str,
    *,
    pool: Optional[object] = None,
    llm_pool: Optional[object] = None,
    settings: Settings = default_settings,
) -> Dict[str, Any]:
    """Drip-worker entrypoint: extract from pre-fetched HTML, then persist.

    Mirrors ``main.fetch_and_parse_product_page`` persistence — atomic merchant
    upsert, fingerprint dedup, listing upsert, price_history append, and a
    durable ``forensic_queue`` enqueue for genuinely new products. Never raises;
    returns a result dict describing the outcome.
    """
    from schemas import ScrapedProductItem

    result: Dict[str, Any] = {"url": url, "status": "error", "notes": []}
    oc = await extract_fields(html, url, pool=pool, llm_pool=llm_pool, settings=settings)
    result["tier"] = oc.tier
    result["notes"].extend(oc.notes)

    if not oc.usable:
        result["status"] = "extract_failed"
        result["notes"].append("no usable product price recovered")
        return result

    try:
        item = ScrapedProductItem(
            raw_title=oc.title or url,
            clean_title=oc.title or url,
            brand=oc.brand,
            price_lkr=float(oc.price_lkr),  # type: ignore[arg-type]
            original_price_lkr=oc.original_price_lkr,
            in_stock=bool(oc.in_stock),
            warranty_claimed=oc.warranty,
            product_url=url,
        )
    except Exception as exc:
        result["status"] = "validation_failed"
        result["notes"].append(f"item validation: {exc}")
        return result

    from normalizer import SpecNormalizer

    spec = SpecNormalizer().normalize(item.raw_title, brand_hint=item.brand)
    result["fingerprint"] = spec.spec_fingerprint[:16]

    if pool is None:
        result["status"] = "extracted_dry_run"
        result["price_lkr"] = item.price_lkr
        result["in_stock"] = item.in_stock
        return result

    try:
        from main import _map_warranty_tier
    except Exception:
        def _map_warranty_tier(_):  # type: ignore
            return "unstated"

    try:
        domain = _host(url)
        warranty_tier = _map_warranty_tier(item.warranty_claimed)
        async with pool.acquire() as conn:  # type: ignore[attr-defined]
            async with conn.transaction():
                merchant_id = await queries.upsert_merchant(conn, domain=domain, display_name=domain)
                existing_id = await queries.lookup_by_fingerprint(conn, spec.spec_fingerprint)
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
                    await queries.enqueue_forensic_job(
                        conn, product_id=product_id, merchant_id=merchant_id, listing_url=url
                    )
                    result["status"] = "new_product_queued"
        result["product_id"] = product_id
        result["price_lkr"] = item.price_lkr
        result["in_stock"] = item.in_stock
    except Exception as exc:
        result["status"] = "persist_failed"
        result["notes"].append(f"db persist error: {exc}")
        log.warn(f"[AutoX] persist failed for {url}: {exc}")

    return result
