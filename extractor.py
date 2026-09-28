"""Track A commerce-forensics extractor.

Track A is the *listing-time* extraction layer that runs during ingestion
(see ``main.fetch_and_parse_product_page``). Where the LLM extractor in
``crawler.py`` pulls the headline product fields, this module pulls the
adversarial commerce signals a merchant would rather hide:

* **BNPL / installment plans** — the advertised teaser is usually the cash
  price; the true cost of an installment plan is ``cycles * installment``.
  ``extract_bnpl_plans`` surfaces that hidden markup (blueprint §A3.29).
* **3-signal stock verification** — button state, schema.org/JSON-LD
  ``availability`` metadata, and stock text/badge are cross-checked and the
  status only flips on **2-of-3 agreement** (blueprint §A1.3).
* **Bank credit-card promos** — local-bank card offers (Commercial Bank, HNB,
  Sampath, Nations Trust, …) that change the effective price for some buyers.
* **Payment surcharges** — card/convenience/processing fees stacked at
  checkout (blueprint §A3.30).

Everything here is deterministic and dependency-light: pure regex/text logic
that is trivially unit-testable without a browser, an LLM, or a database.
HTML parsing (via the optional site profile) uses BeautifulSoup when present
and degrades to raw-text regex otherwise.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from schemas import (
    BankCardPromo,
    BNPLPlan,
    PaymentSurcharge,
    StockVerification,
    TrackAExtras,
)

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

#: Known BNPL / installment providers in the Sri Lankan market.
BNPL_PROVIDERS = ("koko", "mintpay", "payhere", "kko", "webxpay")

#: Local banks whose credit-card promos we surface. Order matters: longer /
#: more specific names first so "Commercial Bank" wins over a bare "bank".
LOCAL_BANKS = (
    ("Commercial Bank", r"commercial\s+bank|combank"),
    ("HNB", r"\bhnb\b|hatton\s+national\s+bank"),
    ("Sampath Bank", r"sampath(?:\s+bank)?"),
    ("Nations Trust", r"nations\s+trust(?:\s+bank)?|\bntb\b"),
    ("People's Bank", r"people'?s\s+bank"),
    ("Bank of Ceylon", r"bank\s+of\s+ceylon|\bboc\b"),
    ("DFCC Bank", r"\bdfcc\b"),
    ("NDB", r"\bndb\b|national\s+development\s+bank"),
    ("Seylan Bank", r"seylan(?:\s+bank)?"),
    ("Pan Asia Bank", r"pan\s*asia(?:\s+bank)?"),
    ("Amana Bank", r"amana(?:\s+bank)?"),
    ("Cargills Bank", r"cargills\s+bank"),
    ("HSBC", r"\bhsbc\b"),
    ("Standard Chartered", r"standard\s+chartered|\bscb\b"),
)

# A currency amount like "Rs. 27,000.00", "LKR 27000", "27,000/-".
_AMOUNT = r"(?:rs\.?|lkr|₨)?\s*([\d][\d,]*(?:\.\d{1,2})?)\s*(?:/-|/=)?"

_OOS_MARKERS = (
    "out of stock",
    "sold out",
    "out-of-stock",
    "currently unavailable",
    "unavailable",
    "no longer available",
    "call for availability",
    "pre-order",
    "preorder",
    "notify me when",
)
_INSTOCK_MARKERS = (
    "in stock",
    "in-stock",
    "available now",
    "add to cart",
    "add to basket",
    "buy now",
    "ready to ship",
)


def _to_float(raw: str) -> Optional[float]:
    """Parse a scraped money string ('27,000.00', '9900') into a float."""
    if raw is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    if not cleaned or cleaned == ".":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# 1. BNPL / installment plans  (blueprint §A3.29)
# --------------------------------------------------------------------------- #

# "3 x Rs 10,000", "3 × 9900", "pay 3 installments of Rs. 10000",
# "3 easy payments of 9,900", "in 3 of Rs 9900".
_BNPL_CYCLE_PATTERNS = (
    re.compile(
        r"(\d{1,2})\s*(?:x|×|installments?\s+of|payments?\s+of|of)\s*" + _AMOUNT,
        re.IGNORECASE,
    ),
    re.compile(
        r"pay\s+in\s+(\d{1,2})\s*(?:of|x|×)?\s*" + _AMOUNT,
        re.IGNORECASE,
    ),
)


def extract_bnpl_plans(text: str, cash_price: Optional[float]) -> List[BNPLPlan]:
    """Extract BNPL/installment plans and compute their *true* hidden markup.

    ``markup_pct = ((cycles * installment_amount) - cash_price) / cash_price * 100``

    Only text within a short window around a recognised BNPL provider keyword is
    considered, so an unrelated "3 x USB cables" line never becomes a fake plan.
    A plan is still emitted when the cash price is unknown (markup left at 0.0).
    """
    if not text:
        return []

    lowered = text.lower()
    plans: List[BNPLPlan] = []
    seen: set = set()

    for provider in BNPL_PROVIDERS:
        start = 0
        while True:
            idx = lowered.find(provider, start)
            if idx == -1:
                break
            start = idx + len(provider)
            # Look at a window around the provider mention for the plan terms.
            window = text[max(0, idx - 80): idx + 160]
            for pat in _BNPL_CYCLE_PATTERNS:
                m = pat.search(window)
                if not m:
                    continue
                cycles = int(m.group(1))
                installment = _to_float(m.group(2))
                if not cycles or cycles < 2 or cycles > 60 or not installment:
                    continue
                total = round(cycles * installment, 2)
                key = (provider, cycles, installment)
                if key in seen:
                    continue
                seen.add(key)
                markup = 0.0
                if cash_price and cash_price > 0:
                    markup = round((total - cash_price) / cash_price * 100.0, 2)
                canonical = "koko" if provider in ("koko", "kko") else provider
                plans.append(
                    BNPLPlan(
                        provider=canonical,
                        cycles=cycles,
                        installment_amount=installment,
                        cash_price=cash_price or 0.0,
                        total_payable=total,
                        markup_pct=markup,
                    )
                )
                break  # one plan per provider mention
    return plans


# --------------------------------------------------------------------------- #
# 2. Three-signal stock verification  (blueprint §A1.3, 2-of-3 agreement)
# --------------------------------------------------------------------------- #

_SCHEMA_INSTOCK_RE = re.compile(
    r"availability\W+(?:https?://schema\.org/)?(InStock|OutOfStock|SoldOut|PreOrder|BackOrder|Discontinued)",
    re.IGNORECASE,
)


def _signal_from_schema(text: str) -> Optional[bool]:
    """Read schema.org/JSON-LD ``availability`` into a stock boolean."""
    if not text:
        return None
    m = _SCHEMA_INSTOCK_RE.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    if token == "instock":
        return True
    return False  # OutOfStock / SoldOut / PreOrder / BackOrder / Discontinued


def _signal_from_text(text: str) -> Optional[bool]:
    """Read visible stock text/badge into a stock boolean (OOS wins ties)."""
    if not text:
        return None
    low = text.lower()
    oos = any(marker in low for marker in _OOS_MARKERS)
    instock = any(marker in low for marker in _INSTOCK_MARKERS)
    if oos and not instock:
        return False
    if instock and not oos:
        return True
    if oos and instock:
        # Explicit out-of-stock language beats a stray "add to cart" template.
        return False
    return None


def verify_stock(
    button_in_stock: Optional[bool],
    schema_availability: Optional[bool],
    stock_text: Optional[str],
) -> StockVerification:
    """Cross-check 3 independent stock signals; require 2-of-3 agreement.

    Parameters
    ----------
    button_in_stock:
        Buy/add-to-cart button state (``True`` enabled, ``False`` disabled,
        ``None`` if not found). Pass a bool directly — it is the most reliable
        signal when a real browser evaluated the DOM.
    schema_availability:
        Either a pre-resolved bool, or ``None`` to derive from ``stock_text``'s
        embedded JSON-LD.
    stock_text:
        Visible stock text/badge (and/or raw JSON-LD) to scan.

    Returns a :class:`StockVerification`; ``in_stock`` is left ``None`` (status
    not updated) unless at least two signals agree.
    """
    schema_signal = (
        schema_availability
        if schema_availability is not None
        else _signal_from_schema(stock_text or "")
    )
    text_signal = _signal_from_text(stock_text or "")

    signals: Dict[str, Optional[bool]] = {
        "button": button_in_stock,
        "schema": schema_signal,
        "text": text_signal,
    }
    votes = [v for v in signals.values() if v is not None]
    true_votes = sum(1 for v in votes if v is True)
    false_votes = sum(1 for v in votes if v is False)

    verdict: Optional[bool] = None
    confidence = 0
    if true_votes >= 2 and true_votes >= false_votes:
        verdict, confidence = True, true_votes
    elif false_votes >= 2 and false_votes > true_votes:
        verdict, confidence = False, false_votes

    return StockVerification(
        in_stock=verdict,
        confidence=confidence,
        agreement_reached=verdict is not None,
        signals=signals,
    )



# --------------------------------------------------------------------------- #
# 3. Bank credit-card promos  (local banks)
# --------------------------------------------------------------------------- #

_DISCOUNT_PCT_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%\s*(?:off|discount|cashback)?", re.IGNORECASE)
_INSTALLMENT_MONTHS_RE = re.compile(
    r"(?:0%?\s*(?:interest)?\s*)?(?:for\s+)?(\d{1,2})\s*(?:month|months|installments?)",
    re.IGNORECASE,
)


def extract_bank_promos(text: str) -> List[BankCardPromo]:
    """Extract local-bank credit-card promotions from listing/merchant text.

    For each recognised bank mentioned, capture any discount percentage and/or
    0%-interest installment tenor stated in the surrounding text window. The
    window is clipped at the next bank mention so a promo for one bank never
    steals the numbers belonging to the bank named right after it.
    """
    if not text:
        return []

    # Locate every bank mention first so windows can be clipped at neighbours.
    hits = []  # (start, end, bank_name)
    for bank_name, pattern in LOCAL_BANKS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            hits.append((m.start(), m.end(), bank_name))
    hits.sort()

    promos: List[BankCardPromo] = []
    seen_banks: set = set()
    for i, (start, end, bank_name) in enumerate(hits):
        if bank_name in seen_banks:
            continue
        seen_banks.add(bank_name)
        prev_end = hits[i - 1][1] if i > 0 else 0
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        left = max(prev_end, start - 60)
        right = min(next_start, end + 120)
        window = text[left:right]
        # Only treat it as a card promo if card/credit/installment/offer context is near.
        if not re.search(r"card|credit|debit|installment|instalment|offer|promo|%|discount", window, re.IGNORECASE):
            continue
        # Prefer the forward window (bank name -> next bank): a "15% off" stated
        # *after* the bank name belongs to that bank, not the one named earlier.
        # Only fall back to the backward context if the forward window is silent,
        # so an earlier bank's trailing discount never bleeds into the next bank.
        forward = text[end:right]
        pct_m = _DISCOUNT_PCT_RE.search(forward) or _DISCOUNT_PCT_RE.search(window)
        months_m = _INSTALLMENT_MONTHS_RE.search(forward) or _INSTALLMENT_MONTHS_RE.search(window)
        promos.append(
            BankCardPromo(
                bank=bank_name,
                discount_pct=_to_float(pct_m.group(1)) if pct_m else None,
                installment_months=int(months_m.group(1)) if months_m else None,
                raw_text=window.strip()[:200],
            )
        )
    return promos


# --------------------------------------------------------------------------- #
# 4. Payment surcharges / convenience fees  (blueprint §A3.30)
# --------------------------------------------------------------------------- #

_SURCHARGE_PATTERNS = (
    # method, regex producing a percentage or amount group
    ("credit_card", re.compile(r"(?:card|visa|master\s*card|credit\s*card)\s*(?:payment)?\s*(?:has\s+)?(?:a\s+)?(?:surcharge|fee|charge)?\s*(?:of\s+)?(\d{1,2}(?:\.\d{1,2})?)\s*%", re.IGNORECASE)),
    # reversed order: "3.5% credit card surcharge"
    ("credit_card", re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%\s*(?:credit\s*card|card|visa|master\s*card)\s*(?:payment\s+)?(?:surcharge|fee|charge)", re.IGNORECASE)),
    ("convenience_fee", re.compile(r"convenience\s+fee\s*(?:of\s+)?" + _AMOUNT, re.IGNORECASE)),
    ("processing_fee", re.compile(r"(?:payment\s+)?processing\s+(?:fee|surcharge)\s*(?:of\s+)?(?:(\d{1,2}(?:\.\d{1,2})?)\s*%|" + _AMOUNT + r")", re.IGNORECASE)),
)


def extract_surcharges(text: str) -> List[PaymentSurcharge]:
    """Extract payment surcharges / convenience fees added on top of cash price."""
    if not text:
        return []

    surcharges: List[PaymentSurcharge] = []
    seen: set = set()

    for method, pat in _SURCHARGE_PATTERNS:
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            raw_val = groups[0]
            is_pct = "%" in m.group(0)
            val = _to_float(raw_val)
            if val is None:
                continue
            key = (method, val, is_pct)
            if key in seen:
                continue
            seen.add(key)
            surcharges.append(
                PaymentSurcharge(
                    method=method,
                    surcharge_pct=val if is_pct else None,
                    surcharge_lkr=None if is_pct else val,
                    raw_text=m.group(0).strip()[:160],
                )
            )
    return surcharges


# --------------------------------------------------------------------------- #
# 5. Orchestrator — extract all Track A signals from a DOM/text blob
# --------------------------------------------------------------------------- #

def _visible_text(html_or_text: str) -> str:
    """Return visible text. Uses BeautifulSoup if available, else the raw blob."""
    if not html_or_text:
        return ""
    if "<" not in html_or_text:
        return html_or_text
    try:  # optional dependency
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_or_text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            # keep JSON-LD scripts — schema.org availability lives there
            if tag.name == "script" and (tag.get("type") or "").endswith("ld+json"):
                continue
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        # Crude tag strip fallback.
        return re.sub(r"<[^>]+>", " ", html_or_text)


def _button_signal(html: str, profile=None) -> Optional[bool]:
    """Infer the buy-button stock signal from HTML (disabled attr / OOS class)."""
    if not html or "<" not in html:
        return None
    # A disabled add-to-cart button is a strong out-of-stock signal.
    if re.search(r"add[\s_-]?to[\s_-]?cart[^>]*\bdisabled\b", html, re.IGNORECASE):
        return False
    if re.search(r"\bdisabled\b[^>]*add[\s_-]?to[\s_-]?cart", html, re.IGNORECASE):
        return False
    if re.search(r"(?:btn|button)[^>]*(?:out-?of-?stock|sold-?out|unavailable)", html, re.IGNORECASE):
        return False
    if re.search(r"add[\s_-]?to[\s_-]?cart|buy[\s_-]?now", html, re.IGNORECASE):
        return True
    return None


def extract_track_a(
    html_or_text: str,
    cash_price: Optional[float] = None,
    profile=None,
) -> TrackAExtras:
    """Run every Track A extractor over a DOM/text blob and bundle the results.

    ``profile`` is an optional :class:`site_profiles.loader.SiteProfile`; when
    present its selectors refine the button/stock signals. The function never
    raises on malformed input — a bad blob yields empty results.
    """
    raw = html_or_text or ""
    text = _visible_text(raw)

    button = _button_signal(raw, profile)
    stock = verify_stock(button_in_stock=button, schema_availability=None, stock_text=raw)

    return TrackAExtras(
        bnpl_plans=extract_bnpl_plans(text, cash_price),
        stock=stock,
        bank_promos=extract_bank_promos(text),
        surcharges=extract_surcharges(text),
    )
