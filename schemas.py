"""Pydantic models: extraction targets and Sentry classification verdicts.

Two families of models live here:

* **Verdicts** — what the Sentry decides about a page (:class:`PageInspectionResult`).
* **Products** — the strictly validated commerce payload
  (:class:`ScrapedProductItem`) plus the LLM-facing :class:`ProductExtraction`
  the extractor fills in before we assemble the final, validated item.

The split matters: Gemini structured output is happiest with flat, fully typed
schemas. We therefore let the model fill ``ProductExtraction`` (no URL, no
timestamp, specs as an explicit key/value list rather than an open-ended map)
and construct the authoritative ``ScrapedProductItem`` ourselves.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Page classification
# --------------------------------------------------------------------------- #
class PageTypeEnum(str, Enum):
    """Coarse page states the Sentry must distinguish."""

    PRODUCT_PAGE = "PRODUCT_PAGE"
    CATEGORY_GRID = "CATEGORY_GRID"
    BOT_CHALLENGE_CAPTCHA = "BOT_CHALLENGE_CAPTCHA"
    OUT_OF_STOCK_PLACEHOLDER = "OUT_OF_STOCK_PLACEHOLDER"
    ERROR_404_PAGE = "ERROR_404_PAGE"
    UNKNOWN_JUNK = "UNKNOWN_JUNK"


class PageInspectionResult(BaseModel):
    """The Sentry's verdict for a single page."""

    page_type: PageTypeEnum = Field(description="Best-guess classification of the page state.")
    is_valid_product: bool = Field(
        description="True only if the page is a single, extractable product listing."
    )
    confidence_score: float = Field(
        description="Confidence in the classification, from 0.0 to 1.0.",
    )
    reasoning: str = Field(description="Short (one sentence) justification for the classification.")

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        """Clamp to [0, 1] and coerce NaN/inf to 0 so a stray value never crashes."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return max(0.0, min(1.0, f))


# --------------------------------------------------------------------------- #
# Product extraction
# --------------------------------------------------------------------------- #
class SpecItem(BaseModel):
    """A single key/value specification pair.

    A list of these is far more reliable with structured output than an
    open-ended ``Dict[str, Any]`` (which maps to JSON-Schema
    ``additionalProperties`` and is inconsistently supported).
    """

    key: str
    value: str


class ProductExtraction(BaseModel):
    """LLM-facing extraction schema. No URL/timestamp — the pipeline adds those."""

    raw_title: str = Field(description="The product title exactly as shown on the page.")
    clean_title: str = Field(
        description="Normalized title with marketing noise (e.g. 'Best!!', 'Free Shipping', "
        "emoji, ALL-CAPS shouting) removed. Keep model numbers and capacities."
    )
    brand: Optional[str] = Field(default=None, description="Manufacturer/brand, or null if unclear.")
    price_lkr: float = Field(
        description="Current selling price as a plain number (strip 'Rs.', 'LKR', commas)."
    )
    original_price_lkr: Optional[float] = Field(
        default=None, description="Pre-discount/RRP price as a plain number, or null."
    )
    in_stock: bool = Field(description="True if the item is purchasable, false if out of stock.")
    warranty_claimed: Optional[str] = Field(
        default=None, description="Warranty text if stated (e.g. '1 Year Warranty'), else null."
    )
    specs: List[SpecItem] = Field(
        default_factory=list, description="A handful of the most important key specifications."
    )

    def to_item(self, product_url: str) -> "ScrapedProductItem":
        """Assemble the authoritative, validated product item."""
        return ScrapedProductItem(
            raw_title=self.raw_title,
            clean_title=self.clean_title,
            brand=self.brand,
            price_lkr=self.price_lkr,
            original_price_lkr=self.original_price_lkr,
            in_stock=self.in_stock,
            warranty_claimed=self.warranty_claimed,
            specs_snippet={s.key: s.value for s in self.specs},
            product_url=product_url,
        )


class ScrapedProductItem(BaseModel):
    """Strictly validated commerce record — the engine's final output."""

    raw_title: str
    clean_title: str
    brand: Optional[str] = None
    price_lkr: float
    original_price_lkr: Optional[float] = None
    in_stock: bool
    warranty_claimed: Optional[str] = None
    specs_snippet: Dict[str, Any] = Field(default_factory=dict)
    product_url: str
    scraped_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("price_lkr")
    @classmethod
    def _price_must_be_positive_finite(cls, v: float) -> float:
        if v is None or math.isnan(v) or math.isinf(v):
            raise ValueError("price_lkr must be a finite number")
        if v <= 0:
            raise ValueError(f"price_lkr must be > 0, got {v}")
        return round(float(v), 2)

    @field_validator("original_price_lkr")
    @classmethod
    def _original_price_sane(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v) or v <= 0:
            # Treat a nonsensical original price as simply absent rather than fatal.
            return None
        return round(float(v), 2)


# --------------------------------------------------------------------------- #
# Forensic pipeline schemas
# --------------------------------------------------------------------------- #

class GroundTruthSpec(BaseModel):
    """Official manufacturer ground-truth for Phase 1 of the forensic pipeline."""

    official_product_name: str = Field(description="Canonical product name as sold globally.")
    chipset: Optional[str] = Field(default=None, description="SoC/processor name.")
    display_inches: Optional[float] = Field(default=None, description="Screen size in inches.")
    battery_mah: Optional[int] = Field(default=None, description="Battery capacity in mAh.")
    main_camera_mp: Optional[int] = Field(default=None, description="Main camera megapixels.")
    launch_msrp_usd: Optional[float] = Field(default=None, description="Official launch MSRP in USD.")
    official_url: Optional[str] = Field(default=None, description="Official brand product page URL.")
    launch_year: Optional[int] = Field(default=None, description="Year the product launched.")
    ip_rating: Optional[str] = Field(default=None, description="IP water/dust resistance rating (e.g. IP68).")
    chassis_material: Optional[str] = Field(default=None, description="Body material (e.g. 'titanium', 'aluminium', 'polycarbonate').")
    confidence: float = Field(description="Confidence 0-1 in the data accuracy.", default=0.5)


class ArbitrageResult(BaseModel):
    """FX arbitrage analysis output for Phase 2 of the forensic pipeline."""

    from typing import Literal as _Literal  # noqa: F401

    global_msrp_usd: Optional[float] = Field(default=None, description="Official MSRP in USD.")
    cbsl_rate_usd_lkr: Optional[float] = Field(default=None, description="Central bank USD→LKR rate used.")
    true_landed_cost_lkr: Optional[float] = Field(
        default=None,
        description="MSRP × FX × duty/freight factor (1.18 by default).",
    )
    merchant_price_lkr: float = Field(description="Actual merchant asking price in LKR.")
    markup_pct: Optional[float] = Field(default=None, description="((merchant - landed) / landed) × 100.")
    price_label: str = Field(
        description="One of: sub_msrp_likely_grey, fair_import_margin, price_gouged.",
        default="fair_import_margin",
    )


class DefectFinding(BaseModel):
    """A single corroborated hardware defect finding."""

    category: str = Field(description="Defect category, e.g. 'thermal_throttling', 'green_line_display'.")
    description: str = Field(description="Plain-language summary of the issue.")
    severity: str = Field(description="One of: critical, high, moderate, low.", default="moderate")
    corroborating_sources: List[str] = Field(
        default_factory=list,
        description="URLs or platform+thread identifiers of ≥2 independent sources.",
    )
    affects_stock_config: bool = Field(
        default=True,
        description="True if the defect affects the standard retail configuration.",
    )
    resolved_in_revision: Optional[str] = Field(
        default=None,
        description="Hardware revision that resolved the issue, if known.",
    )


class DefectReport(BaseModel):
    """Aggregated Phase 3 defect mining result."""

    defects: List[DefectFinding] = Field(default_factory=list)
    overall_astroturf_risk: float = Field(
        default=0.0,
        description="0–1 score: probability that positive reviews are manufactured/astroturfed.",
    )
    confidence: str = Field(
        default="low",
        description="Confidence in findings: low, medium, or high.",
    )


class MerchantAudit(BaseModel):
    """Phase 4 merchant forensic audit result."""

    dark_patterns: Dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Boolean flags for each dark pattern detected: "
            "fake_urgency_timer, false_scarcity_counter, "
            "social_proof_manipulation, bnpl_installment_obfuscation, "
            "no_physical_address."
        ),
    )
    bnpl_surcharge_pct: Optional[float] = Field(
        default=None,
        description="Detected BNPL (Koko/Mintpay) surcharge as a percentage, e.g. 3.5.",
    )
    physical_address_found: bool = Field(
        default=False,
        description="True if a Sri Lankan street/city address was detected on the merchant site.",
    )
    authorized_agent_verified: bool = Field(
        default=False,
        description="True if the merchant appears on the authorised distributor allowlist.",
    )
    is_authorized_agent: bool = Field(
        default=False,
        description="Alias for authorized_agent_verified for backward compatibility.",
    )


class PredecessorComparison(BaseModel):
    """Phase 5 generational comparison and upgrade verdict."""

    predecessor_model: str = Field(description="Name of the immediately preceding generation.")
    key_improvements: List[str] = Field(
        default_factory=list,
        description="List of meaningful upgrades over the predecessor.",
    )
    key_regressions: List[str] = Field(
        default_factory=list,
        description="Any areas where the new model is worse than the predecessor.",
    )
    verdict: str = Field(
        description="Upgrade recommendation: worth_upgrade, marginal, or skip.",
        default="marginal",
    )
    reasoning: str = Field(description="One-paragraph justification for the verdict.")

