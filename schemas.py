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
