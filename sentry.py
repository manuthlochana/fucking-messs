"""The Smart AI Sentry: real-time page-state classification and trap detection.

Two-stage design so we never waste an LLM call (or leak a captcha page into
extraction):

1. **Rule-based pre-flight** — cheap regex/length checks over the *raw* HTML and
   HTTP status catch the unambiguous cases: Cloudflare/Turnstile/DataDome
   challenges, near-empty DOMs, and hard error pages. When a rule fires we
   short-circuit with a high-confidence verdict and never call Gemini.
2. **LLM decision gate** — for everything else, the first ~2,000 tokens of
   cleaned Markdown go to Gemini Flash, which returns a structured
   :class:`PageInspectionResult`.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from config import Settings, settings as default_settings
from llm_pool import GeminiClient, LLMError
from schemas import PageInspectionResult, PageTypeEnum

# --------------------------------------------------------------------------- #
# Rule-based signatures (matched case-insensitively against raw HTML).
# --------------------------------------------------------------------------- #
_CHALLENGE_MARKERS = (
    "cf-chl",
    "cf_chl_opt",
    "/cdn-cgi/challenge-platform",
    "challenge-platform",
    "just a moment...",
    "attention required! | cloudflare",
    "turnstile",
    "g-recaptcha",
    "grecaptcha",
    "h-captcha",
    "hcaptcha",
    "px-captcha",
    "_px3",
    "datadome",
    "please verify you are a human",
    "enable javascript and cookies to continue",
    "checking your browser before accessing",
)

_NOT_FOUND_MARKERS = (
    "404 not found",
    "error 404",
    "page not found",
    "this page could not be found",
    "the page you requested could not be found",
    "http error 404",
)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


SENTRY_SYSTEM_INSTRUCTION = (
    "You are a fast, precise web-page state classifier for an e-commerce scraper "
    "operating on Sri Lankan (LKR) online stores. Given a Markdown slice of a "
    "single page, classify it into exactly one PageType and judge whether it is a "
    "single, extractable product listing. Definitions:\n"
    "- PRODUCT_PAGE: one specific product with a price and buy/add-to-cart intent.\n"
    "- CATEGORY_GRID: a listing/search/collection page linking to many products.\n"
    "- BOT_CHALLENGE_CAPTCHA: captcha, 'verify you are human', or JS/cookie wall.\n"
    "- OUT_OF_STOCK_PLACEHOLDER: a product page whose item is unavailable/sold out "
    "with little real detail to extract.\n"
    "- ERROR_404_PAGE: not-found / dead / error page.\n"
    "- UNKNOWN_JUNK: anything else (home page, blog, empty, unrelated).\n"
    "Set is_valid_product true ONLY for PRODUCT_PAGE with a usable price. Give a "
    "calibrated confidence in [0,1] and a one-sentence reasoning."
)


class SmartSentry:
    """Classifies page states and flags bot traps before extraction."""

    def __init__(self, llm: GeminiClient, settings: Settings = default_settings) -> None:
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Stage 1 — rule-based pre-flight
    # ------------------------------------------------------------------ #
    def preflight(self, html: Optional[str], status_code: Optional[int]) -> Optional[PageInspectionResult]:
        """Return a verdict when a cheap rule fires, else ``None`` (defer to LLM)."""
        text = (html or "")
        lowered = text.lower()

        # Bot challenge markers dominate — a 403 captcha wall is still a captcha.
        for marker in _CHALLENGE_MARKERS:
            if marker in lowered:
                return PageInspectionResult(
                    page_type=PageTypeEnum.BOT_CHALLENGE_CAPTCHA,
                    is_valid_product=False,
                    confidence_score=0.97,
                    reasoning=f"Rule pre-flight: challenge marker '{marker}' present in DOM.",
                )

        # HTTP status short-circuits.
        if status_code is not None:
            if status_code == 404 or status_code == 410:
                return PageInspectionResult(
                    page_type=PageTypeEnum.ERROR_404_PAGE,
                    is_valid_product=False,
                    confidence_score=0.98,
                    reasoning=f"Rule pre-flight: HTTP {status_code}.",
                )
            if status_code in (403, 429):
                # Ambiguous: could be a soft block. Only decide if markers exist;
                # otherwise defer to the LLM on whatever body we got.
                if any(m in lowered for m in _CHALLENGE_MARKERS):
                    return PageInspectionResult(
                        page_type=PageTypeEnum.BOT_CHALLENGE_CAPTCHA,
                        is_valid_product=False,
                        confidence_score=0.9,
                        reasoning=f"Rule pre-flight: HTTP {status_code} with challenge markers.",
                    )
            if status_code >= 500:
                return PageInspectionResult(
                    page_type=PageTypeEnum.UNKNOWN_JUNK,
                    is_valid_product=False,
                    confidence_score=0.85,
                    reasoning=f"Rule pre-flight: server error HTTP {status_code}.",
                )

        # Near-empty DOM — nothing to reason about.
        if len(text.strip()) < self._settings.min_dom_length:
            return PageInspectionResult(
                page_type=PageTypeEnum.UNKNOWN_JUNK,
                is_valid_product=False,
                confidence_score=0.9,
                reasoning=(
                    f"Rule pre-flight: DOM under {self._settings.min_dom_length} chars "
                    f"({len(text.strip())})."
                ),
            )

        # Explicit 404 phrasing in the body/title even when status looked OK.
        title = ""
        m = _TITLE_RE.search(text)
        if m:
            title = m.group(1).strip().lower()
        if any(k in title for k in _NOT_FOUND_MARKERS) or any(k in lowered for k in _NOT_FOUND_MARKERS):
            return PageInspectionResult(
                page_type=PageTypeEnum.ERROR_404_PAGE,
                is_valid_product=False,
                confidence_score=0.85,
                reasoning="Rule pre-flight: not-found phrasing detected in DOM/title.",
            )

        return None

    # ------------------------------------------------------------------ #
    # Stage 2 — LLM decision gate
    # ------------------------------------------------------------------ #
    async def classify(self, markdown: str, status_code: Optional[int] = None) -> PageInspectionResult:
        """Classify cleaned Markdown via Gemini Flash (fail-safe on error)."""
        slice_ = (markdown or "")[: self._settings.sentry_char_budget]
        prompt = (
            f"HTTP status: {status_code if status_code is not None else 'unknown'}\n"
            "Classify the following page content slice.\n"
            "----- BEGIN PAGE CONTENT -----\n"
            f"{slice_}\n"
            "----- END PAGE CONTENT -----"
        )
        try:
            return await self._llm.generate_structured(
                prompt,
                PageInspectionResult,
                system_instruction=SENTRY_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=512,
            )
        except LLMError as exc:
            # Never let a classifier hiccup crash the crawl — degrade to junk.
            return PageInspectionResult(
                page_type=PageTypeEnum.UNKNOWN_JUNK,
                is_valid_product=False,
                confidence_score=0.0,
                reasoning=f"LLM classification failed, defaulting to junk: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Combined entry point
    # ------------------------------------------------------------------ #
    async def inspect(
        self,
        *,
        html: Optional[str],
        markdown: str,
        status_code: Optional[int] = None,
    ) -> Tuple[PageInspectionResult, bool]:
        """Run pre-flight then (if needed) the LLM.

        Returns ``(result, decided_by_rules)`` so callers can log which gate
        produced the verdict.
        """
        rule_verdict = self.preflight(html, status_code)
        if rule_verdict is not None:
            return rule_verdict, True
        return await self.classify(markdown, status_code), False
