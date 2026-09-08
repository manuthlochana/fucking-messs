"""Anti-Collision Spec Normalizer for KALA-BALANA.

Solves the variant-conflation problem: ``iPhone 16 Pro`` and
``iPhone 16 Pro Max`` must NEVER share a ``spec_fingerprint``, even when
a retailer's title mashes them together with marketing noise.

Design principles
-----------------
* Pure functions — zero I/O, zero network, 100% unit-testable.
* Deterministic: same logical product always produces the same fingerprint
  regardless of input capitalization, punctuation, emoji, or marketing junk.
* Conservative extraction: when a field cannot be reliably extracted it
  defaults to an empty string rather than guessing (prevents false merges).

Fingerprint formula
-------------------
    SHA-256( f"{brand}|{model_family}|{sub_model}|{ram_gb}|{storage_gb}|{region_code}".lower() )
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import json
import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class UniversalTechSpec(BaseModel):
    brand: Optional[str] = Field(default=None)
    model_family: Optional[str] = Field(default=None)
    sub_model: Optional[str] = Field(default=None)
    ram_gb: Optional[int] = Field(default=None)
    storage_gb: Optional[int] = Field(default=None)
    region_code: Optional[str] = Field(default="")
    condition: Optional[str] = Field(default="new")
    technical_attributes: Dict[str, Any] = Field(default_factory=dict)

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Noise patterns to strip before extraction
# ---------------------------------------------------------------------------

# Marketing junk phrases (case-insensitive, order matters for specificity).
_JUNK_PHRASES: list[str] = [
    r"\bfree\s+delivery\b",
    r"\bfree\s+shipping\b",
    r"\bfast\s+delivery\b",
    r"\bone\s+year\s+warranty\b",
    r"\b1\s+year\s+warranty\b",
    r"\b2\s+year\s+warranty\b",
    r"\bofficial\s+warranty\b",
    r"\blocal\s+warranty\b",
    r"\bauthorized\b",
    r"\bgenuine\b",
    r"\boriginal\b",
    r"\bsealed\b",
    r"\bnew\s+arrival\b",
    r"\bhot\s+deal\b",
    r"\bbest\s+price\b",
    r"\bbest\s+deal\b",
    r"\bspecial\s+offer\b",
    r"\blimited\s+offer\b",
    r"\bflash\s+sale\b",
    r"\bclearance\b",
    r"\bgift\b",
    r"\bcombo\b",
    r"\bset\b",
    r"\bpack\b",
    r"\bbundle\b",
    r"\binstallment\b",
    r"\beasypayment\b",
    r"\bkoko\s+pay\b",
    r"\bkoko\b",
    r"\bmintpay\b",
    r"\bmint\s+pay\b",
    r"\bpayhere\b",
    r"\bpay\b",
    r"\bbnpl\b",
    r"\bcredit\s+card\b",
    r"\bcash\s+on\s+delivery\b",
    r"\bcod\b",
    r"\bon\s+sale\b",
    r"\bdiscount\b",
    r"\bup\s+to\s+\d+%\s+off\b",
    r"\d+%\s+off\b",
    r"\bex\s+stock\b",
    r"\bpreorder\b",
    r"\bpre-order\b",
    r"\bpre\s+order\b",
]
_JUNK_RE = re.compile(
    "|".join(_JUNK_PHRASES), re.IGNORECASE
)

# Condition indicators (must be extracted before stripping, rule #27 & #36)
_CONDITION_PATTERNS = [
    (r"\b(refurbished|renewed|refurb)\b", "refurbished"),
    (r"\b(used|pre-owned|preowned|second\s*hand|2nd\s*hand)\b", "used"),
    (r"\b(open\s*box|open-box|demo\s*unit|display\s*unit)\b", "open_box"),
]

# Emoji and non-ASCII decorative symbols.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

# Parenthetical noise not captured by junk phrases (e.g., "(Best!)").
_PARENS_NOISE_RE = re.compile(r"\([^)]{0,60}\)")

# Repeated punctuation / special chars used as decoration.
_DECO_RE = re.compile(r"[★✓✅☆►•·\-]{2,}")

# Collapse whitespace.
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Known brand names including mobile, audio, chargers, and PC hardware.
_KNOWN_BRANDS: list[str] = [
    "Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Redmi", "POCO",
    "Oppo", "Vivo", "Realme", "Huawei", "Honor", "Sony", "Nokia", "Motorola",
    "Lenovo", "Asus", "LG", "HTC", "Itel", "Tecno", "Infinix",
    "Dell", "HP", "Acer", "MSI", "Razer", "Microsoft", "Toshiba",
    "Baseus", "Anker", "Ugreen", "JBL", "Bose", "Sennheiser", "Soundcore",
    "Marshall", "Corsair", "Logitech", "Keychron", "Kingston", "Crucial",
    "Gigabyte", "Zotac", "Sapphire", "Belkin",
]
_BRAND_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in _KNOWN_BRANDS) + r")\b",
    re.IGNORECASE,
)


# Sub-model modifiers — order matters: longer/more-specific first.
_SUBMODEL_PATTERNS: list[tuple[str, str]] = [
    (r"\bPro\s+Max\b",    "Pro Max"),
    (r"\bPro\s+Plus\b",   "Pro Plus"),
    (r"\bUltra\b",        "Ultra"),
    (r"\bPro\b",          "Pro"),
    (r"\bPlus\b",         "Plus"),
    (r"\bMax\b",          "Max"),
    (r"\bLite\b",         "Lite"),
    (r"\bFE\b",           "FE"),       # Fan Edition
    (r"\bSE\b",           "SE"),       # Special Edition
    (r"\bmini\b",         "mini"),
    (r"\bEdge\b",         "Edge"),
    (r"\bNote\b",         "Note"),
    (r"\bFold\b",         "Fold"),
    (r"\bFlip\b",         "Flip"),
]
_SUBMODEL_RES: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in _SUBMODEL_PATTERNS
]

# RAM/Storage: "8/256GB", "8GB/256GB", "8/256", "256GB", "512 GB".
_RAM_STORAGE_RE = re.compile(
    r"(\d+)\s*[Gg][Bb]?\s*/\s*(\d+)\s*[Gg][Bb]"
    r"|(\d+)\s*/\s*(\d+)\s*[Gg][Bb]",
    re.IGNORECASE,
)
_STORAGE_ONLY_RE = re.compile(r"(\d+)\s*[Gg][Bb]", re.IGNORECASE)

# Region / variant codes (post-strip, so marketing words are already gone).
_REGION_RE = re.compile(
    r"\b(CN|IN|HK|JP|US|KR|UK|EU|AU|SG|TH|MY|PH|AE|SA)\b",
    re.IGNORECASE,
)

# Model number extractor (e.g., "Galaxy S25", "iPhone 16", "Pixel 9", "A55").
# Looks for a series name followed by optional digit(s).
_MODEL_NUM_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][0-9]{0,2})?\s*\d{1,4}[a-z]?)\b"
)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedSpec:
    """Extracted, canonical product specification."""

    brand: str              # e.g. "apple", "samsung" (lowercased)
    model_family: str       # e.g. "iphone 16", "galaxy s25" (lowercased)
    sub_model: str          # e.g. "pro max", "ultra", "" (lowercased)
    ram_gb: Optional[int]   # e.g. 8, 12, None
    storage_gb: Optional[int]  # e.g. 256, 512, None
    region_code: str        # e.g. "cn", "us", "" (lowercased)
    spec_fingerprint: str   # SHA-256 hex digest
    condition: str = "new"  # "new", "refurbished", "used", "open_box"
    technical_attributes: str = ""

    @classmethod
    def from_parts(
        cls,
        brand: str,
        model_family: str,
        sub_model: str,
        ram_gb: Optional[int],
        storage_gb: Optional[int],
        region_code: str,
        condition: str = "new",
        technical_attributes: str = "",
    ) -> "NormalizedSpec":
        """Compute fingerprint and return an immutable NormalizedSpec."""
        fp = _compute_fingerprint(
            brand, model_family, sub_model, ram_gb, storage_gb, region_code,
            condition=condition, technical_attributes=technical_attributes,
        )
        return cls(
            brand=brand.lower().strip(),
            model_family=model_family.lower().strip(),
            sub_model=sub_model.lower().strip(),
            ram_gb=ram_gb,
            storage_gb=storage_gb,
            region_code=region_code.lower().strip(),
            spec_fingerprint=fp,
            condition=condition.lower().strip(),
            technical_attributes=technical_attributes.lower().strip(),
        )


# ---------------------------------------------------------------------------
# Core normalizer class
# ---------------------------------------------------------------------------


_NON_PHONE_CATEGORY_KEYWORDS = [
    r"\bmonitor\b",
    r"\bdisplay\b",
    r"\bkeyboard\b",
    r"\bssd\b",
    r"\bnvme\b",
    r"\bdrone\b",
    r"\bheadphone\b",
    r"\bheadphones\b",
    r"\bearbuds\b",
    r"\bearbud\b",
    r"\btws\b",
    r"\bcharger\b",
    r"\bpowerbank\b",
    r"\bpower\s+bank\b",
    r"\bmouse\b",
    r"\bgpu\b",
    r"\bgraphics\s+card\b",
]
_NON_PHONE_CATEGORY_RE = re.compile(
    "|".join(_NON_PHONE_CATEGORY_KEYWORDS), re.IGNORECASE
)


class UniversalTechSpecExtractor:
    """Tier 2 Normalizer using Gemini Flash."""
    
    _PROMPT = """
    Extract hardware specifications from the following product title:
    '{title}'
    
    You are extracting details for non-phone categories (Monitors, Storage, Keyboards, Drones, Audio, PC Components).
    Identify the brand, model_family, sub_model, ram_gb, storage_gb, and any relevant technical attributes (wattage, layout, switch type, capacity, resolution, refresh rate).
    Return a structured JSON according to the schema. If an attribute is missing, omit it or return null.
    """

    async def extract(self, raw_title: str, llm_pool: Any) -> UniversalTechSpec:
        prompt = self._PROMPT.format(title=raw_title)
        try:
            return await llm_pool.generate_structured(prompt, UniversalTechSpec)
        except Exception:
            # Fallback to empty spec if LLM fails
            return UniversalTechSpec(technical_attributes={"raw": raw_title})

class SpecNormalizer:
    """Strips noise and deterministically extracts product spec fields."""


    async def normalize_async(
        self,
        raw_title: str,
        llm_pool: Optional[Any] = None,
        brand_hint: Optional[str] = None,
    ) -> NormalizedSpec:
        """Asynchronous entry point that falls back to Tier 2 extraction if Tier 1 confidence is low or non-phone category detected."""
        # Tier 1 extraction
        t1_spec = self.normalize(raw_title, brand_hint)
        
        # Rule: Explicitly check for non-phone keywords (monitor, display, keyboard, ssd, nvme, drone, headphone, etc.)
        # If ANY of these category keywords are detected, or if no phone model matches, FORCIBLY trigger Tier 2 UniversalTechSpecExtractor
        has_non_phone_keyword = bool(_NON_PHONE_CATEGORY_RE.search(raw_title))
        has_phone_model = bool(t1_spec.model_family and len(t1_spec.model_family) >= 3 and not has_non_phone_keyword)
        
        needs_tier_2 = has_non_phone_keyword or (not has_phone_model) or bool(t1_spec.technical_attributes)
        
        if not needs_tier_2 or not llm_pool:
            return t1_spec
            
        # Tier 2 Extraction
        extractor = UniversalTechSpecExtractor()
        t2_res = await extractor.extract(raw_title, llm_pool)
        
        # Merge results, prioritizing Tier 2 technical attributes
        tech_attrs = json.dumps(t2_res.technical_attributes, sort_keys=True) if t2_res.technical_attributes else ""
        
        fp = _compute_universal_fingerprint(
            brand=t2_res.brand or t1_spec.brand,
            model_family=t2_res.model_family or t1_spec.model_family,
            sub_model=t2_res.sub_model or t1_spec.sub_model,
            ram_gb=t2_res.ram_gb or t1_spec.ram_gb,
            storage_gb=t2_res.storage_gb or t1_spec.storage_gb,
            region_code=t2_res.region_code or t1_spec.region_code,
            condition=t2_res.condition or t1_spec.condition,
            technical_attributes=t2_res.technical_attributes
        )
        
        return NormalizedSpec(
            brand=(t2_res.brand or t1_spec.brand).lower().strip(),
            model_family=(t2_res.model_family or t1_spec.model_family).lower().strip(),
            sub_model=(t2_res.sub_model or t1_spec.sub_model).lower().strip(),
            ram_gb=t2_res.ram_gb or t1_spec.ram_gb,
            storage_gb=t2_res.storage_gb or t1_spec.storage_gb,
            region_code=(t2_res.region_code or t1_spec.region_code).lower().strip(),
            spec_fingerprint=fp,
            condition=(t2_res.condition or t1_spec.condition).lower().strip(),
            technical_attributes=tech_attrs
        )

    def normalize(
        self,
        raw_title: str,
        brand_hint: Optional[str] = None,
    ) -> NormalizedSpec:
        """Normalize ``raw_title`` into a ``NormalizedSpec``."""
        cleaned, condition = self._strip_noise(raw_title)
        brand = self._extract_brand(cleaned, brand_hint)
        sub_model = self._extract_sub_model(cleaned)
        ram_gb, storage_gb = self._extract_memory(cleaned)
        region_code = self._extract_region(cleaned)
        model_family = self._extract_model_family(cleaned, brand, sub_model)
        non_phone_attrs = self._extract_non_phone_attributes(cleaned)

        # For non-phone categories (Chargers, Audio, Components) the brand-model
        # chain may be thin or absent. If model_family is empty or very short,
        # fall back to technical-attribute fingerprinting so that two physically
        # different accessories (65W vs 100W, 2C vs 1C) always produce distinct
        # fingerprints even when sold under the same brand name.
        if not model_family or len(model_family.strip()) < 3:
            if non_phone_attrs:
                model_family = non_phone_attrs

        return NormalizedSpec.from_parts(
            brand=brand,
            model_family=model_family,
            sub_model=sub_model,
            ram_gb=ram_gb,
            storage_gb=storage_gb,
            region_code=region_code,
            condition=condition,
            technical_attributes=non_phone_attrs,
        )

    # ------------------------------------------------------------------ #
    # Internal extraction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_noise(title: str) -> tuple[str, str]:
        """Normalize Unicode NFKC, extract condition tag, and remove noise."""
        if not title:
            return "", "new"
        # Rule #31: Unicode NFKC normalization
        s = unicodedata.normalize("NFKC", title)

        # Rule #27 & #36: Extract condition tag before stripping phrases
        condition = "new"
        for pat, cond_val in _CONDITION_PATTERNS:
            if re.search(pat, s, re.IGNORECASE):
                condition = cond_val
                break

        s = _EMOJI_RE.sub(" ", s)
        s = _JUNK_RE.sub(" ", s)
        s = _PARENS_NOISE_RE.sub(" ", s)
        s = _DECO_RE.sub(" ", s)
        return _WS_RE.sub(" ", s).strip(), condition

    @staticmethod
    def _extract_brand(cleaned: str, brand_hint: Optional[str]) -> str:
        m = _BRAND_RE.search(cleaned)
        if m:
            return m.group(1).lower()
        if brand_hint:
            return brand_hint.strip().lower()
        return ""

    @staticmethod
    def _extract_sub_model(cleaned: str) -> str:
        """Return the first matching sub-model modifier (longest match wins)."""
        for pattern, canonical in _SUBMODEL_RES:
            if pattern.search(cleaned):
                return canonical.lower()
        return ""

    @staticmethod
    def _extract_memory(cleaned: str) -> tuple[Optional[int], Optional[int]]:
        """Return (ram_gb, storage_gb). Both may be None."""
        m = _RAM_STORAGE_RE.search(cleaned)
        if m:
            if m.group(1) and m.group(2):  # first alternation: Xgb/Ygb
                return int(m.group(1)), int(m.group(2))
            if m.group(3) and m.group(4):  # second alternation: X/Ygb
                return int(m.group(3)), int(m.group(4))
        # No RAM/storage pair — look for a lone storage spec.
        m2 = _STORAGE_ONLY_RE.search(cleaned)
        if m2:
            return None, int(m2.group(1))
        return None, None

    @staticmethod
    def _extract_region(cleaned: str) -> str:
        m = _REGION_RE.search(cleaned)
        return m.group(1).lower() if m else ""

    @staticmethod
    def _extract_model_family(cleaned: str, brand: str, sub_model: str) -> str:
        """Best-effort model family extraction."""
        s = cleaned
        if brand:
            s = re.sub(r"\b" + re.escape(brand) + r"\b", "", s, flags=re.IGNORECASE)
        if sub_model:
            s = re.sub(r"\b" + re.escape(sub_model) + r"\b", "", s, flags=re.IGNORECASE)
        s = _RAM_STORAGE_RE.sub("", s)
        s = _STORAGE_ONLY_RE.sub("", s)
        s = _REGION_RE.sub("", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = _WS_RE.sub(" ", s).strip().lower()
        tokens = s.split()
        return " ".join(tokens[:5])

    @staticmethod
    def _extract_non_phone_attributes(cleaned: str) -> str:
        """Extract discriminating technical attributes for accessories/components.

        Handles:
        1. Chargers: wattage (e.g. 65W, 100W, 140W), port configurations
           (e.g. 2C1A, 2C, 1A), charging protocol tags (GaN5, PD3.1, PPS, Qi2).
        2. Audio: battery hours, ANC, Bluetooth codecs (LDAC, aptX, AAC).
        3. PC Components: GPU chipset (RTX 4090, RX 7900 XTX), VRAM (16GB VRAM),
           chipset (B650, Z790), DDR5/DDR4, Gen4/Gen5 NVMe.
        """
        parts: list[str] = []

        # GPU series
        gpu_m = re.search(r"\b(rtx\s*40\d0(?:\s*ti)?|rtx\s*30\d0(?:\s*ti)?|rx\s*7\d00(?:\s*xtx|\s*xt)?)\b", cleaned, re.IGNORECASE)
        if gpu_m:
            parts.append(re.sub(r"\s+", "", gpu_m.group(1).lower()))

        # PC Chipset / Socket
        chipset_m = re.search(r"\b(b650[e]?|x670[e]?|z790|b760|x870[e]?|z890|am5|lga1700)\b", cleaned, re.IGNORECASE)
        if chipset_m:
            parts.append(chipset_m.group(1).lower())

        # VRAM
        vram_m = re.search(r"(\d{1,2})\s*gb\s*vram\b", cleaned, re.IGNORECASE)
        if vram_m:
            parts.append(f"{vram_m.group(1)}gb_vram")

        # Memory type
        mem_m = re.search(r"\b(ddr5|ddr4)\b", cleaned, re.IGNORECASE)
        if mem_m:
            parts.append(mem_m.group(1).lower())

        # NVMe / SSD Generation
        ssd_m = re.search(r"\b(gen5|gen4|nvme|m\.2)\b", cleaned, re.IGNORECASE)
        if ssd_m:
            parts.append(ssd_m.group(1).lower().replace(".", ""))

        # Wattage: "65W", "100 W", "45w", "140W", "240W"
        watt_m = re.search(r"(\d{1,4})\s*[Ww]\b", cleaned)
        if watt_m:
            parts.append(f"{watt_m.group(1)}w")

        # Voltage: "110V", "120V", "220V", "230V", "240V" (prevents false merges between 110V and 230V SKUs)
        volt_m = re.search(r"\b(110\s*v|120\s*v|220\s*v|230\s*v|240\s*v)\b", cleaned, re.IGNORECASE)
        if volt_m:
            parts.append(re.sub(r"\s+", "", volt_m.group(1).lower()))

        # Port configuration: 2C1A, 3C1A, 2C, 1C
        port_combo_m = re.search(r"\b(\d[Cc]\d[Aa]|\d[Cc]|\d[Aa])\b", cleaned)
        if port_combo_m:
            parts.append(port_combo_m.group(1).lower())
        else:
            c_port_m = re.search(r"(\d)\s*[-x]?\s*(?:usb[-\s]?c|type[-\s]?c)\b", cleaned, re.IGNORECASE)
            if c_port_m:
                parts.append(f"{c_port_m.group(1)}c")
            elif re.search(r"\busb[-\s]?c\b|\btype[-\s]?c\b", cleaned, re.IGNORECASE):
                parts.append("1c")

            a_port_m = re.search(r"(\d)\s*[-x]?\s*usb[-\s]?a\b", cleaned, re.IGNORECASE)
            if a_port_m:
                parts.append(f"{a_port_m.group(1)}a")
            elif re.search(r"\busb[-\s]?a\b", cleaned, re.IGNORECASE):
                parts.append("1a")

        # Charging technology & Audio protocol tags
        tech_tags = [
            (r"\bgan\s*5\b",       "gan5"),
            (r"\bgan\s*3\b",       "gan3"),
            (r"\bgan\b",           "gan"),
            (r"\bpd\s*3\.1\b",     "pd31"),
            (r"\bpd\s*3\.0\b",     "pd30"),
            (r"\bpd\b",            "pd"),
            (r"\bpps\b",           "pps"),
            (r"\bqc\s*4\b",        "qc4"),
            (r"\bqi\s*2\b",        "qi2"),
            (r"\bqi\b",            "qi"),
            (r"\bmagsafe\b",       "magsafe"),
            (r"\bldac\b",          "ldac"),
            (r"\baptx\b",          "aptx"),
            (r"\bhybrid\s*anc\b",  "anc"),
            (r"\banc\b|\bnoise\s+cancell", "anc"),
            (r"\btws\b",           "tws"),
            (r"\bbluetooth\s*5\.4\b|\bbt\s*5\.4\b", "bt54"),
            (r"\bbluetooth\s*5\.3\b|\bbt\s*5\.3\b", "bt53"),
            (r"\bbluetooth\b",     "bt"),
        ]
        for pat, tag in tech_tags:
            if re.search(pat, cleaned, re.IGNORECASE) and tag not in parts:
                parts.append(tag)

        # Capacity for power banks: "20000mAh", "20000 mah"
        mah_m = re.search(r"(\d{4,6})\s*m[Aa][Hh]\b", cleaned)
        if mah_m:
            parts.append(f"{mah_m.group(1)}mah")

        # Sort parts alphabetically (Rule #39) so attribute order never creates duplicate fingerprints
        return "_".join(sorted(set(parts))) if parts else ""


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def _compute_universal_fingerprint(
    brand: str,
    model_family: str,
    sub_model: str,
    ram_gb: Optional[int],
    storage_gb: Optional[int],
    region_code: str,
    condition: str,
    technical_attributes: Dict[str, Any],
) -> str:
    # Rule 39: Alphabetically sort structural JSON key-values
    tech_str = json.dumps(technical_attributes, sort_keys=True) if technical_attributes else ""
    canonical = (
        f"{brand.lower().strip()}"
        f"|{model_family.lower().strip()}"
        f"|{sub_model.lower().strip()}"
        f"|{ram_gb if ram_gb is not None else ''}"
        f"|{storage_gb if storage_gb is not None else ''}"
        f"|{region_code.lower().strip()}"
    )
    if condition != "new" or tech_str:
        canonical += f"|{condition.lower().strip()}|{tech_str}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_fingerprint(
    brand: str,
    model_family: str,
    sub_model: str,
    ram_gb: Optional[int],
    storage_gb: Optional[int],
    region_code: str,
    condition: str = "new",
    technical_attributes: str = "",
) -> str:
    """Compute a deterministic SHA-256 fingerprint from normalized spec fields.

    Maintains backward compatibility with base fingerprint formula when condition is "new"
    and technical_attributes is empty.
    """
    canonical = (
        f"{brand.lower().strip()}"
        f"|{model_family.lower().strip()}"
        f"|{sub_model.lower().strip()}"
        f"|{ram_gb if ram_gb is not None else ''}"
        f"|{storage_gb if storage_gb is not None else ''}"
        f"|{region_code.lower().strip()}"
    )
    if condition != "new" or technical_attributes:
        canonical += f"|{condition.lower().strip()}|{technical_attributes.lower().strip()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

normalizer = SpecNormalizer()

