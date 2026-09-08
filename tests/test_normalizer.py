"""Parametrized unit tests for the SpecNormalizer anti-collision engine.

These tests verify that:
1. Variant siblings (Pro vs Pro Max) ALWAYS produce different fingerprints.
2. Marketing noise stripping is aggressive but non-destructive.
3. Memory extraction is reliable across common retailer title formats.
4. The fingerprint is deterministic (same input → same hash always).

No network calls, no database, no LLM — pure unit tests.
"""

import hashlib
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(fn):
                    fn._parametrize = (argnames, argvalues)
                    return fn
                return decorator

    pytest = _MockPytest()

# Allow importing from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from normalizer import SpecNormalizer, _compute_fingerprint


@pytest.fixture(scope="module")
def norm():
    return SpecNormalizer()


# ---------------------------------------------------------------------------
# Anti-collision: variant siblings must never share a fingerprint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title_a,title_b", [
    (
        "Apple iPhone 16 Pro 256GB",
        "Apple iPhone 16 Pro Max 256GB",
    ),
    (
        "Samsung Galaxy S25 Ultra 12/512GB",
        "Samsung Galaxy S25 Plus 12/512GB",
    ),
    (
        "iPhone 16 Pro 128GB Natural Titanium",
        "iPhone 16 Pro Max 128GB Natural Titanium",
    ),
    (
        "Google Pixel 9 Pro",
        "Google Pixel 9 Pro Fold",
    ),
    (
        "OnePlus 13 12/256GB",
        "OnePlus 13 Pro 12/256GB",
    ),
])
def test_variant_siblings_have_different_fingerprints(norm, title_a, title_b):
    """Core invariant: variant siblings MUST never share a fingerprint."""
    spec_a = norm.normalize(title_a)
    spec_b = norm.normalize(title_b)
    assert spec_a.spec_fingerprint != spec_b.spec_fingerprint, (
        f"COLLISION: '{title_a}' and '{title_b}' share fingerprint {spec_a.spec_fingerprint}"
    )


# ---------------------------------------------------------------------------
# Noise stripping: noisy and clean titles must share a fingerprint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noisy_title,clean_title", [
    (
        "🔥 Apple iPhone 16 Pro Max 256GB - FREE DELIVERY 🔥 [BEST PRICE!]",
        "Apple iPhone 16 Pro Max 256GB",
    ),
    (
        "Samsung Galaxy S25 Ultra 12/512GB ★★ HOT DEAL 1 Year Warranty ★★",
        "Samsung Galaxy S25 Ultra 12/512GB",
    ),
    (
        "iPhone 16 Pro 128GB | Sealed | Genuine | Free Shipping | Koko Pay",
        "Apple iPhone 16 Pro 128GB",  # brand injected via brand_hint
    ),
    (
        "SAMSUNG GALAXY A55 5G 8/256GB (ORIGINAL, AUTHORIZED DEALER)",
        "Samsung Galaxy A55 5G 8/256GB",
    ),
])
def test_noise_stripped_titles_share_fingerprint(norm, noisy_title, clean_title):
    """Noisy and clean titles for the same product must produce the same fingerprint."""
    brand_hint = None
    if "iphone 16 pro 128" in noisy_title.lower() and "apple" not in noisy_title.lower():
        brand_hint = "Apple"  # simulate LLM brand extraction
    spec_noisy = norm.normalize(noisy_title, brand_hint=brand_hint)
    spec_clean = norm.normalize(clean_title)
    assert spec_noisy.spec_fingerprint == spec_clean.spec_fingerprint, (
        f"Fingerprint mismatch:\n  noisy: {noisy_title!r} -> {spec_noisy.spec_fingerprint}\n"
        f"  clean: {clean_title!r} -> {spec_clean.spec_fingerprint}"
    )


# ---------------------------------------------------------------------------
# Field extraction correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,brand_hint,expected", [
    (
        "Apple iPhone 16 Pro Max 256GB",
        None,
        {"brand": "apple", "sub_model": "pro max", "storage_gb": 256, "ram_gb": None},
    ),
    (
        "Samsung Galaxy S25 Ultra 12/512GB",
        None,
        {"brand": "samsung", "sub_model": "ultra", "ram_gb": 12, "storage_gb": 512},
    ),
    (
        "Google Pixel 9 Pro Fold 16/256GB",
        None,
        {"brand": "google", "sub_model": "pro", "ram_gb": 16, "storage_gb": 256},
    ),
    (
        "OnePlus 13 12/256GB",
        "OnePlus",
        {"brand": "oneplus", "sub_model": "", "ram_gb": 12, "storage_gb": 256},
    ),
    (
        "Xiaomi Redmi Note 14 Pro+ 8/256GB CN",
        None,
        {"brand": "xiaomi", "storage_gb": 256, "ram_gb": 8, "region_code": "cn"},
    ),
    (
        "Samsung Galaxy A55 5G 256GB",
        None,
        {"brand": "samsung", "sub_model": "", "storage_gb": 256, "ram_gb": None},
    ),
    (
        "Apple iPhone SE 64GB",
        None,
        {"brand": "apple", "sub_model": "se", "storage_gb": 64},
    ),
    (
        "Samsung Galaxy Z Fold 6 512GB",
        None,
        {"brand": "samsung", "sub_model": "fold", "storage_gb": 512},
    ),
])
def test_field_extraction(norm, title, brand_hint, expected):
    spec = norm.normalize(title, brand_hint=brand_hint)
    for key, expected_val in expected.items():
        actual = getattr(spec, key)
        assert actual == expected_val, (
            f"Field '{key}' mismatch for title {title!r}: "
            f"expected {expected_val!r}, got {actual!r}"
        )


# ---------------------------------------------------------------------------
# Determinism: same input always produces same fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_is_deterministic(norm):
    title = "Apple iPhone 16 Pro Max 256GB"
    results = [norm.normalize(title).spec_fingerprint for _ in range(10)]
    assert len(set(results)) == 1, "Fingerprint is not deterministic"


def test_fingerprint_matches_manual_computation(norm):
    """Verify the fingerprint formula exactly matches the documented spec."""
    spec = norm.normalize("Apple iPhone 16 Pro Max 256GB")
    expected = _compute_fingerprint(
        spec.brand,
        spec.model_family,
        spec.sub_model,
        spec.ram_gb,
        spec.storage_gb,
        spec.region_code,
    )
    assert spec.spec_fingerprint == expected


def test_fingerprint_is_sha256_length(norm):
    spec = norm.normalize("Apple iPhone 16 Pro Max 256GB")
    assert len(spec.spec_fingerprint) == 64, "SHA-256 hex digest should be 64 chars"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_title_does_not_crash(norm):
    spec = norm.normalize("")
    assert isinstance(spec.spec_fingerprint, str)
    assert len(spec.spec_fingerprint) == 64


def test_emoji_only_title(norm):
    spec = norm.normalize("🔥🔥🔥 🇬🇧")
    assert isinstance(spec.spec_fingerprint, str)


def test_region_extraction(norm):
    spec = norm.normalize("Samsung Galaxy S25 Ultra 256GB CN")
    assert spec.region_code == "cn"


def test_no_region_defaults_to_empty(norm):
    spec = norm.normalize("Apple iPhone 16 Pro Max 256GB")
    assert spec.region_code == ""


def test_brand_hint_fallback(norm):
    # Brand not in known list but provided as hint.
    spec = norm.normalize("XPhone Ultra 256GB", brand_hint="XBrand")
    assert spec.brand == "xbrand"


def test_pro_max_not_confused_with_pro_max_in_storage(norm):
    """'Pro Max' should never be stripped if it's part of the model name."""
    spec_pro = norm.normalize("iPhone 16 Pro 256GB")
    spec_pro_max = norm.normalize("iPhone 16 Pro Max 256GB")
    # sub_model must differ.
    assert spec_pro.sub_model == "pro"
    assert spec_pro_max.sub_model == "pro max"
    assert spec_pro.spec_fingerprint != spec_pro_max.spec_fingerprint


def test_charger_accessories_distinct_fingerprints(norm):
    """Different wattage and port configs must never collide."""
    spec_65w = norm.normalize("Baseus GaN5 Pro 65W Fast Charger 2C1A")
    spec_100w = norm.normalize("Baseus GaN5 100W Fast Charger 2C2A")
    assert spec_65w.spec_fingerprint != spec_100w.spec_fingerprint
    assert "65w" in spec_65w.technical_attributes or "65w" in spec_65w.model_family
    assert "100w" in spec_100w.technical_attributes or "100w" in spec_100w.model_family


def test_condition_overrides_prevent_false_merges(norm):
    """Refurbished and Used listings must never merge with brand new retail SKUs."""
    spec_new = norm.normalize("Apple iPhone 15 128GB Sealed")
    spec_refurb = norm.normalize("Apple iPhone 15 128GB (Refurbished)")
    spec_used = norm.normalize("Apple iPhone 15 128GB Pre-Owned")

    assert spec_new.condition == "new"
    assert spec_refurb.condition == "refurbished"
    assert spec_used.condition == "used"
    assert spec_new.spec_fingerprint != spec_refurb.spec_fingerprint
    assert spec_new.spec_fingerprint != spec_used.spec_fingerprint
    assert spec_refurb.spec_fingerprint != spec_used.spec_fingerprint


def test_pc_components_distinct_fingerprints(norm):
    """RTX 4090 and RTX 4080 graphics cards must have distinct fingerprints."""
    spec_4090 = norm.normalize("Asus ROG Strix GeForce RTX 4090 24GB VRAM GDDR6X")
    spec_4080 = norm.normalize("Asus ROG Strix GeForce RTX 4080 16GB VRAM GDDR6X")
    assert spec_4090.spec_fingerprint != spec_4080.spec_fingerprint


def test_audio_anc_attributes(norm):
    """Audio devices with ANC and codecs extract technical attributes."""
    spec_anc = norm.normalize("Sony WF-1000XM5 Wireless Noise Cancelling Earbuds LDAC")
    assert "anc" in spec_anc.technical_attributes
    assert "ldac" in spec_anc.technical_attributes


def test_voltage_sku_distinct_fingerprints(norm):
    """110V US appliances and 230V local/UK appliances must never collide."""
    spec_110v = norm.normalize("Dyson Airwrap Multi-Styler Complete 110V US Model")
    spec_230v = norm.normalize("Dyson Airwrap Multi-Styler Complete 230V UK Model")
    assert spec_110v.spec_fingerprint != spec_230v.spec_fingerprint
    assert "110v" in spec_110v.technical_attributes
    assert "230v" in spec_230v.technical_attributes



if __name__ == "__main__":
    import inspect
    n = SpecNormalizer()
    passed = 0
    failed = 0
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and name.startswith("test_"):
            if hasattr(obj, "_parametrize"):
                argnames_str, argvalues = obj._parametrize
                argnames = [a.strip() for a in argnames_str.split(",")]
                for val_tuple in argvalues:
                    if not isinstance(val_tuple, tuple):
                        val_tuple = (val_tuple,)
                    kwargs = dict(zip(argnames, val_tuple))
                    try:
                        sig = inspect.signature(obj)
                        if "norm" in sig.parameters:
                            kwargs["norm"] = n
                        obj(**kwargs)
                        passed += 1
                    except Exception as e:
                        print(f"FAIL: {name}({kwargs}): {e}")
                        failed += 1
            else:
                try:
                    sig = inspect.signature(obj)
                    kwargs = {"norm": n} if "norm" in sig.parameters else {}
                    obj(**kwargs)
                    passed += 1
                except Exception as e:
                    print(f"FAIL: {name}: {e}")
                    failed += 1
    print(f"\nResult: {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)
