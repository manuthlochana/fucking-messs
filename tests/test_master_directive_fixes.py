"""Unit tests verifying fixes from the Critical Fix Directive."""

import pytest
import re
from unittest.mock import AsyncMock, MagicMock

from normalizer import SpecNormalizer, _NON_PHONE_CATEGORY_RE
from crawler import AntiFragileCrawler
from schemas import DefectFinding, DefectReport


def test_non_phone_category_keywords_trigger():
    """Verify that non-phone hardware keywords are detected for Tier 2 fallback."""
    test_titles = [
        "Samsung 24 Inch 144Hz Monitor",
        "Keychron K2 Wireless Mechanical Keyboard",
        "Crucial P3 Plus 1TB NVMe PCIe M.2 SSD",
        "DJI Mini 4 Pro Drone Fly More Combo",
        "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "Anker 737 Power Bank 24000mAh 140W",
        "Baseus GaN5 Pro 65W Fast Charger",
        "Logitech MX Master 3S Wireless Mouse",
        "ASUS TUF Gaming GeForce RTX 4070 Ti SUPER 16GB GPU",
    ]
    for title in test_titles:
        assert bool(_NON_PHONE_CATEGORY_RE.search(title)), f"Failed to match non-phone category for: {title}"


@pytest.mark.anyio
async def test_normalize_async_triggers_tier_2_for_monitors():
    """Verify normalize_async triggers Tier 2 UniversalTechSpecExtractor on a monitor."""
    normalizer = SpecNormalizer()
    mock_llm_pool = MagicMock()
    mock_llm_pool.generate_structured = AsyncMock()

    from normalizer import UniversalTechSpec
    mock_llm_pool.generate_structured.return_value = UniversalTechSpec(
        brand="samsung",
        model_family="odyssey g5",
        sub_model="24 inch",
        technical_attributes={"refresh_rate": "144hz", "resolution": "1080p"},
    )

    spec = await normalizer.normalize_async(
        "Samsung 24 Inch 144Hz Monitor",
        llm_pool=mock_llm_pool,
    )

    # Must invoke the LLM extractor
    assert mock_llm_pool.generate_structured.called
    assert spec.brand == "samsung"
    assert spec.model_family == "odyssey g5"
    assert "144hz" in spec.technical_attributes


def test_honeypot_js_purges_elements_from_dom():
    """Verify crawler's injected JS removes elements with el.remove()."""
    from sentry import SmartSentry
    from config import settings
    mock_llm = MagicMock()
    sentry = SmartSentry(mock_llm, settings)
    crawler = AntiFragileCrawler(sentry, mock_llm, settings)

    config_normal = crawler._run_config(stealth=False)
    config_stealth = crawler._run_config(stealth=True)

    assert "el.remove()" in config_normal.js_code
    assert "el.remove()" in config_stealth.js_code
    assert "display === 'none'" in config_stealth.js_code
    assert "opacity === '0'" in config_stealth.js_code
    assert "visibility === 'hidden'" in config_stealth.js_code


def test_delta_poll_regex_extraction():
    """Test regex extraction for price and stock in delta-poll HTML snippets."""
    html_in_stock = """
    <div class="product-info">
        <h1>Samsung Galaxy S24 Ultra</h1>
        <span class="price">Rs. 385,000.00</span>
        <button class="btn btn-cart">Add to Cart</button>
    </div>
    """
    html_out_of_stock = """
    <div class="product-info">
        <h1>Apple iPhone 15 Pro</h1>
        <span class="price">LKR 410,000</span>
        <div class="badge badge-danger">Out of Stock</div>
    </div>
    """

    # Test price extraction
    p1 = re.search(r'(?:Rs\.?|LKR)\s*([\d,]{3,}(?:\.\d{2})?)', html_in_stock, re.I)
    assert p1 is not None
    assert float(p1.group(1).replace(",", "")) == 385000.0

    p2 = re.search(r'(?:Rs\.?|LKR)\s*([\d,]{3,}(?:\.\d{2})?)', html_out_of_stock, re.I)
    assert p2 is not None
    assert float(p2.group(1).replace(",", "")) == 410000.0

    # Test stock detection
    oos_match1 = re.search(r"\b(out of stock|sold out|unavailable|discontinued|no stock)\b", html_in_stock, re.I)
    in_stock1 = re.search(r"\b(in stock|add to cart|buy now|available)\b", html_in_stock, re.I)
    assert in_stock1 is not None and oos_match1 is None

    oos_match2 = re.search(r"\b(out of stock|sold out|unavailable|discontinued|no stock)\b", html_out_of_stock, re.I)
    in_stock2 = re.search(r"\b(in stock|add to cart|buy now|available)\b", html_out_of_stock, re.I)
    assert oos_match2 is not None and in_stock2 is None
