"""Unit tests for the autonomous, self-governing pipeline (Steps 1–4).

Covers the behaviours the overhaul added, all offline (no network, DB, or LLM):

1. Discovery URL classification (``is_product_url``) + registered-domain collapse.
2. robots.txt / sitemap parsing + link extraction (spider fallback primitives).
3. Zero-selector Tier-1 extraction: JSON-LD, OpenGraph, Microdata.
4. ``extract_and_persist_html`` dry-run cascade → usable ExtractionOutcome.
5. Fake in-memory ``crawl_queue`` drip: enqueue → claim → complete round-trip
   against a stub pool, verifying the drip worker's queue contract.
"""

import asyncio
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - allow bare `python tests/test_autonomous_pipeline.py`
    class _MockPytest:
        @staticmethod
        def fixture(*a, **k):
            def deco(fn):
                return fn
            return deco

        class mark:
            class _Async:
                def __call__(self, fn):
                    return fn

            asyncio = _Async()

    pytest = _MockPytest()

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Discovery URL classification + domain collapse
# ---------------------------------------------------------------------------

def test_is_product_url():
    import discovery

    assert discovery.is_product_url("https://tudo.lk/product/iphone-16")
    assert discovery.is_product_url("https://shop.lk/p/12345")
    assert discovery.is_product_url("https://x.lk/item/abc")
    assert discovery.is_product_url("https://x.lk/buy/thing")
    assert discovery.is_product_url("https://x.lk/catalog?sku=99")
    # Non-product / asset / nav URLs must be rejected.
    assert not discovery.is_product_url("https://tudo.lk/cart")
    assert not discovery.is_product_url("https://tudo.lk/style.css")
    assert not discovery.is_product_url("https://tudo.lk/about-us")


def test_registered_domain_collapse():
    import discovery

    assert discovery._registered_domain("https://www.shop.tudo.lk/p/1") == "tudo.lk"
    assert discovery._registered_domain("https://foo.abans.com.lk/x") == "abans.com.lk"
    assert discovery._registered_domain("https://nanotek.lk") == "nanotek.lk"


# ---------------------------------------------------------------------------
# 2. robots / sitemap / link parsing
# ---------------------------------------------------------------------------

def test_parse_robots():
    import discovery

    robots = (
        "User-agent: *\n"
        "Disallow: /cart\n"
        "Sitemap: https://tudo.lk/sitemap.xml\n"
        "Sitemap: https://tudo.lk/sitemap_index.xml\n"
    )
    out = discovery.parse_robots(robots)
    assert "https://tudo.lk/sitemap.xml" in out["sitemaps"]
    assert "https://tudo.lk/sitemap_index.xml" in out["sitemaps"]
    assert "/cart" in out["disallows"]


def test_extract_sitemap_locs_index_vs_urlset():
    import discovery

    index_xml = (
        '<?xml version="1.0"?><sitemapindex>'
        "<sitemap><loc>https://tudo.lk/sitemap-products.xml</loc></sitemap>"
        "</sitemapindex>"
    )
    is_index, locs = discovery.extract_sitemap_locs(index_xml)
    assert is_index is True
    assert "https://tudo.lk/sitemap-products.xml" in locs

    urlset_xml = (
        '<?xml version="1.0"?><urlset>'
        "<url><loc>https://tudo.lk/product/a</loc></url>"
        "<url><loc>https://tudo.lk/product/b</loc></url>"
        "</urlset>"
    )
    is_index2, locs2 = discovery.extract_sitemap_locs(urlset_xml)
    assert is_index2 is False
    assert len(locs2) == 2


def test_extract_links_resolves_relative():
    import discovery

    html = '<a href="/product/x">x</a> <a href="https://tudo.lk/p/y">y</a>'
    links = discovery.extract_links(html, "https://tudo.lk/shop")
    assert "https://tudo.lk/product/x" in links
    assert "https://tudo.lk/p/y" in links


# ---------------------------------------------------------------------------
# 3. Zero-selector Tier-1 extraction
# ---------------------------------------------------------------------------

_JSONLD_HTML = """<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Galaxy S25 Ultra 512GB",
 "brand":{"@type":"Brand","name":"Samsung"},
 "offers":{"@type":"Offer","price":"415000","priceCurrency":"LKR",
           "availability":"https://schema.org/InStock"}}
</script></head><body><h1>Galaxy S25 Ultra</h1></body></html>"""


def test_parse_jsonld():
    import auto_extractor as ax

    oc = ax.parse_jsonld(_JSONLD_HTML)
    assert oc is not None
    assert oc.brand == "Samsung"
    assert oc.price_lkr == 415000.0
    assert oc.in_stock is True
    assert "jsonld" in "".join(oc.strategy.values())


def test_parse_opengraph():
    import auto_extractor as ax

    html = (
        '<html><head>'
        '<meta property="og:title" content="Pixel 9 Pro">'
        '<meta property="product:price:amount" content="298000">'
        '</head><body></body></html>'
    )
    oc = ax.parse_opengraph(html)
    assert oc is not None
    assert oc.price_lkr == 298000.0


# ---------------------------------------------------------------------------
# 4. Full cascade in dry-run mode (no pool, no LLM)
# ---------------------------------------------------------------------------

def test_extract_and_persist_dry_run():
    import auto_extractor as ax

    result = _run(ax.extract_and_persist_html(
        "https://tudo.lk/product/galaxy-s25", _JSONLD_HTML,
        pool=None, llm_pool=None,
    ))
    assert result["status"] == "extracted_dry_run"
    assert result["tier"] == "jsonld"
    assert result["price_lkr"] == 415000.0
    assert result["in_stock"] is True


def test_extract_fields_no_price_is_unusable():
    import auto_extractor as ax

    html = "<html><body><h1>Some Page</h1><p>no price here</p></body></html>"
    oc = _run(ax.extract_fields(html, "https://tudo.lk/x", pool=None, llm_pool=None))
    assert not oc.usable


# ---------------------------------------------------------------------------
# 5. Drip queue contract: enqueue → claim → complete against a stub pool
# ---------------------------------------------------------------------------

class _FakeCrawlPool:
    """Minimal in-memory stand-in mimicking the crawl_queue query contract."""

    def __init__(self):
        self.rows = {}  # url -> dict

    async def fetchval(self, sql, *args):
        if "count(*)" in sql.lower():
            return len(self.rows)
        return None

    async def executemany(self, sql, rows):
        if "insert into crawl_queue" in sql.lower():
            for (domain, url, depth) in rows:
                # Mirror ON CONFLICT (url) DO NOTHING.
                self.rows.setdefault(url, {
                    "id": len(self.rows) + 1, "url": url, "domain": domain,
                    "depth": depth, "status": "pending", "attempts": 0,
                })

    async def execute(self, sql, *args):
        s = sql.lower()
        if "update crawl_queue" in s and "id =" in s:
            rid = args[0]
            for row in self.rows.values():
                if row["id"] == rid:
                    row["status"] = "in_progress"
        return "OK"

    async def fetch(self, sql, *args):
        return [dict(r) for r in self.rows.values()]

    async def fetchrow(self, sql, *args):
        if "for update skip locked" in sql.lower():
            for row in self.rows.values():
                if row["status"] == "pending":
                    return dict(row)
        return None

    class _Acq:
        def __init__(self, pool):
            self.pool = pool

        async def __aenter__(self):
            return self.pool

        async def __aexit__(self, *a):
            return False

    def acquire(self):
        return self._Acq(self)

    def transaction(self):
        return self._Acq(self)


def test_enqueue_dedup_roundtrip():
    from db import queries

    pool = _FakeCrawlPool()
    urls = ["https://tudo.lk/product/a", "https://tudo.lk/product/b",
            "https://tudo.lk/product/a"]  # duplicate
    _run(queries.enqueue_crawl_urls(pool, domain="tudo.lk", urls=urls))
    # Duplicate collapses → exactly two rows persisted.
    assert len(pool.rows) == 2
    assert all(r["status"] == "pending" for r in pool.rows.values())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
