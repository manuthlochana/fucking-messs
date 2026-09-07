"""Integration tests for the KALA-BALANA database layer.

These tests require a live PostgreSQL instance with pgvector. They are
automatically SKIPPED when ``DB_DSN`` is not set in the environment, so
they never block CI without a database.

Run manually:
    export DB_DSN="postgresql://postgres:password@localhost/kala_balana_test"
    pytest tests/test_db.py -v
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest_plugins = ["anyio"]

DB_DSN = os.getenv("DB_DSN", "")
pytestmark = pytest.mark.skipif(
    not DB_DSN,
    reason="DB_DSN not set — skipping DB integration tests",
)


@pytest.fixture(scope="module")
async def pool():
    """Create a test DB pool, run migrations, and clean up on teardown."""
    from db.pool import create_pool, close_pool
    from db.migrations import run_migrations

    p = await create_pool(DB_DSN, max_size=2)
    assert p is not None, "Could not connect to test database"
    await run_migrations(p)
    yield p
    # Teardown: drop test tables would be too destructive in shared DBs.
    # We rely on the test database being disposable.
    await close_pool(p)


@pytest.mark.anyio
async def test_migrations_are_idempotent(pool):
    """Running migrations twice must not raise."""
    from db.migrations import run_migrations
    await run_migrations(pool)  # second run
    # If we reach here without exception, idempotency is confirmed.


@pytest.mark.anyio
async def test_insert_and_lookup_canonical_product(pool):
    from db.queries import insert_canonical_product, lookup_by_fingerprint
    import time

    # Use a unique fingerprint per test run to avoid collisions.
    fingerprint = f"test_fp_{int(time.time() * 1000)}"

    product_id = await insert_canonical_product(
        pool,
        spec_fingerprint=fingerprint,
        brand="apple",
        model_family="iphone 16",
        sub_model="pro max",
        storage_gb=256,
        ram_gb=8,
        region_code="",
        raw_title="Apple iPhone 16 Pro Max 256GB",
        clean_title="iPhone 16 Pro Max 256GB",
    )
    assert product_id is not None, "insert_canonical_product should return a UUID"

    # Lookup by fingerprint.
    found_id = await lookup_by_fingerprint(pool, fingerprint)
    assert found_id == product_id, (
        f"lookup_by_fingerprint returned {found_id!r}, expected {product_id!r}"
    )


@pytest.mark.anyio
async def test_lookup_nonexistent_fingerprint_returns_none(pool):
    from db.queries import lookup_by_fingerprint

    result = await lookup_by_fingerprint(pool, "nonexistent_fingerprint_abc123")
    assert result is None


@pytest.mark.anyio
async def test_upsert_merchant_and_listing(pool):
    from db.queries import insert_canonical_product, upsert_merchant, upsert_listing
    import time

    fp = f"test_listing_fp_{int(time.time() * 1000)}"
    product_id = await insert_canonical_product(
        pool,
        spec_fingerprint=fp,
        brand="samsung",
        model_family="galaxy s25",
        sub_model="ultra",
        storage_gb=512,
        ram_gb=12,
        region_code="",
        raw_title="Samsung Galaxy S25 Ultra 12/512GB",
        clean_title="Galaxy S25 Ultra 12/512GB",
    )

    merchant_id = await upsert_merchant(pool, domain="test-merchant.lk", display_name="Test Merchant")
    assert merchant_id is not None

    listing_id = await upsert_listing(
        pool,
        product_id=product_id,
        merchant_id=merchant_id,
        listing_url=f"https://test-merchant.lk/s25-ultra-{int(time.time())}",
        price_lkr=299999.0,
        warranty_tier="unstated",
        in_stock=True,
    )
    assert listing_id is not None


@pytest.mark.anyio
async def test_price_history_logging(pool):
    from db.queries import (
        insert_canonical_product, upsert_merchant,
        upsert_listing, log_price_history,
    )
    import time

    fp = f"test_price_hist_fp_{int(time.time() * 1000)}"
    product_id = await insert_canonical_product(
        pool, spec_fingerprint=fp,
        brand="google", model_family="pixel 9", sub_model="pro",
        storage_gb=128, ram_gb=12, region_code="",
        raw_title="Google Pixel 9 Pro 128GB",
        clean_title="Pixel 9 Pro 128GB",
    )
    merchant_id = await upsert_merchant(pool, domain=f"test-ph-{int(time.time())}.lk")
    listing_id = await upsert_listing(
        pool, product_id=product_id, merchant_id=merchant_id,
        listing_url=f"https://ph-{int(time.time())}.lk/pixel9pro",
        price_lkr=189999.0,
    )

    # Log two price history entries.
    await log_price_history(pool, listing_id=listing_id, price_lkr=189999.0, in_stock=True)
    await log_price_history(pool, listing_id=listing_id, price_lkr=179999.0, in_stock=True)

    rows = await pool.fetch(
        "SELECT price_lkr FROM price_history WHERE listing_id = $1::uuid ORDER BY scraped_at",
        listing_id,
    )
    prices = [float(r["price_lkr"]) for r in rows]
    assert 189999.0 in prices
    assert 179999.0 in prices


@pytest.mark.anyio
async def test_pipeline_status_update(pool):
    from db.queries import insert_canonical_product, update_pipeline_status
    import time

    fp = f"test_status_fp_{int(time.time() * 1000)}"
    product_id = await insert_canonical_product(
        pool, spec_fingerprint=fp,
        brand="oneplus", model_family="oneplus 13", sub_model="",
        storage_gb=256, ram_gb=12, region_code="",
        raw_title="OnePlus 13 12/256GB",
        clean_title="OnePlus 13 12/256GB",
    )

    await update_pipeline_status(pool, product_id, "phase1_done",
                                  extra_attributes={"test_key": "test_val"})

    row = await pool.fetchrow(
        "SELECT pipeline_status, attributes FROM canonical_products WHERE id = $1::uuid",
        product_id,
    )
    import json
    assert row["pipeline_status"] == "phase1_done"
    attrs = json.loads(row["attributes"])
    assert attrs.get("test_key") == "test_val"
