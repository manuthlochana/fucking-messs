"""Typed async query helpers for KALA-BALANA.

All public functions accept an optional ``pool`` parameter. When it is
``None`` (dry-run mode) every function short-circuits and returns the
safe no-op value (None, [], False) without raising.

All inserts/updates use ``ON CONFLICT … DO UPDATE`` for idempotency so
retrying a failed crawl batch never produces duplicate rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logging_utils import log

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_str() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Merchant helpers
# ---------------------------------------------------------------------------

async def upsert_merchant(
    pool: Optional[object],
    *,
    domain: str,
    display_name: Optional[str] = None,
) -> Optional[str]:
    """Insert or return a merchant row by domain. Returns the merchant UUID."""
    if pool is None:
        return None
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        INSERT INTO merchants (id, domain, display_name)
        VALUES ($1, $2, $3)
        ON CONFLICT (domain) DO UPDATE
            SET display_name = EXCLUDED.display_name,
                updated_at   = NOW()
        RETURNING id::text
        """,
        _uuid_str(), domain, display_name or domain,
    )
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Canonical product helpers
# ---------------------------------------------------------------------------

async def lookup_by_fingerprint(
    pool: Optional[object],
    fingerprint: str,
) -> Optional[str]:
    """Return the canonical product UUID for a given spec_fingerprint, or None."""
    if pool is None:
        return None
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        "SELECT id::text FROM canonical_products WHERE spec_fingerprint = $1",
        fingerprint,
    )
    return row["id"] if row else None


async def insert_canonical_product(
    pool: Optional[object],
    *,
    spec_fingerprint: str,
    brand: Optional[str],
    model_family: Optional[str],
    sub_model: Optional[str],
    storage_gb: Optional[int],
    ram_gb: Optional[int],
    region_code: str,
    raw_title: str,
    clean_title: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Insert a new canonical product row. Returns the new UUID."""
    if pool is None:
        return None
    import json

    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        INSERT INTO canonical_products
            (id, spec_fingerprint, brand, model_family, sub_model,
             storage_gb, ram_gb, region_code, raw_title, clean_title, attributes)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb)
        ON CONFLICT (spec_fingerprint) DO UPDATE
            SET updated_at = NOW()
        RETURNING id::text
        """,
        _uuid_str(), spec_fingerprint, brand, model_family, sub_model,
        storage_gb, ram_gb, region_code, raw_title, clean_title,
        json.dumps(attributes or {}),
    )
    return row["id"] if row else None


async def update_pipeline_status(
    pool: Optional[object],
    product_id: str,
    status: str,
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the pipeline_status of a canonical product and optionally merge attributes."""
    if pool is None:
        return
    import json

    if extra_attributes:
        await pool.execute(  # type: ignore[attr-defined]
            """
            UPDATE canonical_products
            SET pipeline_status = $1::pipeline_status,
                attributes = attributes || $2::jsonb,
                updated_at = NOW()
            WHERE id = $3::uuid
            """,
            status, json.dumps(extra_attributes), product_id,
        )
    else:
        await pool.execute(  # type: ignore[attr-defined]
            """
            UPDATE canonical_products
            SET pipeline_status = $1::pipeline_status,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            status, product_id,
        )


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

async def upsert_listing(
    pool: Optional[object],
    *,
    product_id: str,
    merchant_id: str,
    listing_url: str,
    price_lkr: float,
    warranty_tier: str = "unstated",
    in_stock: bool = True,
) -> Optional[str]:
    """Insert or update a listing row. Returns the listing UUID."""
    if pool is None:
        return None
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        INSERT INTO listings
            (id, product_id, merchant_id, listing_url, current_price_lkr,
             warranty_tier, in_stock)
        VALUES ($1,$2::uuid,$3::uuid,$4,$5,$6::warranty_tier,$7)
        ON CONFLICT (listing_url) DO UPDATE
            SET current_price_lkr = EXCLUDED.current_price_lkr,
                in_stock          = EXCLUDED.in_stock,
                warranty_tier     = EXCLUDED.warranty_tier,
                updated_at        = NOW()
        RETURNING id::text
        """,
        _uuid_str(), product_id, merchant_id, listing_url,
        price_lkr, warranty_tier, in_stock,
    )
    return row["id"] if row else None


async def log_price_history(
    pool: Optional[object],
    *,
    listing_id: str,
    price_lkr: float,
    in_stock: bool,
) -> None:
    """Append a row to price_history."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO price_history (listing_id, price_lkr, in_stock, scraped_at)
        VALUES ($1::uuid, $2, $3, NOW())
        """,
        listing_id, price_lkr, in_stock,
    )


# ---------------------------------------------------------------------------
# Forensic data helpers
# ---------------------------------------------------------------------------

async def insert_arbitrage_log(
    pool: Optional[object],
    *,
    product_id: str,
    global_msrp_usd: Optional[float],
    cbsl_rate: Optional[float],
    true_landed_cost_lkr: Optional[float],
    merchant_price_lkr: float,
    markup_pct: Optional[float],
    price_label: str,
) -> None:
    """Insert a currency arbitrage log row."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO currency_arbitrage_logs
            (product_id, global_msrp_usd, cbsl_rate, true_landed_cost_lkr,
             merchant_price_lkr, markup_pct, price_label)
        VALUES ($1::uuid,$2,$3,$4,$5,$6,$7)
        """,
        product_id, global_msrp_usd, cbsl_rate, true_landed_cost_lkr,
        merchant_price_lkr, markup_pct, price_label,
    )


async def upsert_defect_dossier(
    pool: Optional[object],
    *,
    product_id: str,
    defects: List[Dict[str, Any]],
    source_count: int,
    confidence_label: str,
    astroturf_risk_score: float,
) -> None:
    """Upsert the canonical defect dossier for a product (one row per product)."""
    if pool is None:
        return
    import json

    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO defect_dossiers
            (canonical_product_id, defects, source_count, confidence_label,
             astroturf_risk_score, generated_at, updated_at)
        VALUES ($1::uuid, $2::jsonb, $3, $4, $5, NOW(), NOW())
        ON CONFLICT (canonical_product_id) DO UPDATE
            SET defects              = EXCLUDED.defects,
                source_count         = EXCLUDED.source_count,
                confidence_label     = EXCLUDED.confidence_label,
                astroturf_risk_score = EXCLUDED.astroturf_risk_score,
                updated_at           = NOW()
        """,
        product_id, json.dumps(defects), source_count,
        confidence_label, astroturf_risk_score,
    )


async def get_defect_dossier(
    pool: Optional[object],
    product_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch the defect dossier for a product, or None if not yet generated."""
    if pool is None:
        return None
    import json

    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        SELECT defects, source_count, confidence_label, astroturf_risk_score, generated_at
        FROM defect_dossiers
        WHERE canonical_product_id = $1::uuid
        """,
        product_id,
    )
    if not row:
        return None
    return {
        "defects": json.loads(row["defects"]) if isinstance(row["defects"], str) else row["defects"],
        "source_count": row["source_count"],
        "confidence_label": row["confidence_label"],
        "astroturf_risk_score": float(row["astroturf_risk_score"]),
        "generated_at": row["generated_at"].isoformat() if row["generated_at"] else None,
    }


# ---------------------------------------------------------------------------
# Forensic queue helpers — durable inter-process serial queue
# ---------------------------------------------------------------------------

async def enqueue_forensic_job(
    pool: Optional[object],
    *,
    product_id: str,
    merchant_id: Optional[str],
    listing_url: str,
) -> None:
    """Enqueue a new forensic analysis job. No-op if pool is None."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO forensic_queue (product_id, merchant_id, listing_url)
        VALUES ($1::uuid, $2::uuid, $3)
        """,
        product_id, merchant_id, listing_url,
    )


async def claim_next_forensic_job(
    pool: Optional[object],
) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest pending forensic job.

    Uses ``SELECT … FOR UPDATE SKIP LOCKED`` so multiple worker processes
    can safely compete without races. Returns None when the queue is empty.
    """
    if pool is None:
        return None
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, product_id::text, merchant_id::text, listing_url
                FROM forensic_queue
                WHERE status = 'pending'
                ORDER BY enqueued_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
            )
            if row is None:
                return None
            await conn.execute(
                """
                UPDATE forensic_queue
                SET status = 'claimed', claimed_at = NOW()
                WHERE id = $1
                """,
                row["id"],
            )
    return dict(row)


async def complete_forensic_job(
    pool: Optional[object],
    job_id: int,
) -> None:
    """Mark a claimed job as successfully completed."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        UPDATE forensic_queue
        SET status = 'done', completed_at = NOW()
        WHERE id = $1
        """,
        job_id,
    )


async def fail_forensic_job(
    pool: Optional[object],
    job_id: int,
    error_msg: str,
) -> None:
    """Mark a claimed job as failed with an error message."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        UPDATE forensic_queue
        SET status = 'failed', error_msg = $2, completed_at = NOW()
        WHERE id = $1
        """,
        job_id, error_msg[:2000],  # cap to avoid over-long error blobs
    )



async def upsert_merchant_forensics(
    pool: Optional[object],
    *,
    merchant_id: str,
    bnpl_surcharge_pct: Optional[float],
    cc_surcharge_pct: Optional[float],
    dark_patterns: List[str],
    koko_mintpay_hidden_fees: Dict[str, Any],
    raw_audit_data: Dict[str, Any],
) -> None:
    """Upsert merchant forensic audit data."""
    if pool is None:
        return
    import json

    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO merchant_forensics
            (id, merchant_id, bnpl_surcharge_pct, cc_surcharge_pct,
             dark_patterns, koko_mintpay_hidden_fees, raw_audit_data)
        VALUES ($1,$2::uuid,$3,$4,$5::jsonb,$6::jsonb,$7::jsonb)
        ON CONFLICT (merchant_id) DO UPDATE
            SET bnpl_surcharge_pct       = EXCLUDED.bnpl_surcharge_pct,
                cc_surcharge_pct         = EXCLUDED.cc_surcharge_pct,
                dark_patterns            = EXCLUDED.dark_patterns,
                koko_mintpay_hidden_fees = EXCLUDED.koko_mintpay_hidden_fees,
                raw_audit_data           = EXCLUDED.raw_audit_data,
                audited_at               = NOW()
        """,
        _uuid_str(), merchant_id, bnpl_surcharge_pct, cc_surcharge_pct,
        json.dumps(dark_patterns), json.dumps(koko_mintpay_hidden_fees),
        json.dumps(raw_audit_data),
    )
