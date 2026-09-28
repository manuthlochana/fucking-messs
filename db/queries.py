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
    global_msrp_usd: Optional[float] = None,
    us_msrp_usd: Optional[float] = None,
    eu_msrp_eur: Optional[float] = None,
    uae_msrp_aed: Optional[float] = None,
    india_msrp_inr: Optional[float] = None,
    cbsl_rate: Optional[float] = None,
    fx_rate: Optional[float] = None,
    tariff_pct_applied: Optional[float] = 18.0,
    true_landed_cost_lkr: Optional[float] = None,
    landed_cost_lkr: Optional[float] = None,
    merchant_price_lkr: float,
    local_price_lkr: Optional[float] = None,
    markup_pct: Optional[float] = None,
    margin_pct: Optional[float] = None,
    price_label: str = "unknown",
    classification: Optional[str] = None,
) -> None:
    """Insert a currency arbitrage log row with multi-currency and landed cost support."""
    if pool is None:
        return
    effective_us_msrp = us_msrp_usd or global_msrp_usd
    effective_fx = fx_rate or cbsl_rate
    effective_landed = landed_cost_lkr or true_landed_cost_lkr
    effective_local = local_price_lkr or merchant_price_lkr
    effective_margin = margin_pct or markup_pct
    effective_class = classification or price_label

    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO currency_arbitrage_logs
            (product_id, global_msrp_usd, us_msrp_usd, eu_msrp_eur, uae_msrp_aed,
             india_msrp_inr, cbsl_rate, fx_rate, tariff_pct_applied, true_landed_cost_lkr,
             landed_cost_lkr, merchant_price_lkr, local_price_lkr, markup_pct,
             margin_pct, price_label, classification, logged_at)
        VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,NOW())
        """,
        product_id, global_msrp_usd or effective_us_msrp, effective_us_msrp,
        eu_msrp_eur, uae_msrp_aed, india_msrp_inr, effective_fx, effective_fx,
        tariff_pct_applied, effective_landed, effective_landed,
        effective_local, effective_local, effective_margin, effective_margin,
        price_label, effective_class,
    )


async def upsert_defect_dossier(
    pool: Optional[object],
    *,
    product_id: str,
    defects: List[Dict[str, Any]],
    source_count: int,
    confidence_label: str,
    astroturf_risk_score: float,
    sponsored_content_ratio: float = 0.0,
) -> None:
    """Upsert the canonical defect dossier for a product (one row per product)."""
    if pool is None:
        return
    import json

    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO defect_dossiers
            (canonical_product_id, defects, source_count, confidence_label,
             astroturf_risk_score, sponsored_content_ratio, generated_at, updated_at)
        VALUES ($1::uuid, $2::jsonb, $3, $4, $5, $6, NOW(), NOW())
        ON CONFLICT (canonical_product_id) DO UPDATE
            SET defects                 = EXCLUDED.defects,
                source_count            = EXCLUDED.source_count,
                confidence_label        = EXCLUDED.confidence_label,
                astroturf_risk_score    = EXCLUDED.astroturf_risk_score,
                sponsored_content_ratio = EXCLUDED.sponsored_content_ratio,
                updated_at              = NOW()
        """,
        product_id, json.dumps(defects), source_count,
        confidence_label, astroturf_risk_score, sponsored_content_ratio,
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
        SELECT defects, source_count, confidence_label, astroturf_risk_score,
               sponsored_content_ratio, generated_at, updated_at
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
        "astroturf_risk_score": float(row["astroturf_risk_score"] or 0.0),
        "sponsored_content_ratio": float(row["sponsored_content_ratio"] or 0.0),
        "generated_at": row["generated_at"].isoformat() if row["generated_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def insert_component_teardown(
    pool: Optional[object],
    *,
    product_id: str,
    component_name: Optional[str] = None,
    observed_revision: Optional[str] = None,
    serial_range_start: Optional[str] = None,
    serial_range_end: Optional[str] = None,
    repairability_score: Optional[float] = None,
    source_url: Optional[str] = None,
    source_type: Optional[str] = "ifixit",
    notes: Optional[str] = None,
    revision_label: Optional[str] = None,
    component_changed: Optional[str] = None,
    change_description: Optional[str] = None,
    teardown_source_url: Optional[str] = None,
    ifixit_score: Optional[float] = None,
    silent_revision_detected: bool = False,
) -> None:
    """Insert an append-only hardware teardown record."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO component_teardowns
            (product_id, component_name, observed_revision, serial_range_start,
             serial_range_end, repairability_score, source_url, source_type,
             notes, revision_label, component_changed, change_description,
             teardown_source_url, ifixit_score, silent_revision_detected, recorded_at)
        VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,NOW())
        """,
        product_id,
        component_name or component_changed or "component",
        observed_revision or revision_label or "initial",
        serial_range_start, serial_range_end,
        repairability_score or ifixit_score,
        source_url or teardown_source_url,
        source_type, notes or change_description,
        revision_label or observed_revision or "rev1",
        component_changed or component_name or "part",
        change_description or notes or "",
        teardown_source_url or source_url,
        ifixit_score or repairability_score,
        silent_revision_detected,
    )


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


async def retry_failed_forensic_job(
    pool: Optional[object],
    job_id: int,
) -> bool:
    """Reset a failed or stuck forensic job to pending."""
    if pool is None:
        return False
    res = await pool.execute(  # type: ignore[attr-defined]
        """
        UPDATE forensic_queue
        SET status = 'pending', error_msg = NULL, claimed_at = NULL, completed_at = NULL
        WHERE id = $1
        """,
        job_id,
    )
    return "UPDATE 1" in res


# ---------------------------------------------------------------------------
# Merchant Forensics
# ---------------------------------------------------------------------------

async def upsert_merchant_forensics(
    pool: Optional[object],
    *,
    merchant_id: str,
    physical_presence_verified: Optional[bool] = None,
    physical_addresses: Optional[List[Dict[str, Any]]] = None,
    operating_hours: Optional[Dict[str, Any]] = None,
    business_registry_name: Optional[str] = None,
    business_registry_age_days: Optional[int] = None,
    domain_registration_age_days: Optional[int] = None,
    domain_lineage: Optional[List[str]] = None,
    warranty_claims_verified: Optional[bool] = None,
    surcharge_map: Optional[Dict[str, float]] = None,
    dark_patterns_detected: Optional[List[str]] = None,
    price_devaluation_lag_days_up: Optional[float] = None,
    price_devaluation_lag_days_down: Optional[float] = None,
    reliability_score_override: Optional[float] = None,
    bnpl_surcharge_pct: Optional[float] = None,
    cc_surcharge_pct: Optional[float] = None,
    dark_patterns: Optional[List[str]] = None,
    koko_mintpay_hidden_fees: Optional[Dict[str, Any]] = None,
    raw_audit_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Upsert comprehensive merchant forensic audit data."""
    if pool is None:
        return
    import json

    effective_dark_patterns = dark_patterns_detected or dark_patterns or []

    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO merchant_forensics
            (id, merchant_id, physical_presence_verified, physical_addresses,
             operating_hours, business_registry_name, business_registry_age_days,
             domain_registration_age_days, domain_lineage, warranty_claims_verified,
             surcharge_map, dark_patterns_detected, price_devaluation_lag_days_up,
             price_devaluation_lag_days_down, reliability_score_override,
             bnpl_surcharge_pct, cc_surcharge_pct, dark_patterns,
             koko_mintpay_hidden_fees, raw_audit_data, audited_at)
        VALUES ($1,$2::uuid,$3,$4::jsonb,$5::jsonb,$6,$7,$8,$9::jsonb,$10,$11::jsonb,
                $12::jsonb,$13,$14,$15,$16,$17,$18::jsonb,$19::jsonb,$20::jsonb,NOW())
        ON CONFLICT (merchant_id) DO UPDATE
            SET physical_presence_verified      = COALESCE(EXCLUDED.physical_presence_verified, merchant_forensics.physical_presence_verified),
                physical_addresses              = EXCLUDED.physical_addresses,
                operating_hours                 = EXCLUDED.operating_hours,
                business_registry_name          = COALESCE(EXCLUDED.business_registry_name, merchant_forensics.business_registry_name),
                business_registry_age_days      = COALESCE(EXCLUDED.business_registry_age_days, merchant_forensics.business_registry_age_days),
                domain_registration_age_days    = COALESCE(EXCLUDED.domain_registration_age_days, merchant_forensics.domain_registration_age_days),
                domain_lineage                  = EXCLUDED.domain_lineage,
                warranty_claims_verified        = COALESCE(EXCLUDED.warranty_claims_verified, merchant_forensics.warranty_claims_verified),
                surcharge_map                   = EXCLUDED.surcharge_map,
                dark_patterns_detected          = EXCLUDED.dark_patterns_detected,
                price_devaluation_lag_days_up   = COALESCE(EXCLUDED.price_devaluation_lag_days_up, merchant_forensics.price_devaluation_lag_days_up),
                price_devaluation_lag_days_down = COALESCE(EXCLUDED.price_devaluation_lag_days_down, merchant_forensics.price_devaluation_lag_days_down),
                reliability_score_override      = COALESCE(EXCLUDED.reliability_score_override, merchant_forensics.reliability_score_override),
                bnpl_surcharge_pct              = COALESCE(EXCLUDED.bnpl_surcharge_pct, merchant_forensics.bnpl_surcharge_pct),
                cc_surcharge_pct                = COALESCE(EXCLUDED.cc_surcharge_pct, merchant_forensics.cc_surcharge_pct),
                dark_patterns                   = EXCLUDED.dark_patterns,
                koko_mintpay_hidden_fees        = EXCLUDED.koko_mintpay_hidden_fees,
                raw_audit_data                  = EXCLUDED.raw_audit_data,
                audited_at                      = NOW()
        """,
        _uuid_str(), merchant_id, physical_presence_verified,
        json.dumps(physical_addresses or []),
        json.dumps(operating_hours or {}),
        business_registry_name, business_registry_age_days,
        domain_registration_age_days,
        json.dumps(domain_lineage or []),
        warranty_claims_verified,
        json.dumps(surcharge_map or {}),
        json.dumps(effective_dark_patterns),
        price_devaluation_lag_days_up,
        price_devaluation_lag_days_down,
        reliability_score_override,
        bnpl_surcharge_pct, cc_surcharge_pct,
        json.dumps(effective_dark_patterns),
        json.dumps(koko_mintpay_hidden_fees or {}),
        json.dumps(raw_audit_data or {}),
    )


# ---------------------------------------------------------------------------
# LLM Key Usage Ledger (Rule #99)
# ---------------------------------------------------------------------------

async def log_llm_key_usage(
    pool: Optional[object],
    *,
    provider: str,
    key_identifier: str,
    requests_count: int = 1,
    window_start: datetime,
    window_end: datetime,
    rate_limited: bool = False,
) -> None:
    """Append a record to the LLM key usage ledger."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO llm_key_usage_log
            (provider, key_identifier, requests_count, window_start, window_end, rate_limited)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        provider, key_identifier, requests_count, window_start, window_end, rate_limited,
    )


# ---------------------------------------------------------------------------
# Dashboard queries
# ---------------------------------------------------------------------------

async def get_dashboard_stats(pool: Optional[object]) -> Dict[str, Any]:
    """Return live system counters for the admin dashboard."""
    if pool is None:
        return {
            "products": 0,
            "listings": 0,
            "price_history_rows": 0,
            "defects_count": 0,
            "arbitrage_count": 0,
            "queue_pending": 0,
            "queue_claimed": 0,
            "queue_done": 0,
            "queue_failed": 0,
        }
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        SELECT
            (SELECT COUNT(*) FROM canonical_products) AS products,
            (SELECT COUNT(*) FROM listings) AS listings,
            (SELECT COUNT(*) FROM price_history) AS price_history_rows,
            (SELECT COUNT(*) FROM defect_dossiers) AS defects_count,
            (SELECT COUNT(*) FROM currency_arbitrage_logs) AS arbitrage_count,
            (SELECT COUNT(*) FROM forensic_queue WHERE status = 'pending') AS queue_pending,
            (SELECT COUNT(*) FROM forensic_queue WHERE status = 'claimed') AS queue_claimed,
            (SELECT COUNT(*) FROM forensic_queue WHERE status = 'done') AS queue_done,
            (SELECT COUNT(*) FROM forensic_queue WHERE status = 'failed') AS queue_failed
        """
    )
    return dict(row) if row else {}


async def get_forensic_queue_jobs(
    pool: Optional[object],
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recent forensic queue jobs."""
    if pool is None:
        return []
    if status:
        rows = await pool.fetch(  # type: ignore[attr-defined]
            """
            SELECT fq.id, fq.product_id::text, fq.merchant_id::text, fq.listing_url,
                   fq.status, fq.error_msg, fq.enqueued_at, fq.claimed_at, fq.completed_at,
                   cp.clean_title, cp.brand, cp.model_family
            FROM forensic_queue fq
            LEFT JOIN canonical_products cp ON fq.product_id = cp.id
            WHERE fq.status = $1
            ORDER BY fq.enqueued_at DESC
            LIMIT $2
            """,
            status, limit,
        )
    else:
        rows = await pool.fetch(  # type: ignore[attr-defined]
            """
            SELECT fq.id, fq.product_id::text, fq.merchant_id::text, fq.listing_url,
                   fq.status, fq.error_msg, fq.enqueued_at, fq.claimed_at, fq.completed_at,
                   cp.clean_title, cp.brand, cp.model_family
            FROM forensic_queue fq
            LEFT JOIN canonical_products cp ON fq.product_id = cp.id
            ORDER BY fq.enqueued_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def get_catalog_products(
    pool: Optional[object],
    limit: int = 50,
    offset: int = 0,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List canonical products with lowest store price and defect count."""
    if pool is None:
        return []
    if query:
        q = f"%{query.strip()}%"
        rows = await pool.fetch(  # type: ignore[attr-defined]
            """
            SELECT cp.id::text, cp.clean_title, cp.brand, cp.model_family, cp.sub_model,
                   cp.storage_gb, cp.ram_gb, cp.pipeline_status::text, cp.spec_fingerprint,
                   MIN(l.current_price_lkr) AS lowest_price_lkr,
                   COUNT(l.id) AS listing_count,
                   dd.source_count AS defect_sources,
                   dd.confidence_label
            FROM canonical_products cp
            LEFT JOIN listings l ON cp.id = l.product_id
            LEFT JOIN defect_dossiers dd ON cp.id = dd.canonical_product_id
            WHERE cp.clean_title ILIKE $1 OR cp.brand ILIKE $1 OR cp.model_family ILIKE $1
            GROUP BY cp.id, dd.source_count, dd.confidence_label
            ORDER BY cp.first_seen_at DESC
            LIMIT $2 OFFSET $3
            """,
            q, limit, offset,
        )
    else:
        rows = await pool.fetch(  # type: ignore[attr-defined]
            """
            SELECT cp.id::text, cp.clean_title, cp.brand, cp.model_family, cp.sub_model,
                   cp.storage_gb, cp.ram_gb, cp.pipeline_status::text, cp.spec_fingerprint,
                   MIN(l.current_price_lkr) AS lowest_price_lkr,
                   COUNT(l.id) AS listing_count,
                   dd.source_count AS defect_sources,
                   dd.confidence_label
            FROM canonical_products cp
            LEFT JOIN listings l ON cp.id = l.product_id
            LEFT JOIN defect_dossiers dd ON cp.id = dd.canonical_product_id
            GROUP BY cp.id, dd.source_count, dd.confidence_label
            ORDER BY cp.first_seen_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
    return [dict(r) for r in rows]


async def get_product_detail_with_dossier(
    pool: Optional[object],
    product_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch complete product record, linked listings, defect dossier, arbitrage logs."""
    if pool is None:
        return None
    import json

    product_row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        SELECT id::text, clean_title, raw_title, brand, model_family, sub_model,
               storage_gb, ram_gb, region_code, spec_fingerprint,
               pipeline_status::text, attributes, first_seen_at, updated_at
        FROM canonical_products
        WHERE id = $1::uuid
        """,
        product_id,
    )
    if not product_row:
        return None

    listings_rows = await pool.fetch(  # type: ignore[attr-defined]
        """
        SELECT l.id::text, l.merchant_id::text, l.current_price_lkr,
               l.warranty_tier::text, l.listing_url, l.in_stock, l.updated_at,
               m.domain AS merchant_domain, m.display_name AS merchant_name,
               m.is_verified_agent, mf.dark_patterns_detected, mf.bnpl_surcharge_pct
        FROM listings l
        JOIN merchants m ON l.merchant_id = m.id
        LEFT JOIN merchant_forensics mf ON m.id = mf.merchant_id
        WHERE l.product_id = $1::uuid
        ORDER BY l.current_price_lkr ASC
        """,
        product_id,
    )

    dossier = await get_defect_dossier(pool, product_id)

    arbitrage_rows = await pool.fetch(  # type: ignore[attr-defined]
        """
        SELECT us_msrp_usd, eu_msrp_eur, fx_rate, true_landed_cost_lkr,
               merchant_price_lkr, markup_pct, price_label, classification, logged_at
        FROM currency_arbitrage_logs
        WHERE product_id = $1::uuid
        ORDER BY logged_at DESC
        LIMIT 5
        """,
        product_id,
    )

    teardown_rows = await pool.fetch(  # type: ignore[attr-defined]
        """
        SELECT component_name, observed_revision, repairability_score,
               source_url, source_type, notes, silent_revision_detected, recorded_at
        FROM component_teardowns
        WHERE product_id = $1::uuid
        ORDER BY recorded_at DESC
        """,
        product_id,
    )

    attrs = product_row["attributes"]
    if isinstance(attrs, str):
        attrs = json.loads(attrs)

    return {
        "product": dict(product_row),
        "attributes": attrs,
        "listings": [dict(r) for r in listings_rows],
        "defect_dossier": dossier,
        "arbitrage_logs": [dict(r) for r in arbitrage_rows],
        "teardowns": [dict(r) for r in teardown_rows],
    }


# ---------------------------------------------------------------------------
# Crawl queue helpers — autonomous domain-wide discovery + drip scrape
# ---------------------------------------------------------------------------

async def enqueue_crawl_urls(
    pool: Optional[object],
    *,
    domain: str,
    urls: List[str],
    depth: int = 0,
) -> int:
    """Bulk-insert discovered URLs into crawl_queue. Returns rows actually added.

    Idempotent: ``ON CONFLICT (url) DO NOTHING`` so re-discovering the same URL
    across runs never duplicates a row. No-op (returns 0) in dry-run mode.
    """
    if pool is None or not urls:
        return 0
    # De-dup within the batch while preserving order.
    seen: set = set()
    rows = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            rows.append((domain, u, depth))
    if not rows:
        return 0
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        before = await conn.fetchval("SELECT COUNT(*) FROM crawl_queue WHERE domain = $1", domain)
        await conn.executemany(
            """
            INSERT INTO crawl_queue (domain, url, depth)
            VALUES ($1, $2, $3)
            ON CONFLICT (url) DO NOTHING
            """,
            rows,
        )
        after = await conn.fetchval("SELECT COUNT(*) FROM crawl_queue WHERE domain = $1", domain)
    return int((after or 0) - (before or 0))


async def claim_next_crawl_url(
    pool: Optional[object],
    domain: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest pending crawl URL (optionally per-domain).

    Uses ``FOR UPDATE SKIP LOCKED`` so multiple drip workers never collide.
    Marks the row ``in_progress`` and returns it, or None when nothing pending.
    """
    if pool is None:
        return None
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        async with conn.transaction():
            if domain:
                row = await conn.fetchrow(
                    """
                    SELECT id, domain, url, depth, attempts
                    FROM crawl_queue
                    WHERE status = 'pending' AND domain = $1
                    ORDER BY discovered_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    domain,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT id, domain, url, depth, attempts
                    FROM crawl_queue
                    WHERE status = 'pending'
                    ORDER BY discovered_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                )
            if row is None:
                return None
            await conn.execute(
                """
                UPDATE crawl_queue
                SET status = 'in_progress', claimed_at = NOW(), attempts = attempts + 1
                WHERE id = $1
                """,
                row["id"],
            )
    return dict(row)


async def complete_crawl_url(pool: Optional[object], url: str) -> None:
    """Mark a crawl URL as successfully scraped."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        "UPDATE crawl_queue SET status = 'completed', completed_at = NOW() WHERE url = $1",
        url,
    )


async def fail_crawl_url(pool: Optional[object], url: str, error_msg: str) -> None:
    """Mark a crawl URL as failed with a (truncated) error message."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        UPDATE crawl_queue
        SET status = 'failed', error_msg = $2, completed_at = NOW()
        WHERE url = $1
        """,
        url, (error_msg or "")[:2000],
    )


async def get_crawl_queue_stats(
    pool: Optional[object],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Return crawl-queue counters for the dashboard monitor page.

    When ``domain`` is given the counts are scoped to that domain; otherwise
    aggregated across all domains. ``pages_scraped_today`` counts rows that
    completed since local midnight (UTC).
    """
    empty = {
        "domain": domain,
        "total": 0, "pending": 0, "in_progress": 0,
        "completed": 0, "failed": 0, "pages_scraped_today": 0,
        "last_error": None,
    }
    if pool is None:
        return empty
    where = "WHERE domain = $1" if domain else ""
    args = [domain] if domain else []
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE status = 'pending')      AS pending,
            COUNT(*) FILTER (WHERE status = 'in_progress')  AS in_progress,
            COUNT(*) FILTER (WHERE status = 'completed')    AS completed,
            COUNT(*) FILTER (WHERE status = 'failed')       AS failed,
            COUNT(*) FILTER (WHERE status = 'completed'
                             AND completed_at >= date_trunc('day', NOW())) AS pages_scraped_today
        FROM crawl_queue
        {where}
        """,
        *args,
    )
    err_row = await pool.fetchrow(  # type: ignore[attr-defined]
        f"""
        SELECT error_msg FROM crawl_queue
        {where + ' AND ' if where else 'WHERE '}error_msg IS NOT NULL
        ORDER BY completed_at DESC NULLS LAST
        LIMIT 1
        """,
        *args,
    )
    out = dict(row) if row else dict(empty)
    out["domain"] = domain
    out["last_error"] = err_row["error_msg"] if err_row else None
    return out


async def list_crawl_domains(pool: Optional[object]) -> List[Dict[str, Any]]:
    """Per-domain crawl progress rollup for the dashboard monitor."""
    if pool is None:
        return []
    rows = await pool.fetch(  # type: ignore[attr-defined]
        """
        SELECT domain,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status = 'pending')     AS pending,
               COUNT(*) FILTER (WHERE status = 'completed')   AS completed,
               COUNT(*) FILTER (WHERE status = 'failed')      AS failed,
               COUNT(*) FILTER (WHERE status = 'completed'
                                AND completed_at >= date_trunc('day', NOW())) AS scraped_today,
               MAX(discovered_at) AS last_discovered
        FROM crawl_queue
        GROUP BY domain
        ORDER BY last_discovered DESC
        """
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Domain profile helpers — cached zero-selector extraction hints
# ---------------------------------------------------------------------------

async def get_domain_profile(
    pool: Optional[object],
    domain: str,
) -> Optional[Dict[str, Any]]:
    """Return the cached selector profile for a domain, or None."""
    if pool is None:
        return None
    row = await pool.fetchrow(  # type: ignore[attr-defined]
        """
        SELECT domain, title_selector, price_selector, stock_selector,
               specs_table_selector, extraction_tier, success_count, failure_count
        FROM domain_profiles
        WHERE domain = $1
        """,
        domain,
    )
    return dict(row) if row else None


async def upsert_domain_profile(
    pool: Optional[object],
    *,
    domain: str,
    title_selector: Optional[str] = None,
    price_selector: Optional[str] = None,
    stock_selector: Optional[str] = None,
    specs_table_selector: Optional[str] = None,
    extraction_tier: Optional[str] = None,
) -> None:
    """Insert or update the cached CSS-selector profile for a domain (Tier 2)."""
    if pool is None:
        return
    await pool.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO domain_profiles
            (domain, title_selector, price_selector, stock_selector,
             specs_table_selector, extraction_tier, updated_at)
        VALUES ($1,$2,$3,$4,$5,$6, NOW())
        ON CONFLICT (domain) DO UPDATE
            SET title_selector       = COALESCE(EXCLUDED.title_selector, domain_profiles.title_selector),
                price_selector       = COALESCE(EXCLUDED.price_selector, domain_profiles.price_selector),
                stock_selector       = COALESCE(EXCLUDED.stock_selector, domain_profiles.stock_selector),
                specs_table_selector = COALESCE(EXCLUDED.specs_table_selector, domain_profiles.specs_table_selector),
                extraction_tier      = COALESCE(EXCLUDED.extraction_tier, domain_profiles.extraction_tier),
                updated_at           = NOW()
        """,
        domain, title_selector, price_selector, stock_selector,
        specs_table_selector, extraction_tier,
    )


async def record_domain_profile_result(
    pool: Optional[object],
    domain: str,
    success: bool,
) -> None:
    """Increment the success/failure counter for a domain's cached profile."""
    if pool is None:
        return
    col = "success_count" if success else "failure_count"
    await pool.execute(  # type: ignore[attr-defined]
        f"""
        INSERT INTO domain_profiles (domain, {col})
        VALUES ($1, 1)
        ON CONFLICT (domain) DO UPDATE
            SET {col} = domain_profiles.{col} + 1,
                updated_at = NOW()
        """,
        domain,
    )


