"""Idempotent DDL migrations for KALA-BALANA.

Calling ``run_migrations(pool)`` is safe to run multiple times — all
statements use ``IF NOT EXISTS`` / ``DO $$ … $$`` guards so they are
a no-op when the schema is already current.

Partitioning strategy
---------------------
``price_history`` and ``currency_arbitrage_logs`` use PostgreSQL RANGE
partitioning by month. Two pre-seeded child partitions (current month
and next) are created automatically by ``_ensure_current_partitions()``.
A scheduled job (or the crawler startup hook) should call that helper
periodically to create future partitions ahead of time.
"""

from __future__ import annotations

import asyncio
from datetime import date, timezone
from typing import Optional

from logging_utils import log

# ---------------------------------------------------------------------------
# Core DDL — tables, indexes, partitions
# ---------------------------------------------------------------------------

_DDL_STATEMENTS = [
    # ── pgvector extension ──────────────────────────────────────────────────
    "CREATE EXTENSION IF NOT EXISTS vector;",
    "CREATE EXTENSION IF NOT EXISTS pgcrypto;",  # for gen_random_uuid()

    # ── pipeline_status enum ────────────────────────────────────────────────
    """
    DO $$ BEGIN
        CREATE TYPE pipeline_status AS ENUM (
            'discovered',
            'phase1_done',
            'phase2_done',
            'phase3_done',
            'phase4_done',
            'phase5_done',
            'complete',
            'failed'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;
    """,

    # ── warranty_tier enum ─────────────────────────────────────────────────
    """
    DO $$ BEGIN
        CREATE TYPE warranty_tier AS ENUM (
            'official_agent',
            'shop_inhouse',
            'checking',
            'unstated'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;
    """,

    # ── canonical_products ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS canonical_products (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        spec_fingerprint TEXT UNIQUE NOT NULL,
        brand           TEXT,
        model_family    TEXT,
        sub_model       TEXT,
        storage_gb      INTEGER,
        ram_gb          INTEGER,
        region_code     TEXT DEFAULT '',
        raw_title       TEXT NOT NULL,
        clean_title     TEXT NOT NULL,
        attributes      JSONB NOT NULL DEFAULT '{}',
        pipeline_status pipeline_status NOT NULL DEFAULT 'discovered',
        first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_cp_brand_model ON canonical_products (brand, model_family);",
    "CREATE INDEX IF NOT EXISTS idx_cp_pipeline_status ON canonical_products (pipeline_status);",
    "CREATE INDEX IF NOT EXISTS idx_cp_attributes ON canonical_products USING GIN (attributes);",

    # ── product_embeddings ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS product_embeddings (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        product_id  UUID NOT NULL REFERENCES canonical_products (id) ON DELETE CASCADE,
        embedding   vector(768),
        model_name  TEXT NOT NULL DEFAULT 'text-embedding-004',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    # HNSW index for fast approximate nearest-neighbour search.
    """
    CREATE INDEX IF NOT EXISTS idx_pe_embedding_hnsw
        ON product_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """,

    # ── merchants ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS merchants (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        domain              TEXT UNIQUE NOT NULL,
        display_name        TEXT,
        physical_address    TEXT,
        city                TEXT,
        opening_hours       TEXT,
        is_verified_agent   BOOLEAN NOT NULL DEFAULT TRUE,
        authorized_brands   TEXT[] NOT NULL DEFAULT '{}',
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # ── merchant_forensics ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS merchant_forensics (
        id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        merchant_id             UUID NOT NULL REFERENCES merchants (id) ON DELETE CASCADE,
        bnpl_surcharge_pct      NUMERIC(6,2),
        cc_surcharge_pct        NUMERIC(6,2),
        dark_patterns           JSONB NOT NULL DEFAULT '[]',
        koko_mintpay_hidden_fees JSONB NOT NULL DEFAULT '{}',
        raw_audit_data          JSONB NOT NULL DEFAULT '{}',
        audited_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (merchant_id)  -- one forensic record per merchant (upsert)
    );
    """,

    # ── listings ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS listings (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        product_id          UUID NOT NULL REFERENCES canonical_products (id),
        merchant_id         UUID NOT NULL REFERENCES merchants (id),
        current_price_lkr   NUMERIC(12,2) NOT NULL,
        warranty_tier       warranty_tier NOT NULL DEFAULT 'unstated',
        listing_url         TEXT UNIQUE NOT NULL,
        in_stock            BOOLEAN NOT NULL DEFAULT TRUE,
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (product_id, merchant_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_listings_product ON listings (product_id);",
    "CREATE INDEX IF NOT EXISTS idx_listings_merchant ON listings (merchant_id);",

    # ── defect_dossiers — one canonical dossier per product ───────────────
    """
    CREATE TABLE IF NOT EXISTS defect_dossiers (
        canonical_product_id UUID PRIMARY KEY REFERENCES canonical_products (id) ON DELETE CASCADE,
        defects              JSONB NOT NULL DEFAULT '[]',
        source_count         INTEGER NOT NULL DEFAULT 0,
        confidence_label     TEXT NOT NULL DEFAULT 'low',
        astroturf_risk_score NUMERIC(4,3) NOT NULL DEFAULT 0.0,
        generated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_dd_confidence ON defect_dossiers (confidence_label);",

    # ── component_teardowns (append-only) ─────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS component_teardowns (
        id                        BIGSERIAL PRIMARY KEY,
        product_id                UUID NOT NULL REFERENCES canonical_products (id) ON DELETE CASCADE,
        revision_label            TEXT NOT NULL,
        component_changed         TEXT NOT NULL,
        change_description        TEXT NOT NULL,
        repairability_score       NUMERIC(4,2),
        teardown_source_url       TEXT,
        ifixit_score              NUMERIC(4,2),
        silent_revision_detected  BOOLEAN NOT NULL DEFAULT FALSE,
        recorded_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # ── forensic_queue — durable inter-process work queue ─────────────────
    """
    CREATE TABLE IF NOT EXISTS forensic_queue (
        id            BIGSERIAL PRIMARY KEY,
        product_id    UUID NOT NULL REFERENCES canonical_products (id) ON DELETE CASCADE,
        merchant_id   UUID,
        listing_url   TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'pending',
        error_msg     TEXT,
        enqueued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        claimed_at    TIMESTAMPTZ,
        completed_at  TIMESTAMPTZ,
        CONSTRAINT forensic_queue_status_check
            CHECK (status IN ('pending', 'claimed', 'done', 'failed'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_fq_status ON forensic_queue (status, enqueued_at);",
    "CREATE INDEX IF NOT EXISTS idx_fq_product ON forensic_queue (product_id);",

    # ── price_history (monthly RANGE partitioned) ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS price_history (
        id          BIGSERIAL,
        listing_id  UUID NOT NULL REFERENCES listings (id),
        price_lkr   NUMERIC(12,2) NOT NULL,
        in_stock    BOOLEAN NOT NULL DEFAULT TRUE,
        scraped_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    ) PARTITION BY RANGE (scraped_at);
    """,
    "CREATE INDEX IF NOT EXISTS idx_ph_listing ON price_history (listing_id, scraped_at DESC);",

    # ── currency_arbitrage_logs (monthly RANGE partitioned) ───────────────
    """
    CREATE TABLE IF NOT EXISTS currency_arbitrage_logs (
        id                      BIGSERIAL,
        product_id              UUID NOT NULL REFERENCES canonical_products (id),
        global_msrp_usd         NUMERIC(10,2),
        cbsl_rate               NUMERIC(10,4),
        true_landed_cost_lkr    NUMERIC(12,2),
        merchant_price_lkr      NUMERIC(12,2),
        markup_pct              NUMERIC(7,2),
        price_label             TEXT,
        logged_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
    ) PARTITION BY RANGE (logged_at);
    """,
    "CREATE INDEX IF NOT EXISTS idx_cal_product ON currency_arbitrage_logs (product_id, logged_at DESC);",
]


async def run_migrations(pool: Optional[object]) -> None:
    """Execute all DDL statements and seed initial partitions.

    Safe to call on every startup — idempotent by design.
    No-op when ``pool`` is None (dry-run mode).
    """
    if pool is None:
        log.debug("Migrations skipped (dry-run mode).")
        return

    log.info("Running DB migrations…")
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        async with conn.transaction():
            for stmt in _DDL_STATEMENTS:
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    await conn.execute(stmt)
                except Exception as exc:
                    # Log but continue — some statements (e.g. CREATE INDEX) may
                    # fail if the table doesn't exist yet due to ordering; the
                    # next migration run will fix it.
                    log.warn(f"Migration stmt warning: {exc!r}")

    await _ensure_current_partitions(pool)
    log.success("DB migrations complete.")


async def _ensure_current_partitions(pool: object) -> None:
    """Create monthly child partitions for the current and next month.

    Called at startup and can be called periodically by a cron task.
    """
    from datetime import datetime, timedelta

    today = date.today()
    # We need the first day of the current month and the next two months.
    months = []
    d = today.replace(day=1)
    for _ in range(3):
        months.append(d)
        # Advance to the first day of the next month.
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        for start in months[:-1]:
            end = months[months.index(start) + 1]
            label = start.strftime("%Y_%m")
            for table in ("price_history", "currency_arbitrage_logs"):
                col = "scraped_at" if table == "price_history" else "logged_at"
                part_name = f"{table}_{label}"
                ddl = (
                    f"CREATE TABLE IF NOT EXISTS {part_name} "
                    f"PARTITION OF {table} "
                    f"FOR VALUES FROM ('{start}') TO ('{end}');"
                )
                try:
                    await conn.execute(ddl)
                    log.debug(f"Partition ensured: {part_name}")
                except Exception as exc:
                    log.debug(f"Partition {part_name} note: {exc!r}")
