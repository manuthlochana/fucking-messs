-- ============================================================================
-- KALA-BALANA — consolidated PostgreSQL + pgvector schema
-- ============================================================================
-- Generated from db/migrations.py::_DDL_STATEMENTS as the canonical, human-
-- readable DDL reference. Applying this file is equivalent to running
-- run_migrations(pool) once. Everything is idempotent (IF NOT EXISTS / guarded
-- DO blocks), so re-applying is a no-op on an already-current database.
--
--   psql "$DB_DSN" -f schema.sql
--
-- Partitioning: price_history and currency_arbitrage_logs are RANGE-partitioned
-- by month. Current + next month partitions are seeded at the bottom of this
-- file; the ensure_monthly_partitions() function (also defined below) creates
-- future partitions and is intended to be run monthly by cron / pg_cron.
-- ============================================================================

-- ── extensions ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ── pipeline_status enum ─────────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE pipeline_status AS ENUM (
        'discovered',
        'phase1_done',
        'phase2_done',
        'phase3_done',
        'phase4_done',
        'phase5_done',
        'complete',
        'failed',
        'partial'
    );
EXCEPTION WHEN duplicate_object THEN
    ALTER TYPE pipeline_status ADD VALUE IF NOT EXISTS 'partial';
END $$;

-- ── warranty_tier enum ───────────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE warranty_tier AS ENUM (
        'official_agent',
        'shop_inhouse',
        'checking',
        'unstated'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ── canonical_products ───────────────────────────────────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_cp_brand_model ON canonical_products (brand, model_family);
CREATE INDEX IF NOT EXISTS idx_cp_pipeline_status ON canonical_products (pipeline_status);
CREATE INDEX IF NOT EXISTS idx_cp_pipeline_status_active ON canonical_products (pipeline_status) WHERE pipeline_status NOT IN ('complete');
CREATE INDEX IF NOT EXISTS idx_cp_attributes ON canonical_products USING GIN (attributes);

-- ── product_embeddings (HNSW ANN index) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS product_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID UNIQUE NOT NULL REFERENCES canonical_products (id) ON DELETE CASCADE,
    embedding   vector(768),
    model_name  TEXT NOT NULL DEFAULT 'text-embedding-004',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_product_embedding_hnsw
    ON product_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── merchants ─────────────────────────────────────────────────────────────────
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

-- ── merchant_forensics ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS merchant_forensics (
    id                             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id                    UUID NOT NULL REFERENCES merchants (id) ON DELETE CASCADE,
    physical_presence_verified     BOOLEAN,
    physical_addresses             JSONB DEFAULT '[]',
    operating_hours                JSONB DEFAULT '{}',
    business_registry_name         TEXT,
    business_registry_age_days     INTEGER,
    domain_registration_age_days   INTEGER,
    domain_lineage                 JSONB DEFAULT '[]',
    warranty_claims_verified       BOOLEAN,
    surcharge_map                  JSONB DEFAULT '{}',
    dark_patterns_detected         JSONB DEFAULT '[]',
    price_devaluation_lag_days_up  NUMERIC(5,2),
    price_devaluation_lag_days_down NUMERIC(5,2),
    reliability_score_override     NUMERIC(3,2),
    bnpl_surcharge_pct             NUMERIC(6,2),
    cc_surcharge_pct               NUMERIC(6,2),
    dark_patterns                  JSONB NOT NULL DEFAULT '[]',
    koko_mintpay_hidden_fees       JSONB NOT NULL DEFAULT '{}',
    raw_audit_data                 JSONB NOT NULL DEFAULT '{}',
    audited_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (merchant_id)
);
CREATE INDEX IF NOT EXISTS idx_mf_dark_patterns ON merchant_forensics USING GIN (dark_patterns_detected jsonb_path_ops);

-- ── listings ──────────────────────────────────────────────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_listings_product ON listings (product_id);
CREATE INDEX IF NOT EXISTS idx_listings_merchant ON listings (merchant_id);

-- ── defect_dossiers — one canonical dossier per product ───────────────────────
CREATE TABLE IF NOT EXISTS defect_dossiers (
    canonical_product_id    UUID PRIMARY KEY REFERENCES canonical_products (id) ON DELETE CASCADE,
    defects                 JSONB NOT NULL DEFAULT '[]',
    source_count            INTEGER NOT NULL DEFAULT 0,
    astroturf_risk_score    NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    sponsored_content_ratio NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    confidence_label        TEXT NOT NULL DEFAULT 'low',
    generated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dd_confidence ON defect_dossiers (confidence_label);
CREATE INDEX IF NOT EXISTS idx_dd_gin ON defect_dossiers USING GIN (defects jsonb_path_ops);

-- ── component_teardowns (append-only) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS component_teardowns (
    id                        BIGSERIAL PRIMARY KEY,
    product_id                UUID NOT NULL REFERENCES canonical_products (id) ON DELETE CASCADE,
    component_name            TEXT,
    observed_revision         TEXT,
    serial_range_start        TEXT,
    serial_range_end          TEXT,
    manufacture_date_range    DATERANGE,
    repairability_score       NUMERIC(4,2),
    source_url                TEXT,
    source_type               TEXT,
    notes                     TEXT,
    revision_label            TEXT,
    component_changed         TEXT,
    change_description        TEXT,
    teardown_source_url       TEXT,
    ifixit_score              NUMERIC(4,2),
    silent_revision_detected  BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_teardowns_product ON component_teardowns (product_id);
CREATE INDEX IF NOT EXISTS idx_ct_date_range ON component_teardowns USING gist (manufacture_date_range);

-- ── forensic_queue — durable inter-process work queue ─────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_fq_status ON forensic_queue (status, enqueued_at);
CREATE INDEX IF NOT EXISTS idx_fq_product ON forensic_queue (product_id);

-- ── price_history (monthly RANGE partitioned) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS price_history (
    id          BIGSERIAL,
    listing_id  UUID NOT NULL REFERENCES listings (id),
    price_lkr   NUMERIC(12,2) NOT NULL,
    in_stock    BOOLEAN NOT NULL DEFAULT TRUE,
    scraped_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (scraped_at);
CREATE INDEX IF NOT EXISTS idx_ph_listing ON price_history (listing_id, scraped_at DESC);

-- ── currency_arbitrage_logs (monthly RANGE partitioned) ───────────────────────
CREATE TABLE IF NOT EXISTS currency_arbitrage_logs (
    id                      BIGSERIAL,
    product_id              UUID NOT NULL REFERENCES canonical_products (id),
    global_msrp_usd         NUMERIC(10,2),
    us_msrp_usd             NUMERIC(10,2),
    eu_msrp_eur             NUMERIC(10,2),
    uae_msrp_aed            NUMERIC(10,2),
    india_msrp_inr          NUMERIC(12,2),
    cbsl_rate               NUMERIC(10,4),
    fx_rate                 NUMERIC(10,4),
    tariff_pct_applied      NUMERIC(5,2),
    true_landed_cost_lkr    NUMERIC(12,2),
    landed_cost_lkr         NUMERIC(12,2),
    merchant_price_lkr      NUMERIC(12,2),
    local_price_lkr         NUMERIC(12,2),
    markup_pct              NUMERIC(7,2),
    margin_pct              NUMERIC(6,2),
    price_label             TEXT,
    classification          TEXT,
    logged_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (logged_at);
CREATE INDEX IF NOT EXISTS idx_cal_product ON currency_arbitrage_logs (product_id, logged_at DESC);

-- ── llm_key_usage_log ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_key_usage_log (
    id              BIGSERIAL PRIMARY KEY,
    provider        TEXT NOT NULL,
    key_identifier  TEXT NOT NULL,
    requests_count  INTEGER NOT NULL DEFAULT 0,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    rate_limited    BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_window ON llm_key_usage_log (provider, key_identifier, window_start DESC);

-- ============================================================================
-- Monthly partition maintenance
-- ============================================================================
-- ensure_monthly_partitions(months_ahead) creates the current month plus the
-- next N months of child partitions for both RANGE-partitioned tables. It is
-- idempotent (CREATE TABLE IF NOT EXISTS) and mirrors
-- db/migrations.py::_ensure_current_partitions. Schedule it monthly, e.g. with
-- pg_cron:
--
--   SELECT cron.schedule('kb-partitions', '0 0 1 * *',
--                        $$SELECT ensure_monthly_partitions(2)$$);
--
CREATE OR REPLACE FUNCTION ensure_monthly_partitions(months_ahead INTEGER DEFAULT 2)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    tbl       TEXT;
    m         INTEGER;
    part_from DATE;
    part_to   DATE;
    part_name TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY['price_history', 'currency_arbitrage_logs'] LOOP
        FOR m IN 0..months_ahead LOOP
            part_from := date_trunc('month', CURRENT_DATE)::date + (m || ' month')::interval;
            part_to   := part_from + INTERVAL '1 month';
            part_name := format('%s_%s', tbl, to_char(part_from, 'YYYY_MM'));
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L);',
                part_name, tbl, part_from, part_to
            );
        END LOOP;
    END LOOP;
END;
$$;

-- Seed the current + next 2 months of partitions immediately.
SELECT ensure_monthly_partitions(2);

-- ============================================================================
-- Optional autovacuum tuning for the hot append-only partitioned tables.
-- Price history and arbitrage logs are write-heavy and append-only; tightening
-- the autovacuum scale factor keeps statistics fresh for the planner. Applied
-- at the parent level so future partitions inherit sensible defaults.
-- ============================================================================
ALTER TABLE price_history           SET (autovacuum_vacuum_scale_factor = 0.02,
                                         autovacuum_analyze_scale_factor = 0.01);
ALTER TABLE currency_arbitrage_logs SET (autovacuum_vacuum_scale_factor = 0.02,
                                         autovacuum_analyze_scale_factor = 0.01);

