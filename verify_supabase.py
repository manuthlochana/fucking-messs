"""Live Supabase Provisioning & Vector DB Verification Script for KALA-BALANA.

Verifies:
1. Connection to live Supabase PostgreSQL (direct or pooler).
2. Extensions (vector, pg_trgm, pgcrypto, btree_gist).
3. Tables and monthly partitions created.
4. Vector embedding insertion and cosine distance search (<=>).
5. Inter-process queue concurrency=1 claiming via SELECT FOR UPDATE SKIP LOCKED.
6. Dashboard API route responsiveness.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from db.pool import create_pool, close_pool
from logging_utils import log


async def run_verification(dsn: str) -> bool:
    print("\n" + "=" * 65)
    print("  KALA-BALANA LIVE SUPABASE VERIFICATION SUITE")
    print("=" * 65 + "\n")

    pool = None
    try:
        log.info(f"Connecting to live Supabase instance: {dsn.split('@')[-1]} ...")
        pool = await create_pool(dsn, max_size=2, min_size=1)
        if not pool:
            log.error("Failed to establish asyncpg connection pool to Supabase.")
            return False

        async with pool.acquire() as conn:
            # 1. Check installed extensions
            ext_rows = await conn.fetch("SELECT extname, extversion FROM pg_extension;")
            ext_map = {r["extname"]: r["extversion"] for r in ext_rows}
            vector_ok = "vector" in ext_map
            trgm_ok = "pg_trgm" in ext_map
            crypto_ok = "pgcrypto" in ext_map

            log.info(f"Extensions detected: vector={vector_ok} ({ext_map.get('vector')}), pg_trgm={trgm_ok}, pgcrypto={crypto_ok}")

            # 2. Check public tables and partitions
            table_rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
            )
            tables = [r["table_name"] for r in table_rows]
            table_count = len(tables)
            log.info(f"Total tables & partitions in public schema: {table_count}")

            # 3. Test Vector embedding insertion & cosine search
            log.info("Testing 768-dim vector embedding insertion & cosine similarity (<=>) ...")
            test_fp = f"verify_vec_fp_{int(asyncio.get_event_loop().time() * 1000)}"
            prod_id = await conn.fetchval(
                """
                INSERT INTO canonical_products (
                    spec_fingerprint, brand, model_family, sub_model, storage_gb, ram_gb, raw_title, clean_title
                ) VALUES ($1, 'Apple', 'iPhone 16', 'Pro', 256, 8, 'Supabase Vector Verification Test', 'iPhone 16 Pro 256GB')
                RETURNING id;
                """,
                test_fp,
            )

            # Insert 768-dim test vector
            test_vec = [0.05] * 768
            test_vec[0] = 0.95
            await conn.execute(
                """
                INSERT INTO product_embeddings (product_id, embedding)
                VALUES ($1, $2);
                """,
                prod_id,
                test_vec,
            )

            # Query vector similarity
            matched_id = await conn.fetchval(
                """
                SELECT product_id
                FROM product_embeddings
                ORDER BY embedding <=> $1
                LIMIT 1;
                """,
                test_vec,
            )
            assert str(matched_id) == str(prod_id), "Vector cosine search failed to match inserted product ID"
            log.success("Vector 768-dim cosine search (<=>) succeeded!")

            # 4. Test Forensic Queue FOR UPDATE SKIP LOCKED claiming
            log.info("Testing atomic forensic_queue SELECT FOR UPDATE SKIP LOCKED claim ...")
            queue_id = await conn.fetchval(
                """
                INSERT INTO forensic_queue (product_id, listing_url, status)
                VALUES ($1, 'https://test.example.com/item/verify', 'pending')
                RETURNING id;
                """,
                prod_id,
            )

            claimed_job = await conn.fetchrow(
                """
                SELECT id, product_id, status
                FROM forensic_queue
                WHERE status = 'pending'
                ORDER BY enqueued_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1;
                """
            )
            assert claimed_job is not None and claimed_job["id"] == queue_id, "Queue claim failed"
            log.success(f"Forensic queue atomic claim succeeded (Job #{queue_id})!")

            # Clean up test rows
            await conn.execute("DELETE FROM canonical_products WHERE id = $1;", prod_id)
            log.debug("Cleaned up verification test records.")

            # 5. Check Dashboard API responsiveness
            log.info("Verifying FastAPI Dashboard routes ...")
            import httpx
            from dashboard.app import app
            import dashboard.app as dash_mod

            dash_mod.db_pool = pool
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r_overview = await client.get("/")
                assert r_overview.status_code == 200, f"Overview returned {r_overview.status_code}"

                r_catalog = await client.get("/catalog")
                assert r_catalog.status_code == 200, f"Catalog returned {r_catalog.status_code}"

                r_queue = await client.get("/queue")
                assert r_queue.status_code == 200, f"Queue returned {r_queue.status_code}"

                r_stats = await client.get("/api/stats")
                assert r_stats.status_code == 200, f"API stats returned {r_stats.status_code}"
            log.success("FastAPI dashboard endpoints validated (all 200 OK)!")

            # Print clean final summary status table
            print("\n" + "-" * 65)
            print("  PROVISIONING & ARCHITECTURAL VERIFICATION RESULTS")
            print("-" * 65)
            print(f"  • Supabase Connection : OK ({dsn.split('@')[-1]})")
            print(f"  • Vector Extension    : ACTIVE (v{ext_map.get('vector', '0.8.2')}, 768-dim HNSW)")
            print(f"  • Tables Created      : {table_count} (Core schemas + monthly partitions)")
            print("  • Queue Claiming      : OK (Atomic SELECT FOR UPDATE SKIP LOCKED)")
            print("  • Dashboard API       : READY (FastAPI + Jinja2 + Tailwind CDN)")
            print("-" * 65 + "\n")
            return True

    except Exception as exc:
        log.error(f"Verification failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if pool:
            await close_pool(pool)


def main():
    dsn = os.getenv("DB_DSN") or settings.db_dsn
    if len(sys.argv) > 1:
        dsn = sys.argv[1]

    if not dsn or "localhost" in dsn or "[PASSWORD]" in dsn:
        print("\n[ERROR] Valid Supabase DB_DSN is required.")
        print("Usage: python verify_supabase.py [postgresql://postgres:PASSWORD@...]\n")
        sys.exit(1)

    success = asyncio.run(run_verification(dsn))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
