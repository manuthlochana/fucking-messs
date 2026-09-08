"""asyncpg connection pool factory for KALA-BALANA.

Optimized for a 2-core / 4 GB VPS: max_size=5 prevents connection
starvation under low-concurrency workloads while staying inside
PostgreSQL's default max_connections=100.

Setting ``dsn`` to an empty string or None puts the entire DB layer
into dry-run mode — all ``db.queries`` helpers return None/[] without
making any network calls. This lets the crawler run without a
PostgreSQL instance during development.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from logging_utils import log

try:
    import asyncpg  # type: ignore
    _ASYNCPG_OK = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_OK = False


async def create_pool(
    dsn: Optional[str],
    *,
    max_size: int = 5,
    min_size: int = 1,
) -> Optional["asyncpg.Pool"]:
    """Create and return an asyncpg connection pool.

    Returns ``None`` when:
    - ``dsn`` is empty/None (dry-run mode), or
    - ``asyncpg`` is not installed.

    The caller can pass the returned value directly to all query helpers;
    they accept ``None`` and short-circuit gracefully.
    """
    if not dsn:
        log.debug("DB_DSN not set — running in dry-run mode (no database).")
        return None
    if not _ASYNCPG_OK:
        log.warn(
            "asyncpg not installed — DB disabled. "
            "Run: pip install asyncpg pgvector"
        )
        return None
    try:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            # Codecs for pgvector are registered per-connection via init.
            init=_register_codecs,
            command_timeout=30,
            server_settings={"application_name": "kala_balana"},
        )
        log.success(f"DB pool ready (max_size={max_size}, dsn=...{dsn[-20:]}).")
        return pool
    except Exception as exc:
        log.error(f"Failed to create DB pool: {exc}")
        return None


async def close_pool(pool: Optional["asyncpg.Pool"]) -> None:
    """Gracefully close an asyncpg pool (no-op if None)."""
    if pool is not None:
        await pool.close()
        log.debug("DB pool closed.")


async def _register_codecs(conn: "asyncpg.Connection") -> None:
    """Register pgvector codecs on each new connection.

    Uses pgvector.asyncpg if available, otherwise registers a custom text codec.
    """
    try:
        from pgvector.asyncpg import register_vector
        await register_vector(conn)
        return
    except Exception:
        pass

    try:
        # asyncpg needs to know the OID of the 'vector' type.
        row = await conn.fetchrow(
            "SELECT oid FROM pg_type WHERE typname = 'vector'"
        )
        if row:
            oid = row["oid"]
            await conn.set_type_codec(
                "vector",
                schema="public",
                encoder=_encode_vector,
                decoder=_decode_vector,
                format="text",
                oid=oid,
            )
    except Exception:
        pass  # Vector type not available; embedding inserts will be skipped.


def _encode_vector(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def _decode_vector(s: str) -> list[float]:
    return [float(x) for x in s.strip("[]").split(",")]
