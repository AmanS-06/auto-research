# LOCKED BY Worker B

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import AsyncConnectionPool

    HAS_CHECKPOINTER = True
except ImportError:
    PostgresSaver = None  # type: ignore
    AsyncConnectionPool = None  # type: ignore
    HAS_CHECKPOINTER = False
    logger.warning(
        "PostgreSQL checkpointer not available. Install psycopg[binary] to enable checkpointing."
    )

from app.core.config import settings

_pool: AsyncConnectionPool | None = None
_checkpointer: PostgresSaver | None = None

_POSTGRES_ASYNC_RE = re.compile(r"^postgresql\+asyncpg://")
_POSTGRES_SYNC_RE = re.compile(r"^postgresql://")


def _get_sync_dsn() -> str | None:
    if not settings.database_url:
        logger.warning("DATABASE_URL is empty; checkpointer unavailable")
        return None

    url = settings.database_url

    if _POSTGRES_ASYNC_RE.match(url):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)

    if _POSTGRES_SYNC_RE.match(url):
        return url

    logger.error(
        "Cannot determine sync DSN from DATABASE_URL=%r. "
        "Expected postgresql+asyncpg:// or postgresql:// prefix. "
        "Checkpointer unavailable.",
        url,
    )
    return None


async def get_checkpointer() -> PostgresSaver | None:
    if not HAS_CHECKPOINTER:
        logger.error("Cannot create checkpointer: psycopg not available")
        return None

    global _pool, _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    dsn = _get_sync_dsn()
    if dsn is None:
        return None

    _pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=5,
    )
    await _pool.open()

    _checkpointer = PostgresSaver(_pool)
    await _checkpointer.setup()
    return _checkpointer


async def close_checkpointer():
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
    _pool = None
    _checkpointer = None
