# LOCKED BY Worker B

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from app.core.config import settings

logger = logging.getLogger(__name__)

_engine = None
_session_maker = None


def _get_database_url() -> str:
    url = settings.database_url or ""
    if not url:
        raise ValueError(
            "DATABASE_URL is not configured. Please set DATABASE_URL in your environment "
            "variables or .env file before starting the server."
        )
    return url


def _pydantic_json_serializer(*args: object) -> str:
    """JSON serializer that handles Pydantic BaseModel instances.

    SQLAlchemy JSON columns (``job.state``, ``report.citations``, etc.)
    receive Pydantic model objects. The default ``json.dumps`` cannot
    serialize them; this serializer converts any ``BaseModel`` via
    ``model_dump(mode="json")`` first.
    """
    if len(args) == 1:
        obj = args[0]
        if isinstance(obj, BaseModel):
            return json.dumps(obj.model_dump(mode="json"))
        return json.dumps(obj)
    return json.dumps(*args)


def _get_async_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        if url:
            _engine = create_async_engine(
                url,
                echo=settings.debug,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                future=True,
                json_serializer=_pydantic_json_serializer,
            )
    return _engine


def _get_session_maker():
    global _session_maker
    if _session_maker is None:
        engine = _get_async_engine()
        if engine:
            _session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
    return _session_maker


async def get_async_session_factory():
    try:
        maker = _get_session_maker()
        if maker is None:
            raise ValueError(
                "Database not configured. Set DATABASE_URL in .env before starting the server"
            )
        return maker
    except ValueError as e:
        logger.error("Failed to get session factory: %s", e)
        raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    try:
        maker = _get_session_maker()
        if maker is None:
            raise ValueError(
                "Database not configured. Set DATABASE_URL in .env before starting the server"
            )
        async with maker() as session:
            yield session
    except ValueError as e:
        logger.error("Failed to get database session: %s", e)
        raise


def get_async_engine():
    return _get_async_engine()


# Export for tests
__all__ = [
    "get_async_engine",
    "get_session",
    "init_db",
    "create_tables",
    "_get_async_engine",
    "_pydantic_json_serializer",
    "_get_database_url",
    "_get_session_maker",
]


async def init_db():
    try:
        engine = _get_async_engine()
        if engine is None:
            raise ValueError(
                "Database not configured. Set DATABASE_URL in .env before starting the server"
            )
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
    except ValueError as e:
        logger.error("Database initialization failed: %s", e)
        raise


create_tables = init_db
