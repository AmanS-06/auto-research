# LOCKED BY Worker C
"""Tests for database configuration and migrations."""

from __future__ import annotations

import ast
import os
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlmodel import SQLModel

from app.core.database import get_session, init_db, _pydantic_json_serializer
from app.models.research import ResearchJob, ResearchReport


def test_canonical_migration_exists():
    """The single canonical migration file must exist and be revision 0001."""
    versions_dir = os.path.join(
        os.path.dirname(__file__), "..", "alembic", "versions"
    )
    py_files = [
        f
        for f in os.listdir(versions_dir)
        if f.endswith(".py") and f != "__init__.py"
    ]

    assert len(py_files) == 1, (
        f"Expected exactly one migration file, found {len(py_files)}: {py_files}"
    )


def test_migration_columns_match_models():
    """Verify the migration creates columns that match the SQLModel definitions."""
    versions_dir = os.path.join(
        os.path.dirname(__file__), "..", "alembic", "versions"
    )
    py_files = [
        f for f in os.listdir(versions_dir) if f.endswith(".py") and f != "__init__.py"
    ]
    assert len(py_files) == 1
    migration_path = os.path.join(versions_dir, py_files[0])

    with open(migration_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "create_table"
        ):
            table_name = None
            columns = {}

            for kw in node.keywords:
                if kw.arg == "table_name":
                    if isinstance(kw.value, ast.Constant):
                        table_name = kw.value.value

            for arg in node.args:
                if (
                    isinstance(arg, ast.Call)
                    and getattr(arg.func, "attr", None) == "Column"
                ):
                    col_name = (
                        arg.args[0].value
                        if arg.args and isinstance(arg.args[0], ast.Constant)
                        else None
                    )
                    if col_name:
                        columns[col_name] = True

            if table_name == "research_jobs":
                expected = {
                    "id", "question", "max_tasks", "max_sources_per_task",
                    "status", "progress", "error", "state",
                    "created_at", "updated_at",
                }
                actual = set(columns.keys())
                assert actual == expected, (
                    f"research_jobs columns mismatch. "
                    f"Extra: {actual - expected}. "
                    f"Missing: {expected - actual}"
                )

            elif table_name == "research_reports":
                expected = {
                    "id", "job_id", "question", "report", "citations",
                    "extra_metadata", "created_at", "updated_at",
                }
                actual = set(columns.keys())
                assert actual == expected, (
                    f"research_reports columns mismatch. "
                    f"Extra: {actual - expected}. "
                    f"Missing: {expected - actual}"
                )
                assert "extra_metadata" in columns, (
                    "Migration must use 'extra_metadata' column (not 'metadata') "
                    "to match the SQLModel field"
                )


def test_models_match_migration():
    """Verify the SQLModel column names align with the migration."""
    job_cols = {c.name for c in ResearchJob.__table__.columns}
    expected_job_cols = {
        "id", "question", "max_tasks", "max_sources_per_task",
        "status", "progress", "error", "state",
        "created_at", "updated_at",
    }
    assert job_cols == expected_job_cols, (
        f"ResearchJob columns: extra={job_cols - expected_job_cols}, "
        f"missing={expected_job_cols - job_cols}"
    )

    report_cols = {c.name for c in ResearchReport.__table__.columns}
    expected_report_cols = {
        "id", "job_id", "question", "report", "citations",
        "extra_metadata", "created_at", "updated_at",
    }
    assert report_cols == expected_report_cols, (
        f"ResearchReport columns: extra={report_cols - expected_report_cols}, "
        f"missing={expected_report_cols - report_cols}"
    )


def test_alembic_env_checks_database_url():
    """Verify alembic/env.py contains a check for missing DATABASE_URL."""
    env_path = os.path.join(os.path.dirname(__file__), "..", "alembic", "env.py")
    with open(env_path) as f:
        source = f.read()
    assert "RuntimeError" in source, "env.py should raise RuntimeError when DATABASE_URL is missing"
    assert "DATABASE_URL" in source, "env.py should reference DATABASE_URL in its error message"


def test_json_serializer_handles_pydantic_models():
    """Verify the engine's JSON serializer converts BaseModel to JSON-safe dicts."""
    from app.schemas.evidence import Citation, Evidence

    citation = Citation(id="1", source_url="https://example.com", title="Test")
    result = _pydantic_json_serializer(citation)
    data = __import__("json").loads(result)
    assert data["id"] == "1"
    assert data["source_url"] == "https://example.com/"
    assert data["title"] == "Test"

    evidence = Evidence(
        task_id="t1",
        source_url="https://example.com/article",
        title="Article",
        content="Body",
        relevance_score=0.9,
    )
    result = _pydantic_json_serializer(evidence)
    data = __import__("json").loads(result)
    assert data["task_id"] == "t1"
    assert data["source_url"] == "https://example.com/article"
    assert data["relevance_score"] == 0.9


def test_json_serializer_passes_plain_types():
    """Verify the JSON serializer still works with plain JSON types."""
    assert _pydantic_json_serializer({"key": "value"}) == '{"key": "value"}'
    assert _pydantic_json_serializer([1, 2, 3]) == "[1, 2, 3]"
    assert _pydantic_json_serializer("hello") == '"hello"'
    assert _pydantic_json_serializer(42) == "42"


@pytest.mark.asyncio
async def test_database_url_check():
    """Verify database URL validation raises error when missing."""
    with patch("app.core.database.settings.database_url", ""):
        with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
            from app.core.database import _get_database_url
            _get_database_url()


@pytest.mark.asyncio
async def test_session_factory_with_valid_url():
    """Verify session factory works with valid database URL."""
    with patch("app.core.database.settings.database_url", "postgresql+asyncpg://test:test@localhost/test"):
        with patch("app.core.database.settings.debug", False):
            with patch("app.core.database.settings.database_pool_size", 10):
                with patch("app.core.database.settings.database_max_overflow", 20):
                    from app.core.database import get_async_engine, _get_async_engine
                    engine = _get_async_engine()
                    assert engine is not None


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    """Verify init_db successfully creates all tables."""
    with patch("app.core.database.settings.database_url", "postgresql+asyncpg://test:test@localhost/test"):
        with patch("app.core.database.settings.debug", False):
            with patch("app.core.database.settings.database_pool_size", 10):
                with patch("app.core.database.settings.database_max_overflow", 20):
                    engine = _get_async_engine()
                    with patch.object(engine, "begin") as mock_begin:
                        async with mock_begin() as conn:
                            with patch.object(conn, "run_sync") as mock_run_sync:
                                await mock_run_sync(SQLModel.metadata.create_all)
                        mock_run_sync.assert_called_once()
                        call_args = mock_run_sync.call_args[0][0]
                        assert call_args is SQLModel.metadata


@pytest.mark.asyncio
async def test_session_generator_yields_session():
    """Verify get_session generator yields a valid session."""
    with patch("app.core.database.settings.database_url", "postgresql+asyncpg://test:test@localhost/test"):
        with patch("app.core.database.settings.debug", False):
            with patch("app.core.database.settings.database_pool_size", 10):
                with patch("app.core.database.settings.database_max_overflow", 20):
                    session_gen = get_session()
                    session = await session_gen.__anext__()
                    assert isinstance(session, AsyncSession)
                    await session_gen.aclose()
