"""Tests for the Postgres checkpointer (Worker A).

Focuses on DSN validation and graceful degradation when Postgres is
unavailable, without needing a real database connection.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_settings():
    with patch("app.services.checkpoint.settings") as mock:
        yield mock


class TestGetSyncDsn:
    def test_returns_sync_url_for_asyncpg_url(self, mock_settings):
        from app.services.checkpoint import _get_sync_dsn

        mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        result = _get_sync_dsn()
        assert result == "postgresql://user:pass@localhost:5432/db"

    def test_passes_through_sync_url(self, mock_settings):
        from app.services.checkpoint import _get_sync_dsn

        mock_settings.database_url = "postgresql://user:pass@localhost:5432/db"
        result = _get_sync_dsn()
        assert result == "postgresql://user:pass@localhost:5432/db"

    def test_returns_none_for_empty_url(self, mock_settings):
        from app.services.checkpoint import _get_sync_dsn

        mock_settings.database_url = ""
        assert _get_sync_dsn() is None

    def test_returns_none_for_sqlite_url(self, mock_settings):
        from app.services.checkpoint import _get_sync_dsn

        mock_settings.database_url = "sqlite:///test.db"
        assert _get_sync_dsn() is None

    def test_returns_none_for_non_postgres_url(self, mock_settings):
        from app.services.checkpoint import _get_sync_dsn

        mock_settings.database_url = "mysql://user:pass@localhost/db"
        assert _get_sync_dsn() is None


class TestGetCheckpointer:
    def test_returns_none_when_psycopg_unavailable(self):
        with patch("app.services.checkpoint.HAS_CHECKPOINTER", False):
            from app.services.checkpoint import get_checkpointer

            result = get_checkpointer()
            import asyncio

            assert asyncio.run(result) is None

    def test_returns_none_when_dsn_invalid(self, mock_settings):
        mock_settings.database_url = ""
        from app.services.checkpoint import get_checkpointer

        result = get_checkpointer()
        import asyncio

        assert asyncio.run(result) is None
