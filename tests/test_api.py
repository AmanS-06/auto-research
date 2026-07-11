"""Tests for API endpoints (Worker A).

Uses FastAPI TestClient with mocked database session to test route behavior.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_session
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session


@pytest.fixture
def override_get_session(mock_session):
    async def _override():
        yield mock_session

    app.dependency_overrides[get_session] = _override
    yield
    app.dependency_overrides.clear()


@patch("app.api.v1.research.run_research_pipeline")
class TestCreateResearch:
    def test_creates_job_and_returns_202(
        self, mock_run_pipeline, client, mock_session, override_get_session
    ):
        response = client.post(
            "/api/v1/research",
            json={"question": "What is quantum computing?"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        mock_run_pipeline.assert_called_once()

    def test_accepts_custom_limits(
        self, mock_run_pipeline, client, mock_session, override_get_session
    ):
        response = client.post(
            "/api/v1/research",
            json={
                "question": "Latest AI developments?",
                "max_tasks": 10,
                "max_sources_per_task": 5,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        args, _ = mock_run_pipeline.call_args
        assert args[1].max_tasks == 10
        assert args[1].max_sources_per_task == 5

    def test_rejects_empty_question(self, mock_run_pipeline, client, override_get_session):
        response = client.post(
            "/api/v1/research",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_rejects_missing_question(self, mock_run_pipeline, client, override_get_session):
        response = client.post(
            "/api/v1/research",
            json={},
        )
        assert response.status_code == 422


class TestGetResearchStatus:
    def test_returns_status_for_pending_job(self, client, mock_session, override_get_session):
        from app.models.research import ResearchJob

        job_id = uuid.uuid4()
        job = ResearchJob(
            id=job_id,
            question="Test?",
            status="pending",
            progress="Waiting...",
        )
        mock_session.get.return_value = job

        response = client.get(f"/api/v1/research/{job_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(job_id)
        assert data["status"] == "pending"
        assert data["progress"] == "Waiting..."

    def test_returns_404_for_missing_job(self, client, mock_session, override_get_session):
        mock_session.get.return_value = None

        response = client.get(f"/api/v1/research/{uuid.uuid4()}/status")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_error_for_failed_job(self, client, mock_session, override_get_session):
        from app.models.research import ResearchJob

        job_id = uuid.uuid4()
        job = ResearchJob(
            id=job_id,
            question="Test?",
            status="failed",
            error="Something went wrong",
            progress="Research failed",
        )
        mock_session.get.return_value = job

        response = client.get(f"/api/v1/research/{job_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Something went wrong"


class TestGetResearch:
    def test_returns_report_for_complete_job(self, client, mock_session, override_get_session):
        from unittest.mock import MagicMock

        from app.models.research import ResearchJob, ResearchReport
        from app.schemas.evidence import Citation

        job_id = uuid.uuid4()
        job = ResearchJob(
            id=job_id,
            question="Test?",
            status="complete",
        )
        mock_session.get.return_value = job

        report = ResearchReport(
            job_id=job_id,
            question="Test?",
            report="# Final Report\n\nContent here.",
            citations=[
                Citation(
                    id="1",
                    source_url="https://example.com",
                    title="Example",
                )
            ],
        )
        mock_exec = MagicMock()
        mock_exec.first.return_value = report

        async def mock_exec_fn(*a, **kw):
            return mock_exec

        mock_session.exec = mock_exec_fn

        response = client.get(f"/api/v1/research/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert "# Final Report" in data["report"]
        assert len(data["citations"]) == 1

    def test_returns_status_for_incomplete_job(self, client, mock_session, override_get_session):
        from app.models.research import ResearchJob

        job_id = uuid.uuid4()
        job = ResearchJob(
            id=job_id,
            question="Test?",
            status="running",
            progress="Working...",
        )
        mock_session.get.return_value = job

        response = client.get(f"/api/v1/research/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["report"] is None

    def test_returns_error_for_failed_job(self, client, mock_session, override_get_session):
        from app.models.research import ResearchJob

        job_id = uuid.uuid4()
        job = ResearchJob(
            id=job_id,
            question="Test?",
            status="failed",
            error="Graph error",
        )
        mock_session.get.return_value = job

        response = client.get(f"/api/v1/research/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Graph error"

    def test_returns_404_for_missing_job(self, client, mock_session, override_get_session):
        mock_session.get.return_value = None

        response = client.get(f"/api/v1/research/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Autonomous Research Pipeline API" in data["message"]
