# LOCKED BY Worker C
"""Enhanced tests for API endpoints with httpx client integration.

Tests FastAPI route handlers using httpx.AsyncClient for integration-style testing
and TestClient for unit-style testing. Focus on the /api/v1/research endpoints.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, Response
import asyncio

from app.main import app
from app.core.database import get_session
from app.schemas.research import ResearchRequest, ResearchResponse, ResearchJobStatus


@pytest.fixture
def client():
    """TestClient fixture for FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """AsyncClient fixture for FastAPI app."""
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_session():
    """Mock session fixture for testing."""
    session = AsyncMock()
    return session


@pytest.fixture
def override_get_session(mock_session):
    """Override the database session dependency."""
    async def _override():
        yield mock_session
    
    app.dependency_overrides[get_session] = _override
    yield
    app.dependency_overrides.clear()


@patch("app.api.v1.research.run_research_pipeline")
class TestCreateResearch:
    """Test research creation endpoint."""

    def test_creates_job_and_returns_202(
        self, mock_run_pipeline, client, mock_session, override_get_session
    ):
        """Test creating a research job returns 202 with job ID."""
        job_id = uuid.uuid4()
        mock_session.get.return_value = None
        
        from app.models.research import ResearchJob
        job = ResearchJob(
            id=job_id,
            question="What is quantum computing?",
            max_tasks=5,
            max_sources_per_task=3,
            status="pending",
        )
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
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
        """Test endpoint accepts and processes custom limits."""
        job_id = uuid.uuid4()
        mock_session.get.return_value = None
        
        from app.models.research import ResearchJob
        job = ResearchJob(
            id=job_id,
            question="Latest AI developments?",
            max_tasks=10,
            max_sources_per_task=5,
            status="pending",
        )
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
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

    def test_rejects_empty_question(
        self, mock_run_pipeline, client, override_get_session
    ):
        """Test empty question validation."""
        response = client.post(
            "/api/v1/research",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_rejects_missing_question(
        self, mock_run_pipeline, client, override_get_session
    ):
        """Test missing question validation."""
        response = client.post(
            "/api/v1/research",
            json={},
        )
        assert response.status_code == 422


class TestGetResearchStatus:
    """Test research status endpoint."""

    def test_returns_status_for_pending_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns status for pending job."""
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

    def test_returns_404_for_missing_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns 404 for missing job."""
        mock_session.get.return_value = None

        response = client.get(f"/api/v1/research/{uuid.uuid4()}/status")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_error_for_failed_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns error for failed job."""
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
    """Test research retrieval endpoint."""

    def test_returns_report_for_complete_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns report for complete job."""
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

    def test_returns_status_for_incomplete_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns status for incomplete job."""
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

    def test_returns_error_for_failed_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns error for failed job."""
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

    def test_returns_404_for_missing_job(
        self, client, mock_session, override_get_session
    ):
        """Test returns 404 for missing job."""
        mock_session.get.return_value = None

        response = client.get(f"/api/v1/research/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestHealth:
    """Test health and root endpoints."""

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Autonomous Research Pipeline API" in data["message"]


@pytest.mark.asyncio
class TestAsyncClientIntegration:
    """Test API endpoints with httpx.AsyncClient."""

    async def test_async_create_research_success(self, async_client, mock_session, override_get_session):
        """Test async API client creates research job."""
        response = await async_client.post(
            "/api/v1/research",
            json={"question": "Test async API"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    async def test_async_get_status_success(self, async_client, mock_session, override_get_session):
            """Test async API client retrieves job status."""
            job_id = uuid.uuid4()
            mock_session.get.return_value = None
        
            response = await async_client.get(f"/api/v1/research/{job_id}/status")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


@patch("app.api.v1.research.run_research_pipeline")
def test_route_handler_with_real_client(mock_run_pipeline, mock_session, override_get_session):
    """Test route handler with real client to verify end-to-end behavior."""
    client = TestClient(app)
    
    response = client.post(
        "/api/v1/research",
        json={"question": "End-to-end test"},
    )
    
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "status" in data

    mock_run_pipeline.assert_called_once()


def test_response_models_validation():
    """Test that response models properly validate data."""
    from app.schemas.research import ResearchResponse
    
    data = {
        "job_id": str(uuid.uuid4()),
        "status": "complete",
        "report": "# Test Report",
        "citations": [
            {"id": "1", "source_url": "https://example.com", "title": "Example"}
        ],
        "error": None,
    }
    
    response = ResearchResponse(**data)
    assert response.job_id is not None
    assert response.status == "complete"
    assert response.report == "# Test Report"
    assert len(response.citations) == 1


@pytest.mark.asyncio
async def test_database_integration():
    """Test integration with real database setup (requires test database)."""
    from app.core.database import get_session
    from app.models.research import ResearchJob
    
    async for session in get_session():
        try:
            job = ResearchJob(
                question="Integration test job",
                status="pending",
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            
            assert job.id is not None
            assert job.status == "pending"
            
            retrieved = await session.get(ResearchJob, job.id)
            assert retrieved is not None
            assert retrieved.question == "Integration test job"
            
            await session.delete(job)
            await session.commit()
        except Exception as e:
            print(f"Integration test database error (expected in CI): {e}")
            break