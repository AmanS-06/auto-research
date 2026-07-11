from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.langgraph.state import ResearchState
from app.schemas.evidence import Evidence
from app.schemas.research import ResearchRequest
from app.services.research_service import ResearchService


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def research_service(mock_session):
    return ResearchService(mock_session)


@pytest.fixture
def sample_request():
    return ResearchRequest(
        question="What are the latest developments in quantum computing?",
        max_tasks=3,
        max_sources_per_task=2,
    )


@pytest.mark.asyncio
async def test_research_service_initialization(research_service):
    assert research_service.session is not None
    assert research_service.graph is None


@pytest.mark.asyncio
async def test_planner_node():
    from app.core.langgraph.nodes.planner import planner_node

    state: ResearchState = {
        "question": "Test question",
        "research_tasks": [],
        "evidence": [],
        "verified_evidence": [],
        "report": "",
        "citations": [],
        "status": "planning",
        "error": None,
        "job_id": uuid4(),
        "current_task_index": 0,
        "max_tasks": 3,
    }

    from app.core.langgraph.nodes.planner import PlannerOutput

    fake_planner_output = PlannerOutput(
        tasks=[
            {"id": "t1", "question": "What is quantum computing?", "rationale": "Core definition"},
            {"id": "t2", "question": "Who are the key players?", "rationale": "Market landscape"},
        ]
    )
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=fake_planner_output)
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured
    with patch("app.core.langgraph.nodes.planner.get_default_llm", return_value=mock_chat):
        result = await planner_node(state)

        assert result["status"] == "researching"
        assert len(result["research_tasks"]) == 2


@pytest.mark.asyncio
async def test_citation_formatting():
    from app.core.citations import format_citations, generate_bibliography

    evidence = [
        Evidence(
            source_url="https://example.com/1",
            title="Test Source 1",
            content="Content 1",
            relevance_score=0.9,
            task_id="task_0",
        ),
        Evidence(
            source_url="https://example.com/2",
            title="Test Source 2",
            content="Content 2",
            relevance_score=0.8,
            task_id="task_0",
        ),
    ]

    citations = format_citations(evidence)
    assert len(citations) == 2
    assert citations[0].id == "[1]"
    assert citations[1].id == "[2]"

    bib = generate_bibliography(citations)
    assert "References" in bib
    assert "Test Source 1" in bib
    assert "Test Source 2" in bib


@pytest.mark.asyncio
async def test_deduplication():
    from app.core.citations import deduplicate_evidence

    evidence = [
        Evidence(
            source_url="https://example.com/1",
            title="Test",
            content="This is very similar content about AI",
            relevance_score=0.9,
            task_id="task_0",
        ),
        Evidence(
            source_url="https://example.com/2",
            title="Test 2",
            content="This is very similar content about AI",
            relevance_score=0.8,
            task_id="task_0",
        ),
        Evidence(
            source_url="https://example.com/3",
            title="Test 3",
            content="Completely different content about quantum physics",
            relevance_score=0.7,
            task_id="task_1",
        ),
    ]

    result = deduplicate_evidence(evidence)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_source_scoring():
    from app.core.citations import score_source_quality

    evidence = Evidence(
        source_url="https://arxiv.org/abs/1234.5678",
        title="ArXiv Paper",
        content="Quantum computing research",
        relevance_score=0.8,
        task_id="task_0",
    )

    score = score_source_quality(evidence)
    assert score > 0.8

    evidence2 = Evidence(
        source_url="https://unknown-blog.com/post",
        title="Random Blog",
        content="Some content",
        relevance_score=0.5,
        task_id="task_0",
    )

    score2 = score_source_quality(evidence2)
    assert score2 < score
