"""Tests for the Researcher agent node."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.core.langgraph.nodes.researcher import (
    _ResearcherItem,
    _ResearcherOutput,
    _source_quality,
    make_researcher_node,
)
from app.core.langgraph.tools.web_search import SerperSearchResult
from app.schemas.evidence import Evidence, ResearchTask


class FakeSerperClient:
    """Minimal stand-in for SerperClient."""

    def __init__(self, responses: dict[str, list[SerperSearchResult]] | None = None):
        self._responses = responses or {}
        self.queries: list[str] = []
        self.aclose_called = False

    async def search(
        self, query: str, *, num: int = 10, **_kwargs: Any
    ) -> list[SerperSearchResult]:
        self.queries.append(query)
        return self._responses.get(query, [])

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.fixture
def pipeline_settings() -> Settings:
    return Settings(
        max_research_tasks=5,
        max_sources_per_task=2,
        min_source_relevance=0.5,
    )


def _serper_result(idx: int, link: str, title: str) -> SerperSearchResult:
    return SerperSearchResult(title=title, link=link, snippet=f"snippet {idx}", position=idx)


class TestSourceQualityHeuristic:
    def test_high_quality_tld(self) -> None:
        assert _source_quality("https://nist.gov/page") >= 0.85

    def test_high_quality_domain(self) -> None:
        assert _source_quality("https://arxiv.org/abs/1234") >= 0.8

    def test_medium_quality_domain(self) -> None:
        assert 0.5 <= _source_quality("https://reuters.com/article") < 0.8

    def test_low_quality_hint(self) -> None:
        assert _source_quality("https://something.blogspot.com/x") < 0.5

    def test_unknown_domain_default(self) -> None:
        assert _source_quality("https://random-blog.example/x") == 0.5

    def test_malformed_url(self) -> None:
        assert _source_quality("not a url") < 0.5


@pytest.mark.asyncio
async def test_researcher_extracts_evidence_for_each_task(fake_llm, pipeline_settings):
    task1 = ResearchTask(id="t1", question="what is X?")
    task2 = ResearchTask(id="t2", question="how is X measured?")

    search_responses = {
        "what is X?": [
            _serper_result(1, "https://example.com/a", "A"),
            _serper_result(2, "https://example.com/b", "B"),
        ],
        "how is X measured?": [
            _serper_result(1, "https://example.com/c", "C"),
        ],
    }
    client = FakeSerperClient(responses=search_responses)

    # One output per task (since `with_structured_output` is called per task).
    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=1, relevance_score=0.9, content="A says X is..."),
                _ResearcherItem(source_index=2, relevance_score=0.7, content="B agrees."),
            ]
        )
    )
    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=1, relevance_score=0.8, content="C measures X by..."),
            ]
        )
    )

    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": [task1, task2]})

    assert result["status"] == "fact_checking"
    evidence = result["evidence"]
    assert len(evidence) == 3
    assert all(isinstance(e, Evidence) for e in evidence)
    # External client should NOT be closed by the node.
    assert client.aclose_called is False
    # Each task triggered exactly one search.
    assert set(client.queries) == {"what is X?", "how is X measured?"}


@pytest.mark.asyncio
async def test_researcher_filters_below_relevance_threshold(fake_llm, pipeline_settings):
    task = ResearchTask(id="t1", question="what is q?")
    client = FakeSerperClient(
        responses={"what is q?": [_serper_result(1, "https://example.com/a", "A")]}
    )

    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=1, relevance_score=0.3, content="weak"),
            ]
        )
    )

    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": [task]})

    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_researcher_caps_evidence_per_task(fake_llm, pipeline_settings):
    task = ResearchTask(id="t1", question="what is q?")
    client = FakeSerperClient(
        responses={
            "what is q?": [_serper_result(i, f"https://e.com/{i}", f"T{i}") for i in range(1, 6)]
        }
    )

    # LLM picks 5 items above threshold; cap is 2.
    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=i, relevance_score=0.6 + i * 0.05, content=f"c{i}")
                for i in range(1, 6)
            ]
        )
    )

    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": [task]})

    assert len(result["evidence"]) == 2
    # Should keep the two highest-relevance ones.
    scores = [e.relevance_score for e in result["evidence"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_researcher_ignores_invalid_source_index(fake_llm, pipeline_settings):
    task = ResearchTask(id="t1", question="what is q?")
    client = FakeSerperClient(
        responses={"what is q?": [_serper_result(1, "https://e.com/1", "T1")]}
    )

    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=99, relevance_score=0.9, content="bad index"),
                _ResearcherItem(source_index=1, relevance_score=0.9, content="ok"),
            ]
        )
    )

    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": [task]})

    assert len(result["evidence"]) == 1
    assert result["evidence"][0].content == "ok"


@pytest.mark.asyncio
async def test_researcher_handles_no_search_results(fake_llm, pipeline_settings):
    task = ResearchTask(id="t1", question="what is q?")
    client = FakeSerperClient(responses={"what is q?": []})

    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": [task]})

    assert result["evidence"] == []
    assert result["status"] == "fact_checking"


@pytest.mark.asyncio
async def test_researcher_errors_when_no_tasks(fake_llm, pipeline_settings):
    client = FakeSerperClient()
    node = make_researcher_node(llm=fake_llm, search_client=client, settings=pipeline_settings)
    result = await node({"research_tasks": []})

    assert result["status"] == "error"
    assert result["evidence"] == []
