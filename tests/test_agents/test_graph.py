"""End-to-end smoke test for the research StateGraph.

Wires all four nodes with mocked LLM and Serper, then invokes the
compiled graph against a sample question. Verifies state flows through
Planner -> Researcher -> Fact Checker -> Writer and produces a citation-
bearing report.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.core.langgraph.graph import build_research_graph
from app.core.langgraph.nodes.planner import PlannerOutput
from app.core.langgraph.nodes.researcher import _ResearcherItem, _ResearcherOutput
from app.core.langgraph.nodes.writer import _WriterOutput
from app.core.langgraph.tools.web_search import SerperSearchResult
from app.schemas.evidence import ResearchTask


class FakeSerperClient:
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


def _result(idx: int, link: str, title: str) -> SerperSearchResult:
    return SerperSearchResult(title=title, link=link, snippet=f"snippet {idx}", position=idx)


def _enqueue_full_pipeline(fake_llm) -> None:
    """Queue canned LLM outputs for: planner, 2x researcher, fact-checker, writer."""
    # 1. Planner: 2 tasks.
    fake_llm.queue(
        PlannerOutput(
            tasks=[
                ResearchTask(id="t1", question="what is X?", rationale="defines"),
                ResearchTask(id="t2", question="how is X measured?", rationale="metrics"),
            ]
        )
    )
    # 2. Researcher for task t1: pick result 1.
    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(source_index=1, relevance_score=0.9, content="X is defined as ..."),
            ]
        )
    )
    # 3. Researcher for task t2: pick result 1.
    fake_llm.queue(
        _ResearcherOutput(
            items=[
                _ResearcherItem(
                    source_index=1, relevance_score=0.85, content="X is measured by ..."
                ),
            ]
        )
    )
    # 4. Fact-checker: keep both, slight re-score.
    # The assessment must reference the actual evidence fingerprints; since we
    # don't know them yet, we'll fall back through the LLM-failure path by
    # queueing a wrong-shape output. Instead, queue a wildcard "keep all" by
    # constructing assessments from the URLs we know the researcher selects.
    # Simpler: queue a RuntimeError so the fact-checker falls back to the
    # deterministic dedup path (which is part of the contract).
    fake_llm.queue(RuntimeError("fact-checker LLM down - fall back to deterministic"))
    # 5. Writer.
    fake_llm.queue(
        _WriterOutput(
            summary="X is a thing [1]. It is measured by Y [2].",
            body_markdown=(
                "# X explained\n\n"
                "## Summary\nX is a thing [1]. It is measured by Y [2].\n\n"
                "## Sources\n1. A\n2. C\n"
            ),
        )
    )


@pytest.mark.asyncio
async def test_full_graph_runs_end_to_end(fake_llm):
    search_responses = {
        "what is X?": [_result(1, "https://e.com/a", "A"), _result(2, "https://e.com/b", "B")],
        "how is X measured?": [_result(1, "https://e.com/c", "C")],
    }
    serper = FakeSerperClient(responses=search_responses)
    settings = Settings(
        max_research_tasks=3,
        max_sources_per_task=2,
        min_source_relevance=0.5,
    )

    _enqueue_full_pipeline(fake_llm)

    graph = build_research_graph(llm=fake_llm, search_client=serper, settings=settings)

    final = await graph.ainvoke({"question": "Tell me about X", "max_tasks": 2})

    assert final["status"] == "complete"
    assert final["summary"].startswith("X is a thing")
    assert "# X explained" in final["report_markdown"]
    citations = final["citations"]
    assert len(citations) == 2
    assert [c.id for c in citations] == ["1", "2"]
    # The graph should have triggered both searches.
    assert set(serper.queries) == {"what is X?", "how is X measured?"}


@pytest.mark.asyncio
async def test_graph_short_circuits_on_planner_error(fake_llm):
    # Planner raises -> error status -> END (no other LLM calls consumed).
    fake_llm.queue(RuntimeError("planner down"))

    serper = FakeSerperClient()
    settings = Settings()

    graph = build_research_graph(llm=fake_llm, search_client=serper, settings=settings)

    final = await graph.ainvoke({"question": "Q"})

    assert final["status"] == "error"
    assert "planner down" in final["error"]
    # No search should have run.
    assert serper.queries == []
    # No more LLM outputs were consumed.
    assert fake_llm.remaining == 0


@pytest.mark.asyncio
async def test_graph_handles_empty_evidence_path(fake_llm):
    """Planner succeeds, researcher returns nothing, writer emits fallback."""
    fake_llm.queue(PlannerOutput(tasks=[ResearchTask(id="t1", question="what is X?")]))
    # Researcher LLM is called once but returns no items.
    fake_llm.queue(_ResearcherOutput(items=[]))

    serper = FakeSerperClient(responses={"what is X?": [_result(1, "https://e.com/a", "A")]})
    settings = Settings(max_research_tasks=2, max_sources_per_task=2)

    graph = build_research_graph(llm=fake_llm, search_client=serper, settings=settings)

    final = await graph.ainvoke({"question": "Q", "max_tasks": 1})

    assert final["status"] == "complete"
    assert final["citations"] == []
    # Writer fallback report uses canonical Markdown structure.
    assert "## Sources" in final["report_markdown"]
