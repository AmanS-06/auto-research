"""Tests for the Planner agent node."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.langgraph.nodes.planner import PlannerOutput, make_planner_node
from app.schemas.evidence import ResearchTask


@pytest.fixture
def planner_settings() -> Settings:
    # Tight cap so we can test clipping.
    return Settings(max_research_tasks=3, max_sources_per_task=3)


@pytest.mark.asyncio
async def test_planner_returns_tasks_and_advances_status(fake_llm, planner_settings):
    fake_llm.queue(
        PlannerOutput(
            tasks=[
                ResearchTask(id="t1", question="What is X?", rationale="defines X"),
                ResearchTask(id="t2", question="How is X measured?", rationale="metrics"),
            ]
        )
    )
    node = make_planner_node(llm=fake_llm, settings=planner_settings)

    result = await node({"question": "Tell me about X"})

    assert result["status"] == "researching"
    assert len(result["research_tasks"]) == 2
    assert all(isinstance(t, ResearchTask) for t in result["research_tasks"])
    assert [t.id for t in result["research_tasks"]] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_planner_clips_to_max_tasks(fake_llm, planner_settings):
    many = [ResearchTask(id=f"orig{i}", question=f"question number {i}?") for i in range(10)]
    fake_llm.queue(PlannerOutput(tasks=many))
    node = make_planner_node(llm=fake_llm, settings=planner_settings)

    result = await node({"question": "Q"})

    # MAX_RESEARCH_TASKS=3
    assert len(result["research_tasks"]) == 3
    # IDs are re-numbered t1..tN by the normalizer.
    assert [t.id for t in result["research_tasks"]] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_planner_deduplicates_identical_questions(fake_llm, planner_settings):
    fake_llm.queue(
        PlannerOutput(
            tasks=[
                ResearchTask(id="a", question="What is X?"),
                ResearchTask(id="b", question="what is x?"),  # case-insensitive dup
                ResearchTask(id="c", question="How is X used?"),
            ]
        )
    )
    node = make_planner_node(llm=fake_llm, settings=planner_settings)

    result = await node({"question": "Q"})

    assert len(result["research_tasks"]) == 2
    assert [t.id for t in result["research_tasks"]] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_planner_errors_on_empty_question(fake_llm, planner_settings):
    node = make_planner_node(llm=fake_llm, settings=planner_settings)
    result = await node({"question": "   "})

    assert result["status"] == "error"
    assert "empty question" in result["error"].lower()
    assert result["research_tasks"] == []


@pytest.mark.asyncio
async def test_planner_surfaces_llm_error(fake_llm, planner_settings):
    fake_llm.queue(RuntimeError("boom"))
    node = make_planner_node(llm=fake_llm, settings=planner_settings)

    result = await node({"question": "Q"})

    assert result["status"] == "error"
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_planner_caps_max_tasks_from_state(fake_llm, planner_settings):
    fake_llm.queue(
        PlannerOutput(
            tasks=[
                ResearchTask(id="t1", question="q1?"),
                ResearchTask(id="t2", question="q2?"),
                ResearchTask(id="t3", question="q3?"),
            ]
        )
    )
    node = make_planner_node(llm=fake_llm, settings=planner_settings)

    # Caller asks for 2 (below the settings cap of 3) -> we respect the caller.
    result = await node({"question": "Q", "max_tasks": 2})

    assert len(result["research_tasks"]) == 2
