"""Tests for the Writer agent node."""

from __future__ import annotations

import pytest

from app.core.langgraph.nodes.writer import (
    _build_citations,
    _WriterOutput,
    make_writer_node,
)
from app.schemas.evidence import Citation, Evidence


def _evidence(
    url: str = "https://example.com/a",
    title: str = "A",
    task_id: str = "t1",
    relevance: float = 0.8,
) -> Evidence:
    return Evidence(
        task_id=task_id,
        source_url=url,
        title=title,
        snippet="snippet",
        content="content",
        relevance_score=relevance,
        source_quality=0.7,
    )


def test_build_citations_assigns_sequential_ids() -> None:
    e = [
        _evidence(url="https://e.com/1", title="One"),
        _evidence(url="https://e.com/2", title="Two"),
        _evidence(url="https://e.com/3", title="Three"),
    ]
    citations = _build_citations(e)
    assert [c.id for c in citations] == ["1", "2", "3"]
    assert [c.title for c in citations] == ["One", "Two", "Three"]
    assert all(isinstance(c, Citation) for c in citations)


@pytest.mark.asyncio
async def test_writer_produces_report_with_citations(fake_llm):
    evidence = [
        _evidence(url="https://e.com/a", title="A"),
        _evidence(url="https://e.com/b", title="B"),
    ]
    fake_llm.queue(
        _WriterOutput(
            summary="Short summary [1][2].",
            body_markdown="# Title\n\n## Summary\nIt is so [1].",
        )
    )

    node = make_writer_node(llm=fake_llm)
    result = await node({"question": "Q", "verified_evidence": evidence})

    assert result["status"] == "complete"
    assert result["summary"] == "Short summary [1][2]."
    assert result["report_markdown"].startswith("# Title")
    assert len(result["citations"]) == 2
    assert [c.id for c in result["citations"]] == ["1", "2"]


@pytest.mark.asyncio
async def test_writer_emits_insufficient_evidence_when_empty(fake_llm):
    node = make_writer_node(llm=fake_llm)
    result = await node({"question": "Q", "verified_evidence": []})

    assert result["status"] == "complete"
    assert result["citations"] == []
    assert "Insufficient" in result["summary"] or "sufficient" in result["summary"].lower()
    assert "## Sources" in result["report_markdown"]
    # Fake LLM should not have been called.
    assert fake_llm.invocations == []


@pytest.mark.asyncio
async def test_writer_surfaces_llm_error(fake_llm):
    evidence = [_evidence()]
    fake_llm.queue(RuntimeError("writer down"))

    node = make_writer_node(llm=fake_llm)
    result = await node({"question": "Q", "verified_evidence": evidence})

    assert result["status"] == "error"
    assert "writer down" in result["error"]
    # Citations are still built so downstream can render the source list.
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_writer_uses_default_question_when_missing(fake_llm):
    node = make_writer_node(llm=fake_llm)
    result = await node({"question": "", "verified_evidence": []})

    assert result["status"] == "complete"
    assert "no question provided" in result["report_markdown"]
