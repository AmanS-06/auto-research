"""Tests for the Fact Checker agent node."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.langgraph.nodes.fact_checker import (
    _Assessment,
    _deduplicate,
    _FactCheckOutput,
    make_fact_checker_node,
)
from app.schemas.evidence import Evidence


def _evidence(
    url: str = "https://example.com/a",
    title: str = "A",
    task_id: str = "t1",
    relevance: float = 0.7,
    quality: float = 0.6,
    content: str = "some content",
) -> Evidence:
    return Evidence(
        task_id=task_id,
        source_url=url,
        title=title,
        snippet="s",
        content=content,
        relevance_score=relevance,
        source_quality=quality,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(min_source_relevance=0.5)


class TestDeduplicate:
    def test_keeps_highest_relevance_per_fingerprint(self) -> None:
        a_lo = _evidence(url="https://e.com/a", title="A", relevance=0.4)
        a_hi = _evidence(url="https://e.com/a/", title="a", relevance=0.9)  # same fingerprint
        b = _evidence(url="https://e.com/b", title="B", relevance=0.5)

        result = _deduplicate([a_lo, a_hi, b])
        assert len(result) == 2
        kept_a = next(e for e in result if "/a" in str(e.source_url))
        assert kept_a.relevance_score == 0.9


@pytest.mark.asyncio
async def test_fact_checker_dedups_and_applies_assessments(fake_llm, settings):
    e1 = _evidence(url="https://e.com/a", title="A", relevance=0.7)
    e1_dup = _evidence(url="https://e.com/a/", title="a", relevance=0.6)  # same fingerprint
    e2 = _evidence(url="https://e.com/b", title="B", relevance=0.6)
    e3 = _evidence(url="https://e.com/c", title="C", relevance=0.6)

    fake_llm.queue(
        _FactCheckOutput(
            assessments=[
                _Assessment(
                    fingerprint=e1.fingerprint,
                    keep=True,
                    relevance_score=0.9,
                    source_quality=0.8,
                    reason="great",
                ),
                _Assessment(
                    fingerprint=e2.fingerprint,
                    keep=False,
                    relevance_score=0.5,
                    source_quality=0.4,
                    reason="off-topic",
                ),
                _Assessment(
                    fingerprint=e3.fingerprint,
                    keep=True,
                    relevance_score=0.7,
                    source_quality=0.7,
                    reason="ok",
                ),
            ]
        )
    )

    node = make_fact_checker_node(llm=fake_llm, settings=settings)
    result = await node({"question": "q", "evidence": [e1, e1_dup, e2, e3]})

    assert result["status"] == "writing"
    verified = result["verified_evidence"]
    kept_urls = {str(e.source_url).rstrip("/") for e in verified}
    assert kept_urls == {"https://e.com/a", "https://e.com/c"}

    # Sorted by (relevance, quality) desc.
    assert verified[0].relevance_score >= verified[-1].relevance_score
    # Revised scores were applied.
    a = next(e for e in verified if str(e.source_url).rstrip("/") == "https://e.com/a")
    assert a.relevance_score == pytest.approx(0.9)
    assert a.source_quality == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_fact_checker_drops_below_threshold_before_llm(fake_llm, settings):
    weak = _evidence(relevance=0.2)  # below MIN_SOURCE_RELEVANCE=0.5

    # No LLM call should happen because nothing passes the floor.
    node = make_fact_checker_node(llm=fake_llm, settings=settings)
    result = await node({"question": "q", "evidence": [weak]})

    assert result["status"] == "writing"
    assert result["verified_evidence"] == []
    # Fake LLM should not have been called.
    assert fake_llm.invocations == []


@pytest.mark.asyncio
async def test_fact_checker_drops_empty_content_before_llm(fake_llm, settings):
    blank = _evidence(content="   ", relevance=0.9)

    node = make_fact_checker_node(llm=fake_llm, settings=settings)
    result = await node({"question": "q", "evidence": [blank]})

    assert result["verified_evidence"] == []
    assert fake_llm.invocations == []


@pytest.mark.asyncio
async def test_fact_checker_handles_no_evidence(fake_llm, settings):
    node = make_fact_checker_node(llm=fake_llm, settings=settings)
    result = await node({"question": "q", "evidence": []})

    assert result["verified_evidence"] == []
    assert result["status"] == "writing"


@pytest.mark.asyncio
async def test_fact_checker_falls_back_on_llm_failure(fake_llm, settings):
    e1 = _evidence(url="https://e.com/a", relevance=0.8)
    e2 = _evidence(url="https://e.com/b", relevance=0.6)

    fake_llm.queue(RuntimeError("llm down"))

    node = make_fact_checker_node(llm=fake_llm, settings=settings)
    result = await node({"question": "q", "evidence": [e1, e2]})

    assert result["status"] == "writing"
    # Both pass the floor, so both are kept (sorted by relevance).
    assert len(result["verified_evidence"]) == 2
    assert (
        result["verified_evidence"][0].relevance_score
        >= result["verified_evidence"][1].relevance_score
    )
