"""Tests for shared schemas (Evidence, Citation, ResearchTask, Report)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    Citation,
    Evidence,
    ResearchReport,
    ResearchTask,
)
from app.schemas.research import (
    MAX_SOURCES_PER_TASK_CAP,
    MAX_TASKS_CAP,
    ResearchRequest,
)


def _evidence(
    url: str = "https://example.com/a", title: str = "A", task_id: str = "t1"
) -> Evidence:
    return Evidence(
        task_id=task_id,
        source_url=url,
        title=title,
        snippet="snippet",
        content="some content",
        relevance_score=0.8,
        source_quality=0.7,
    )


class TestResearchTask:
    def test_valid_task(self) -> None:
        task = ResearchTask(id="t1", question="What is X?", rationale="defines X")
        assert task.id == "t1"
        assert task.question == "What is X?"

    def test_strips_id_whitespace(self) -> None:
        task = ResearchTask(id="  t1  ", question="What is X?")
        assert task.id == "t1"

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            ResearchTask(id="   ", question="What is X?")

    def test_rejects_short_question(self) -> None:
        with pytest.raises(ValidationError):
            ResearchTask(id="t1", question="?")


class TestEvidence:
    def test_valid_evidence(self) -> None:
        e = _evidence()
        assert e.relevance_score == 0.8
        assert e.source_quality == 0.7

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(
                task_id="t1",
                source_url="https://example.com/a",
                title="A",
                relevance_score=1.5,
            )

    def test_fingerprint_normalizes_url_and_title(self) -> None:
        e1 = _evidence(url="https://example.com/a/", title="A Title")
        e2 = _evidence(url="https://example.com/a", title="a title")
        assert e1.fingerprint == e2.fingerprint

    def test_fingerprint_differs_for_different_urls(self) -> None:
        e1 = _evidence(url="https://example.com/a", title="A")
        e2 = _evidence(url="https://example.com/b", title="A")
        assert e1.fingerprint != e2.fingerprint


class TestCitation:
    def test_valid_citation(self) -> None:
        c = Citation(
            id="1",
            source_url="https://example.com",
            title="t",
            snippet="s",
            task_id="t1",
        )
        assert c.id == "1"

    def test_task_id_optional(self) -> None:
        c = Citation(id="1", source_url="https://example.com", title="t")
        assert c.task_id is None


class TestResearchReport:
    def test_minimal_report(self) -> None:
        report = ResearchReport(
            question="Q",
            summary="S",
            body_markdown="# Title\n\nbody",
        )
        assert report.status == "complete"
        assert report.citations == []


class TestResearchRequest:
    def test_defaults(self) -> None:
        req = ResearchRequest(question="hello")
        assert req.max_tasks == 5
        assert req.max_sources_per_task == 3

    def test_rejects_zero_max_tasks(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="hello", max_tasks=0)

    def test_rejects_negative_max_tasks(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="hello", max_tasks=-1)

    def test_rejects_huge_max_tasks(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="hello", max_tasks=10_000)

    def test_accepts_max_tasks_at_cap(self) -> None:
        req = ResearchRequest(question="hello", max_tasks=MAX_TASKS_CAP)
        assert req.max_tasks == MAX_TASKS_CAP

    def test_rejects_zero_max_sources(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="hello", max_sources_per_task=0)

    def test_rejects_huge_max_sources(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="hello", max_sources_per_task=1000)

    def test_accepts_max_sources_at_cap(self) -> None:
        req = ResearchRequest(question="hello", max_sources_per_task=MAX_SOURCES_PER_TASK_CAP)
        assert req.max_sources_per_task == MAX_SOURCES_PER_TASK_CAP

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="")

    def test_rejects_overlong_question(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="x" * 3000)
