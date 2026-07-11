"""Tests for the ResearchService (Worker C).

These tests target the service layer in isolation. They use a fake graph
and a fake search client (no LangGraph, no Serper) so the tests stay
deterministic and fast. The point is to lock the contract between the
service and the persistence layer / graph output.

Coverage:
    * Successful job execution writes a complete Report with the right
      field names and metadata.
    * A graph ``status == "error"`` is persisted as a failed job.
    * A setup exception (graph or search client can't be built) marks
      the job as failed instead of leaving it pending/running.
    * A graph execution exception (raised by ``ainvoke``) marks the job
      as failed.
    * ``extract_report_payload`` normalizes missing keys and surfaces
      ``report_markdown`` (not ``report``).
    * The service reads ``report_markdown`` from the final state (not
      ``report``) so the bug from the v1 service is regression-protected.
    * When the graph reports no verified evidence, the writer's fallback
      ``report_markdown`` is still persisted verbatim.
    * Calling ``execute`` for an unknown job_id is a no-op (logged error).
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.langgraph.tools.web_search import SerperSearchResult
from app.models.research import ResearchJob, ResearchReport
from app.schemas.evidence import Citation, Evidence, ResearchTask
from app.schemas.research import ResearchRequest
from app.services.research_service import (
    ResearchService,
    assert_json_safe,
    extract_report_payload,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSearchClient:
    """Stand-in for SerperClient — only what the service needs."""

    def __init__(self) -> None:
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True

    async def search(self, *args: Any, **kwargs: Any) -> list[SerperSearchResult]:
        return []


class FakeGraph:
    """Minimal graph that returns a pre-canned final state.

    Supports both ``ainvoke`` (the legacy interface) and ``astream``
    (the new values-mode interface that drives progress reporting).
    The two share the same ``_final_state`` so tests don't have to
    duplicate fixtures.
    """

    def __init__(
        self,
        final_state: dict[str, Any] | Exception,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._final_state = final_state
        self.invocations: list[dict[str, Any]] = []
        # Optional explicit astream sequence. If not provided, astream
        # yields a single values-mode event carrying the final state.
        self._stream_events = stream_events
        self.stream_invocations: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
        self.invocations.append({"state": state, "config": config})
        if isinstance(self._final_state, Exception):
            raise self._final_state
        return self._final_state

    async def astream(
        self,
        state: dict[str, Any],
        config: dict | None = None,
        stream_mode: str = "values",
        **_kwargs: Any,
    ):
        self.stream_invocations.append({"state": state, "config": config, "mode": stream_mode})
        if isinstance(self._final_state, Exception):
            raise self._final_state
        if self._stream_events is not None:
            for ev in self._stream_events:
                yield ev
            return
        # Default: yield a single values-mode event with the final state.
        yield dict(self._final_state)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_request() -> ResearchRequest:
    return ResearchRequest(
        question="What is quantum computing?",
        max_tasks=3,
        max_sources_per_task=2,
    )


def _make_job(status: str = "pending") -> ResearchJob:
    return ResearchJob(
        id=uuid.uuid4(),
        question="What is quantum computing?",
        max_tasks=3,
        max_sources_per_task=2,
        status=status,
    )


def _make_session_with_job(job: ResearchJob) -> AsyncMock:
    """A mock AsyncSession that returns ``job`` from ``session.get``."""
    session = AsyncMock()
    session.get = AsyncMock(return_value=job)
    session.commit = AsyncMock(return_value=None)
    session.add = MagicMock()
    return session


def _attach_report_to_session(session: AsyncMock, report: ResearchReport) -> None:
    """Make ``session.exec(...).first()`` return ``report``."""
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=report)
    session.exec = AsyncMock(return_value=exec_result)


# ---------------------------------------------------------------------------
# extract_report_payload
# ---------------------------------------------------------------------------


class TestExtractReportPayload:
    def test_returns_expected_keys(self) -> None:
        state: dict[str, Any] = {
            "status": "complete",
            "error": None,
            "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            "evidence": [],
            "verified_evidence": [],
            "report_markdown": "# hello",
            "summary": "hi",
            "citations": [],
        }
        payload = extract_report_payload(state)
        assert payload["status"] == "complete"
        assert payload["error"] is None
        assert payload["report_markdown"] == "# hello"
        assert payload["summary"] == "hi"
        assert isinstance(payload["citations"], list)

    def test_defaults_when_keys_missing(self) -> None:
        payload = extract_report_payload({})
        assert payload["status"] == "error"
        assert payload["error"] is None
        assert payload["report_markdown"] == ""
        assert payload["summary"] == ""
        assert payload["citations"] == []
        assert payload["research_tasks"] == []

    def test_does_not_expose_misleading_report_key(self) -> None:
        """The graph returns ``report_markdown``; there is no ``report`` key.

        This guards against the v1 service bug where ``final_state["report"]``
        was read. If anyone reintroduces a ``report`` key in the graph,
        this helper will silently drop it — that's the desired behavior.
        """
        state = {"report": "stale", "report_markdown": "fresh"}
        payload = extract_report_payload(state)
        assert payload["report_markdown"] == "fresh"
        assert "report" not in payload


# ---------------------------------------------------------------------------
# execute — happy path
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    @pytest.mark.asyncio
    async def test_writes_complete_report_and_metadata(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)

        citations = [
            Citation(id="1", source_url="https://a.example", title="A"),
            Citation(id="2", source_url="https://b.example", title="B"),
        ]
        evidence = [
            Evidence(
                task_id="t1",
                source_url="https://a.example",
                title="A",
                content="body",
                relevance_score=0.9,
                source_quality=0.7,
            ),
        ]
        verified = evidence
        tasks = [
            ResearchTask(id="t1", question="what is X?"),
            ResearchTask(id="t2", question="how is X measured?"),
        ]
        final_state = {
            "status": "complete",
            "error": None,
            "research_tasks": tasks,
            "evidence": evidence,
            "verified_evidence": verified,
            "report_markdown": "# Final Report",
            "summary": "Two sentences.",
            "citations": citations,
        }
        graph = FakeGraph(final_state)

        service = ResearchService(session)
        service.graph = graph  # bypass lazy init
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        # Job transitioned pending -> running -> complete.
        assert job.status == "complete"
        assert job.error is None
        assert "complete" in job.progress.lower()

        # State was serialized and is JSON-safe (P0 fix: was returning
        # Pydantic objects, which SQLAlchemy JSON columns cannot store).
        assert job.state["report_markdown"] == "# Final Report"
        # The report was added and committed.
        session.add.assert_called_once()
        report = session.add.call_args.args[0]
        assert isinstance(report, ResearchReport)
        assert report.job_id == job.id
        assert report.question == sample_request.question
        assert report.report == "# Final Report"
        # Citations are now JSON-safe dicts (was: Pydantic Citation objects).
        assert isinstance(report.citations, list)
        assert len(report.citations) == 2
        assert all(isinstance(c, dict) for c in report.citations)
        assert {c["id"] for c in report.citations} == {"1", "2"}
        # Metadata uses the SQLModel field name, not ``metadata``.
        assert "tasks_completed" in report.extra_metadata
        # json.dumps must succeed against the persisted state and metadata.
        import json as _json

        _json.dumps(job.state)
        _json.dumps(report.extra_metadata)
        assert report.extra_metadata["tasks_completed"] == 2
        assert report.extra_metadata["sources_found"] == 1
        assert report.extra_metadata["verified_sources"] == 1
        assert report.extra_metadata["citation_count"] == 2
        assert report.extra_metadata["summary"] == "Two sentences."

    @pytest.mark.asyncio
    async def test_passes_request_limits_to_graph_initial_state(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        graph = FakeGraph(
            {
                "status": "complete",
                "report_markdown": "x",
                "summary": "",
                "citations": [],
            }
        )
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        # Graph was streamed once with the request's limits.
        assert len(graph.stream_invocations) == 1
        initial_state = graph.stream_invocations[0]["state"]
        assert initial_state["max_tasks"] == 3
        assert initial_state["max_sources_per_task"] == 2
        assert initial_state["question"] == sample_request.question
        # Initial state must NOT include the misleading ``report`` key.
        assert "report" not in initial_state
        assert "job_id" not in initial_state
        assert "current_task_index" not in initial_state

    @pytest.mark.asyncio
    async def test_uses_canonical_thread_id(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        graph = FakeGraph({"status": "complete", "report_markdown": "x", "citations": []})
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        config = graph.stream_invocations[0]["config"]
        assert config == {"configurable": {"thread_id": str(job.id)}}


# ---------------------------------------------------------------------------
# execute — graph returns an error state
# ---------------------------------------------------------------------------


class TestExecuteGraphErrorState:
    @pytest.mark.asyncio
    async def test_persists_failed_status_with_error_message(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        final_state = {
            "status": "error",
            "error": "Planner failed: LLM timeout",
            "report_markdown": "",
            "summary": "",
            "citations": [],
            "research_tasks": [],
            "evidence": [],
            "verified_evidence": [],
        }
        graph = FakeGraph(final_state)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        assert job.status == "failed"
        assert job.error == "Planner failed: LLM timeout"
        assert "failed" in job.progress.lower()
        # No report was written.
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_persists_failed_status_when_error_text_present(self, sample_request):
        """Even if ``status`` is not literally 'error', an ``error`` message
        is enough to mark the job as failed."""
        job = _make_job()
        session = _make_session_with_job(job)
        final_state = {
            "status": "writing",
            "error": "Writer failed: exception",
            "report_markdown": "",
            "summary": "",
            "citations": [],
        }
        graph = FakeGraph(final_state)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        assert job.status == "failed"
        assert "Writer failed" in job.error


# ---------------------------------------------------------------------------
# execute — exception paths
# ---------------------------------------------------------------------------


class TestExecuteExceptions:
    @pytest.mark.asyncio
    async def test_setup_exception_marks_job_failed(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        service = ResearchService(session)
        # Force _get_graph to raise.
        service._get_graph = AsyncMock(side_effect=RuntimeError("graph build boom"))

        await service.execute(job.id, sample_request)

        assert job.status == "failed"
        assert "graph build boom" in job.error
        assert "setup" in job.progress.lower() or "failed" in job.progress.lower()
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_ainvoke_exception_marks_job_failed(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        graph = FakeGraph(RuntimeError("graph crashed mid-run"))
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        assert job.status == "failed"
        assert "graph crashed mid-run" in job.error
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_job_is_silent_noop(self, sample_request):
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        service = ResearchService(session)
        # Should NOT raise.
        await service.execute(uuid.uuid4(), sample_request)
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_never_propagates(self, sample_request):
        """The service contract: ``execute`` swallows all exceptions."""
        job = _make_job()
        session = _make_session_with_job(job)
        service = ResearchService(session)
        service._get_graph = AsyncMock(side_effect=RuntimeError("boom"))

        # No assertion error: execute returns None regardless.
        result = await service.execute(job.id, sample_request)
        assert result is None


# ---------------------------------------------------------------------------
# execute — fallback / empty paths
# ---------------------------------------------------------------------------


class TestExecuteFallbacks:
    @pytest.mark.asyncio
    async def test_empty_evidence_fallback_is_persisted(self, sample_request):
        """When the writer emits the insufficient-evidence fallback report,
        the service must still persist it as a complete job (not failed)."""
        job = _make_job()
        session = _make_session_with_job(job)
        fallback_markdown = "# What is quantum computing?\n\n## Summary\nInsufficient evidence.\n\n## Sources\n_None_\n"
        final_state = {
            "status": "complete",
            "error": None,
            "report_markdown": fallback_markdown,
            "summary": "Insufficient evidence.",
            "citations": [],
            "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            "evidence": [],
            "verified_evidence": [],
        }
        graph = FakeGraph(final_state)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        assert job.status == "complete"
        session.add.assert_called_once()
        report = session.add.call_args.args[0]
        assert report.report == fallback_markdown
        assert report.citations == []
        assert report.extra_metadata["verified_sources"] == 0
        assert report.extra_metadata["citation_count"] == 0

    @pytest.mark.asyncio
    async def test_writes_report_field_not_extra_report_key(self, sample_request):
        """Regression guard for the v1 bug: service wrote ``metadata=`` instead
        of ``extra_metadata=``, and read ``report`` instead of
        ``report_markdown``. The persisted report's ``report`` column must
        hold the markdown text — not the citations list, not the metadata."""
        job = _make_job()
        session = _make_session_with_job(job)
        final_state = {
            "status": "complete",
            "report_markdown": "## Body",
            "summary": "S",
            "citations": [Citation(id="1", source_url="https://x", title="X")],
            "research_tasks": [],
            "evidence": [],
            "verified_evidence": [],
        }
        graph = FakeGraph(final_state)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        report = session.add.call_args.args[0]
        assert report.report == "## Body"  # NOT the citations
        assert report.extra_metadata["summary"] == "S"
        # No ``metadata`` attribute on the SQLModel.
        assert not hasattr(report, "metadata") or "extra_metadata" in report.model_fields_set


# ---------------------------------------------------------------------------
# Progress streaming (P1)
# ---------------------------------------------------------------------------


class FakeStreamingGraph:
    """Yields a deterministic sequence of values-mode events."""

    def __init__(self, events: list[dict[str, Any]], final: dict[str, Any]) -> None:
        self._events = events
        self._final = final
        self.stream_invocations: list[dict[str, Any]] = []
        self.invoke_invocations: list[dict[str, Any]] = []

    async def astream(
        self,
        state: dict[str, Any],
        config: dict | None = None,
        stream_mode: str = "values",
        **_kwargs: Any,
    ):
        self.stream_invocations.append({"state": state, "mode": stream_mode})
        for ev in self._events:
            yield ev
        yield dict(self._final)

    async def ainvoke(self, state: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
        self.invoke_invocations.append({"state": state})
        return dict(self._final)


class TestProgressStreaming:
    @pytest.mark.asyncio
    async def test_progress_advances_through_phases(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        # Three intermediate events + the final state, mimicking the
        # values-mode stream of a real LangGraph run.
        events = [
            {
                "status": "researching",
                "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            },
            {
                "status": "fact_checking",
                "evidence": [
                    Evidence(
                        task_id="t1",
                        source_url="https://e.com/a",
                        title="A",
                        content="c",
                        relevance_score=0.8,
                        source_quality=0.7,
                    )
                ],
            },
            {"status": "writing", "verified_evidence": []},
        ]
        final = {
            "status": "complete",
            "report_markdown": "# R",
            "summary": "s",
            "citations": [],
            "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            "evidence": [],
            "verified_evidence": [],
        }
        graph = FakeStreamingGraph(events, final)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        # astream was called once with values mode.
        assert len(graph.stream_invocations) == 1
        assert graph.stream_invocations[0]["mode"] == "values"
        # ainvoke was NOT called — the final state came from astream.
        assert graph.invoke_invocations == []

        # The job's progress string was updated as phases advanced.
        # We can't observe the in-between writes (they hit the mock
        # session), but the *final* progress message is "Research complete".
        assert job.status == "complete"
        assert "complete" in job.progress.lower()

    @pytest.mark.asyncio
    async def test_progress_falls_back_when_graph_lacks_astream(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)

        # Build a graph that ONLY has ainvoke (mimics older fakes).
        class _NoAstreamGraph:
            def __init__(self) -> None:
                self.invoke_calls = 0

            async def ainvoke(self, state, config=None):
                self.invoke_calls += 1
                return {
                    "status": "complete",
                    "report_markdown": "ok",
                    "summary": "",
                    "citations": [],
                }

        graph = _NoAstreamGraph()
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        assert graph.invoke_calls == 1
        assert job.status == "complete"

    @pytest.mark.asyncio
    async def test_progress_messages_table(self):
        """Lock the phase-name -> human-readable string mapping."""
        assert ResearchService.PROGRESS_MESSAGES == {
            "planner": "Planning research tasks...",
            "researcher": "Researching the web...",
            "fact_checker": "Fact-checking sources...",
            "writer": "Writing report...",
        }


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


class TestReads:
    @pytest.mark.asyncio
    async def test_get_job_status_delegates_to_session(self):
        job = _make_job()
        session = AsyncMock()
        session.get = AsyncMock(return_value=job)
        service = ResearchService(session)
        result = await service.get_job_status(job.id)
        assert result is job
        session.get.assert_awaited_once_with(ResearchJob, job.id)

    @pytest.mark.asyncio
    async def test_get_job_status_returns_none_for_missing(self):
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        service = ResearchService(session)
        result = await service.get_job_status(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_report_returns_first_match(self):
        job = _make_job()
        report = ResearchReport(
            job_id=job.id,
            question=job.question,
            report="x",
            citations=[],
        )
        session = AsyncMock()
        exec_result = MagicMock()
        exec_result.first = MagicMock(return_value=report)
        session.exec = AsyncMock(return_value=exec_result)

        service = ResearchService(session)
        result = await service.get_report(job.id)
        assert result is report


# ---------------------------------------------------------------------------
# Lazy dependency construction
# ---------------------------------------------------------------------------


class TestLazyDependencies:
    @pytest.mark.asyncio
    async def test_get_search_client_uses_settings_kwarg(self):
        """Regression guard for the v1 bug: the service constructed
        ``SerperClient(api_key=...)`` and crashed at runtime. It must
        use the real constructor pattern (``SerperClient(settings=...)``)."""
        session = AsyncMock()
        service = ResearchService(session)
        with patch("app.services.research_service.SerperClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            client = await service._get_search_client()
            assert client is mock_instance
            # Called with the settings kwarg, not api_key.
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert "settings" in call_kwargs
            assert "api_key" not in call_kwargs
            # Cached on the second call.
            await service._get_search_client()
            assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_get_graph_uses_injected_clients(self, sample_request):
        session = AsyncMock()
        service = ResearchService(session)
        service._search_client = FakeSearchClient()

        with (
            patch("app.services.research_service.LLMFactory") as mock_factory,
            patch(
                "app.services.research_service.get_checkpointer", new=AsyncMock(return_value=None)
            ),
            patch("app.services.research_service.build_research_graph") as mock_build,
        ):
            mock_factory.get_llm.return_value = MagicMock(name="fake-llm")
            mock_build.return_value = MagicMock(name="fake-graph")

            graph = await service._get_graph()
            assert graph is mock_build.return_value

            # build_research_graph was called with the expected kwargs.
            mock_build.assert_called_once()
            kwargs = mock_build.call_args.kwargs
            assert kwargs["settings"] is not None
            assert kwargs["llm"] is mock_factory.get_llm.return_value
            assert kwargs["search_client"] is service._search_client


class TestJsonSerialization:
    def test_to_json_safe_converts_citation(self):
        from app.schemas.evidence import Citation
        from app.services.research_service import ResearchService

        citation = Citation(
            id="1",
            source_url="https://example.com/article",
            title="Example",
            snippet="Some snippet",
        )
        result = ResearchService._to_json_safe(citation)
        assert result == {
            "id": "1",
            "source_url": "https://example.com/article",
            "title": "Example",
            "snippet": "Some snippet",
            "task_id": None,
        }

    def test_to_json_safe_converts_list_of_citations(self):
        from app.schemas.evidence import Citation
        from app.services.research_service import ResearchService

        citations = [
            Citation(id="1", source_url="https://a.com", title="A"),
            Citation(id="2", source_url="https://b.com", title="B"),
        ]
        result = ResearchService._to_json_safe(citations)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_to_json_safe_passes_plain_types(self):
        from app.services.research_service import ResearchService

        assert ResearchService._to_json_safe("hello") == "hello"
        assert ResearchService._to_json_safe(42) == 42
        assert ResearchService._to_json_safe(None) is None
        assert ResearchService._to_json_safe([1, 2]) == [1, 2]
        assert ResearchService._to_json_safe({"a": 1}) == {"a": 1}

    def test_to_json_safe_converts_nested_dict(self):
        from app.schemas.evidence import Citation
        from app.services.research_service import ResearchService

        data = {
            "citations": [
                Citation(id="1", source_url="https://x.com", title="X"),
            ],
            "count": 1,
        }
        result = ResearchService._to_json_safe(data)
        assert isinstance(result["citations"][0], dict)
        assert result["citations"][0]["id"] == "1"
        assert result["count"] == 1


class TestJsonRoundTrip:
    """P0 regression tests: nothing we persist may crash ``json.dumps``.

    The SQLAlchemy ``JSON`` / PostgreSQL ``JSONB`` columns on
    ``ResearchJob.state``, ``ResearchReport.citations``, and
    ``ResearchReport.extra_metadata`` accept only JSON-safe primitives.
    A Pydantic ``BaseModel`` or an ``HttpUrl`` slips through Python type
    checks but blows up at INSERT time (or, worse, silently round-trips
    incorrectly on a SQLite test engine).
    """

    def test_extract_report_payload_is_json_safe_with_typed_objects(self):
        state = {
            "status": "complete",
            "error": None,
            "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            "evidence": [
                Evidence(
                    task_id="t1",
                    source_url="https://example.com/a",
                    title="A",
                    content="body",
                    relevance_score=0.9,
                    source_quality=0.7,
                ),
            ],
            "verified_evidence": [],
            "report_markdown": "# X",
            "summary": "s",
            "citations": [Citation(id="1", source_url="https://example.com", title="A")],
        }
        payload = extract_report_payload(state)
        # Must round-trip through json.dumps without raising.
        serialized = assert_json_safe(payload)
        assert serialized is None  # returns None on success

    def test_extract_report_payload_can_disable_typing(self):
        """``json_safe=False`` preserves typed objects for callers that
        need to introspect Pydantic models (e.g. Writer tests)."""
        state = {
            "status": "complete",
            "citations": [Citation(id="1", source_url="https://x", title="X")],
        }
        payload = extract_report_payload(state, json_safe=False)
        assert isinstance(payload["citations"][0], Citation)

    def test_http_url_becomes_str_in_safe_payload(self):
        state = {
            "citations": [Citation(id="1", source_url="https://example.com/path?x=1", title="t")],
            "status": "complete",
        }
        payload = extract_report_payload(state)
        url = payload["citations"][0]["source_url"]
        assert isinstance(url, str)
        assert url == "https://example.com/path?x=1"

    @pytest.mark.asyncio
    async def test_persisted_state_round_trips_through_json(self, sample_request):
        job = _make_job()
        session = _make_session_with_job(job)
        citation = Citation(id="1", source_url="https://a.example/x", title="A")
        evidence = [
            Evidence(
                task_id="t1",
                source_url="https://a.example/x",
                title="A",
                content="body",
                relevance_score=0.9,
                source_quality=0.7,
            ),
        ]
        final_state = {
            "status": "complete",
            "report_markdown": "# R",
            "summary": "s",
            "citations": [citation],
            "research_tasks": [ResearchTask(id="t1", question="what is X?")],
            "evidence": evidence,
            "verified_evidence": evidence,
        }
        graph = FakeGraph(final_state)
        service = ResearchService(session)
        service.graph = graph
        service._search_client = FakeSearchClient()

        await service.execute(job.id, sample_request)

        # Both persisted blobs must be JSON-serializable end-to-end.
        import json

        json.dumps(job.state)
        report = session.add.call_args.args[0]
        json.dumps(report.citations)
        json.dumps(report.extra_metadata)
