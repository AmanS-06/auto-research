# LOCKED BY Worker B

"""Research service.

Owns the orchestration between the LangGraph state machine and the
persistence layer (SQLModel). It is the single place where the contract
between the graph (``app.core.langgraph``) and the database
(``app.models.research``) is enforced.

Graph output contract (see :class:`app.core.langgraph.state.ResearchState`):

    final_state = {
        "status": "complete" | "error",
        "error": str | None,
        "research_tasks": list[ResearchTask],
        "evidence": list[Evidence],
        "verified_evidence": list[Evidence],
        "report_markdown": str,
        "summary": str,
        "citations": list[Citation],
    }

Database report shape (see :class:`app.models.research.ResearchReport`):

    ResearchReport(
        job_id=job_id,
        question=request.question,
        report=final_state["report_markdown"],
        citations=final_state["citations"],
        extra_metadata={...},
    )

Job lifecycle in the DB:

    pending  -> running  -> complete
                          \\-> failed

The graph's internal phases (``planning`` / ``researching`` /
``fact_checking`` / ``writing``) are *not* persisted as the job status.
They are reflected in ``job.progress`` (a human-readable string) and the
final graph state is serialized into ``job.state`` once the run finishes.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.config import settings
from app.core.langgraph.graph import build_research_graph
from app.core.langgraph.state import ResearchState
from app.core.langgraph.tools.web_search import SerperClient
from app.core.llm import LLMFactory
from app.models.research import ResearchJob, ResearchReport
from app.schemas.evidence import Citation
from app.schemas.research import ResearchRequest
from app.services.checkpoint import get_checkpointer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert a value to JSON-safe primitives.

    Pydantic ``BaseModel`` instances become dicts (``model_dump(mode="json")``);
    nested lists / dicts are walked; everything else is returned as-is.

    This is what the persistence layer (PostgreSQL ``JSONB`` columns on
    ``ResearchJob.state`` / ``ResearchReport.citations`` /
    ``ResearchReport.extra_metadata``) needs: SQLAlchemy does not
    auto-serialize Pydantic models or ``HttpUrl`` instances, and the
    test suite is allowed to use a SQLite in-memory engine that is even
    pickier about JSON column payloads.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def extract_report_payload(
    final_state: dict[str, Any] | ResearchState,
    *,
    json_safe: bool = True,
) -> dict[str, Any]:
    """Pull the report-relevant fields out of a final graph state.

    Worker B's graph returns a dict that conforms to ``ResearchState``;
    callers (this service, the API) should never reach into it with
    ad-hoc key names. Use this helper instead.

    Args:
        final_state: The dict returned by ``graph.ainvoke(...)``.
        json_safe: When True (the default), Pydantic models inside the
            payload are recursively ``model_dump(mode="json")``-ed so the
            result round-trips through ``json.dumps`` and the SQLAlchemy
            ``JSON``/``JSONB`` columns. Set False in tests that want to
            inspect typed objects.
    """
    raw: dict[str, Any] = {
        "status": final_state.get("status", "error"),
        "error": final_state.get("error"),
        "report_markdown": final_state.get("report_markdown", ""),
        "summary": final_state.get("summary", ""),
        "citations": list(final_state.get("citations") or []),
        "research_tasks": list(final_state.get("research_tasks") or []),
        "evidence": list(final_state.get("evidence") or []),
        "verified_evidence": list(final_state.get("verified_evidence") or []),
    }
    if not json_safe:
        return raw
    return _to_jsonable(raw)


def assert_json_safe(payload: Any) -> None:
    """Raise ``TypeError`` if ``payload`` is not JSON-serializable.

    Used by tests to lock the contract that everything we hand to a
    JSON column (job.state, citations, extra_metadata) can be round-
    tripped through :func:`json.dumps`.
    """
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ResearchService:
    """High-level pipeline orchestration.

    The service is *stateless across calls* aside from a lazily-built graph
    and search client. New instances are cheap; tests should create one per
    test.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.graph = None
        self._search_client: SerperClient | None = None

    # -- Lazy dependencies -------------------------------------------------

    async def _get_search_client(self) -> SerperClient:
        if self._search_client is None:
            self._search_client = SerperClient(settings=settings)
        return self._search_client

    async def _get_graph(self):
        if self.graph is None:
            checkpointer = await get_checkpointer()
            search_client = await self._get_search_client()
            llm = LLMFactory.get_llm()
            self.graph = build_research_graph(
                checkpointer=checkpointer,
                search_client=search_client,
                settings=settings,
                llm=llm,
            )
        return self.graph

    # -- Public API --------------------------------------------------------

    async def execute(self, job_id: UUID, request: ResearchRequest) -> None:
        """Run the research pipeline for a persisted job.

        This method never raises: any error during graph execution is
        caught and persisted on the job row as ``status="failed"`` with
        ``error`` set. The caller (``run_research_pipeline``) is therefore
        safe to use as a FastAPI BackgroundTask.

        Progress is streamed: after each LangGraph node completes, the
        service updates ``job.progress`` so polling clients see the
        current phase (``Planning...`` -> ``Researching...`` ->
        ``Fact-checking...`` -> ``Writing...`` -> ``Complete``). This is
        driven by ``astream(stream_mode="values")``; if the graph does
        not expose ``astream`` (e.g. some test fakes), we fall back to
        a single ``ainvoke`` and emit a coarse progress.
        """
        job = await self.session.get(ResearchJob, job_id)
        if not job:
            logger.error("Job %s not found", job_id)
            return

        try:
            await self._mark_running(job, "Initializing research pipeline...")
            graph = await self._get_graph()
        except Exception as exc:
            await self._mark_failed(job, exc, "Research failed during setup")
            return

        initial_state: ResearchState = self._build_initial_state(request)
        config = {"configurable": {"thread_id": str(job_id)}}

        try:
            final_state = await self._stream_with_progress(graph, initial_state, config, job)
        except Exception as exc:
            logger.exception("Graph execution crashed for job %s", job_id)
            await self._mark_failed(job, exc, "Research failed during execution")
            return

        payload = extract_report_payload(final_state)

        if payload["status"] == "error" or payload["error"]:
            error_message = payload["error"] or "Research failed"
            job.status = "failed"
            job.error = error_message
            job.progress = "Research failed"
            job.state = self._serialize_state(final_state)
            await self.session.commit()
            logger.info("Job %s finished with error: %s", job_id, error_message)
            return

        await self._persist_success(job, request, payload, final_state)
        logger.info("Job %s completed successfully", job_id)

    # -- Streaming ---------------------------------------------------------

    PROGRESS_MESSAGES: dict[str, str] = {
        "planner": "Planning research tasks...",
        "researcher": "Researching the web...",
        "fact_checker": "Fact-checking sources...",
        "writer": "Writing report...",
    }

    async def _stream_with_progress(
        self,
        graph: Any,
        initial_state: ResearchState,
        config: dict[str, Any],
        job: ResearchJob,
    ) -> dict[str, Any]:
        """Run the graph, updating ``job.progress`` after each node.

        Returns the final state (last event when ``stream_mode="values"``,
        or the single ``ainvoke`` result when streaming is unavailable).
        """
        astream = getattr(graph, "astream", None)
        if astream is None:
            # Fake graph in tests — single progress update.
            await self._update_progress(job, "Running research pipeline...")
            return await graph.ainvoke(initial_state, config=config)

        last_state: dict[str, Any] = dict(initial_state)
        seen: set[str] = set()
        try:
            async for state in astream(initial_state, config=config, stream_mode="values"):
                last_state = state
                # ``stream_mode="values"`` yields the *full* state after
                # each node. We don't get the node name in this mode, so
                # we advance progress based on which keys have appeared.
                newly_completed: list[str] = []
                if "research_tasks" in state and state.get("research_tasks"):
                    newly_completed.append("planner")
                if "evidence" in state and state.get("evidence"):
                    newly_completed.append("researcher")
                if "verified_evidence" in state and state.get("verified_evidence") is not None:
                    newly_completed.append("fact_checker")
                if state.get("report_markdown") or state.get("citations") is not None:
                    newly_completed.append("writer")
                for name in newly_completed:
                    if name in seen:
                        continue
                    seen.add(name)
                    msg = self.PROGRESS_MESSAGES.get(name)
                    if msg:
                        await self._update_progress(job, msg)
        except TypeError:
            # Older LangGraph versions don't accept ``stream_mode``.
            # Fall back to plain astream (updates mode) — the event
            # *keys* are node names, which gives us even better progress.
            seen.clear()
            async for event in astream(initial_state, config=config):
                if not isinstance(event, dict):
                    continue
                for node_name in event.keys():
                    if node_name in seen:
                        continue
                    seen.add(node_name)
                    msg = self.PROGRESS_MESSAGES.get(node_name)
                    if msg:
                        await self._update_progress(job, msg)
            return await graph.ainvoke(initial_state, config=config)
        return last_state

    async def get_job_status(self, job_id: UUID) -> ResearchJob | None:
        return await self.session.get(ResearchJob, job_id)

    async def get_report(self, job_id: UUID) -> ResearchReport | None:
        result = await self.session.exec(
            select(ResearchReport).where(ResearchReport.job_id == job_id)
        )
        return result.first()

    # -- Internals ---------------------------------------------------------

    def _build_initial_state(self, request: ResearchRequest) -> ResearchState:
        """Build the graph's initial state from a user request.

        Mirrors :class:`app.core.langgraph.state.ResearchState`; we only
        set keys the graph actually reads. ``report_markdown`` is
        intentionally *not* seeded here — the writer fills it in.
        """
        return {
            "question": request.question,
            "max_tasks": request.max_tasks,
            "max_sources_per_task": request.max_sources_per_task,
            "research_tasks": [],
            "evidence": [],
            "verified_evidence": [],
            "citations": [],
            "status": "planning",
            "error": None,
        }

    async def _mark_running(self, job: ResearchJob, progress: str) -> None:
        job.status = "running"
        job.progress = progress
        job.error = None
        await self.session.commit()

    async def _update_progress(self, job: ResearchJob, progress: str) -> None:
        job.progress = progress
        await self.session.commit()

    async def _mark_failed(
        self, job: ResearchJob, exc: BaseException, progress_message: str
    ) -> None:
        logger.exception("Marking job %s as failed", job.id)
        job.status = "failed"
        job.error = str(exc) if exc else "Unknown error"
        job.progress = progress_message
        await self.session.commit()

    async def _persist_success(
        self,
        job: ResearchJob,
        request: ResearchRequest,
        payload: dict[str, Any],
        final_state: dict[str, Any],
    ) -> None:
        job.status = "complete"
        job.progress = "Research complete"
        job.error = None
        job.state = self._serialize_state(final_state)

        citations: list[Citation] = payload["citations"]
        report = ResearchReport(
            job_id=job.id,
            question=request.question,
            report=payload["report_markdown"],
            citations=citations,
            extra_metadata={
                "summary": payload["summary"],
                "tasks_completed": len(payload["research_tasks"]),
                "sources_found": len(payload["evidence"]),
                "verified_sources": len(payload["verified_evidence"]),
                "citation_count": len(citations),
            },
        )
        self.session.add(report)
        await self.session.commit()

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        """Recursively convert Pydantic models to JSON-safe primitives.

        Alias for :func:`_to_jsonable`. Kept as a static method for
        backwards-compat with existing tests / external callers.
        """
        return _to_jsonable(value)

    @staticmethod
    def _serialize_state(final_state: dict[str, Any]) -> dict[str, Any]:
        """Reduce a graph final state to JSON-safe primitives for the DB.

        We can't store Pydantic / ``HttpUrl`` objects in the JSON column
        without surprises; flatten them out and keep only the keys the
        API exposes.
        """
        return _to_jsonable(extract_report_payload(final_state))
