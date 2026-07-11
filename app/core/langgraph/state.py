# LOCKED BY Worker B

"""LangGraph state definition.

`ResearchState` is the single source of truth flowing through the graph.
All node functions take a `ResearchState` and return a partial dict that
LangGraph merges using the reducers declared via `Annotated`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.schemas.evidence import Citation, Evidence, ResearchTask

Status = Literal[
    "planning",
    "researching",
    "fact_checking",
    "writing",
    "complete",
    "error",
]


class ResearchState(TypedDict, total=False):
    """State passed between nodes in the research graph."""

    # --- Input ---
    question: str
    max_tasks: int
    max_sources_per_task: int

    # --- Planning ---
    research_tasks: list[ResearchTask]

    # --- Research ---
    # Researcher nodes may run in parallel per task, so we use list-concat
    # as the reducer (this lets multiple branches contribute evidence).
    evidence: Annotated[list[Evidence], operator.add]

    # --- Fact-checking ---
    verified_evidence: list[Evidence]

    # --- Writing ---
    report_markdown: str
    summary: str
    citations: list[Citation]

    # --- Bookkeeping ---
    status: Status
    error: str | None


def extract_report_payload(state: ResearchState) -> dict:
    """Extract the final report payload from a completed graph state.

    This is the stable contract between the graph (Worker B) and the
    service layer (Worker C). The service should call this on the final
    state returned by ``graph.ainvoke()`` to get fields it can persist.

    Args:
        state: The final ``ResearchState`` returned by the compiled graph.

    Returns:
        A dict with keys:
        - ``status``: "complete" | "error"
        - ``error``: str | None
        - ``research_tasks``: list[ResearchTask]
        - ``evidence``: list[Evidence]
        - ``verified_evidence``: list[Evidence]
        - ``report_markdown``: str
        - ``summary``: str
        - ``citations``: list[Citation]
    """
    return {
        "status": state.get("status", "error"),
        "error": state.get("error"),
        "research_tasks": state.get("research_tasks", []),
        "evidence": state.get("evidence", []),
        "verified_evidence": state.get("verified_evidence", []),
        "report_markdown": state.get("report_markdown", ""),
        "summary": state.get("summary", ""),
        "citations": state.get("citations", []),
    }


__all__ = ["ResearchState", "Status", "extract_report_payload"]
