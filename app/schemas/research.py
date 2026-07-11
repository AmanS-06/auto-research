# LOCKED BY Worker A

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.evidence import Citation

# Hard caps independent of Settings, so the API can never accept a
# value the planner / researcher nodes won't actually honor. These match
# the defaults documented in :class:`app.core.config.Settings`.
MAX_TASKS_CAP: int = 25
MAX_SOURCES_PER_TASK_CAP: int = 10


class ResearchRequest(BaseModel):
    """Public API request payload for ``POST /api/v1/research``.

    Limits are bounded so a single client cannot pin the service with
    an absurd request (e.g. ``max_tasks=10_000`` triggering thousands of
    Serper / LLM calls). The Planner and Researcher nodes also clamp
    internally, but rejecting at the boundary gives clients a clean
    422 response instead of silently clipping their request.
    """

    question: str = Field(..., min_length=1, max_length=2000, description="Research question")
    max_tasks: int = Field(
        5,
        ge=1,
        le=MAX_TASKS_CAP,
        description=(
            "Maximum number of sub-questions the Planner will produce. "
            f"Clamped to [1, {MAX_TASKS_CAP}]."
        ),
    )
    max_sources_per_task: int = Field(
        3,
        ge=1,
        le=MAX_SOURCES_PER_TASK_CAP,
        description=(
            "Maximum number of evidence items per sub-question. "
            f"Clamped to [1, {MAX_SOURCES_PER_TASK_CAP}]."
        ),
    )


class ResearchResponse(BaseModel):
    job_id: UUID
    status: Literal["pending", "running", "complete", "failed"]
    report: str | None = None
    citations: list[Citation] = []
    error: str | None = None


class ResearchJobStatus(BaseModel):
    job_id: UUID
    status: Literal["pending", "running", "complete", "failed"]
    progress: str | None = None
    error: str | None = None


class ResearchJobResponse(BaseModel):
    job_id: UUID
    question: str
    status: Literal["pending", "running", "complete", "failed"]
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ResearchReportResponse(BaseModel):
    job_id: UUID
    status: Literal["pending", "running", "complete", "failed"]
    report: str | None = None
    citations: list[Citation] = []
    error: str | None = None
