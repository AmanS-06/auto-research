# LOCKED BY Worker A

"""Shared evidence and citation models.

These models are the contract that Worker B exposes to Worker A (persistence)
and Worker C (API integration). Keep them stable.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ResearchTask(BaseModel):
    """A single sub-question produced by the Planner.

    Each task corresponds to one focused web search + extraction pass.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable identifier, e.g. 't1', 't2', ...")
    question: str = Field(..., min_length=3, description="Focused sub-question to research.")
    rationale: str = Field(
        default="",
        description="Why this task matters for answering the user's question.",
    )

    @field_validator("id")
    @classmethod
    def _strip_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Task id cannot be empty")
        return v


class Evidence(BaseModel):
    """A piece of evidence collected by the Researcher for a given task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Owning ResearchTask.id")
    source_url: HttpUrl
    title: str = Field(..., min_length=1)
    snippet: str = Field(
        default="",
        description="Short summary returned by the search provider.",
    )
    content: str = Field(
        default="",
        description="Extracted/condensed content used by the agents.",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0..1 estimate of how relevant this source is to the task.",
    )
    source_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0..1 heuristic for the trustworthiness of the domain.",
    )

    @property
    def fingerprint(self) -> str:
        """Stable dedup key based on (normalized url, title)."""
        url = str(self.source_url).rstrip("/").lower()
        key = f"{url}|{self.title.strip().lower()}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()


class Citation(BaseModel):
    """A citation rendered in the final report.

    The `id` is what appears in the report body (e.g. ``[1]``) and the
    `source_url` is what the reader will follow.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Display id, typically a 1-based integer as string.")
    source_url: HttpUrl
    title: str
    snippet: str = ""
    task_id: str | None = None


class ResearchReport(BaseModel):
    """Final structured report returned by the Writer agent."""

    model_config = ConfigDict(extra="forbid")

    question: str
    summary: str = Field(..., description="One-paragraph executive summary.")
    body_markdown: str = Field(..., description="Full report rendered as Markdown.")
    citations: list[Citation] = Field(default_factory=list)
    status: Literal["complete", "partial", "error"] = "complete"
