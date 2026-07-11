# LOCKED BY Worker A

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import JSON, Column, Field, SQLModel

from app.schemas.evidence import Citation


class ResearchJob(SQLModel, table=True):
    __tablename__ = "research_jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    question: str
    max_tasks: int = Field(default=5)
    max_sources_per_task: int = Field(default=3)
    status: str = Field(default="pending")
    progress: str | None = None
    error: str | None = None
    state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchReport(SQLModel, table=True):
    __tablename__ = "research_reports"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID = Field(foreign_key="research_jobs.id", index=True)
    question: str
    report: str
    citations: list[Citation] = Field(default_factory=list, sa_column=Column(JSON))
    extra_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
