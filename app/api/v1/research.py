# LOCKED BY Worker A

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.database import get_session
from app.models.research import ResearchJob, ResearchReport
from app.schemas.research import ResearchJobStatus, ResearchRequest, ResearchResponse
from app.services.research_service import ResearchService

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/research", response_model=ResearchResponse, status_code=202)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    job = ResearchJob(
        question=request.question,
        max_tasks=request.max_tasks,
        max_sources_per_task=request.max_sources_per_task,
        status="pending",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(run_research_pipeline, job.id, request)

    return ResearchResponse(
        job_id=job.id,
        status="pending",
    )


async def run_research_pipeline(job_id: UUID, request: ResearchRequest):
    from app.core.database import get_async_session_factory

    logger.info("Starting research pipeline for job %s", job_id)

    try:
        factory = await get_async_session_factory()
        async with factory() as session:
            service = ResearchService(session)
            await service.execute(job_id, request)
    except Exception as exc:
        logger.exception("Research pipeline crashed for job %s", job_id)
        try:
            _factory = await get_async_session_factory()
            async with _factory() as _session:
                job = await _session.get(ResearchJob, job_id)
                if job:
                    job.status = "failed"
                    job.error = str(exc)
                    await _session.commit()
                    logger.info("Job %s marked as failed in DB", job_id)
        except Exception as db_exc:
            logger.exception("Could not mark job %s as failed in DB: %s", job_id, db_exc)


@router.get("/research/{job_id}", response_model=ResearchResponse)
async def get_research(job_id: UUID, session: AsyncSession = Depends(get_session)):
    job = await session.get(ResearchJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")

    if job.status == "complete":
        report = await session.exec(select(ResearchReport).where(ResearchReport.job_id == job_id))
        report = report.first()
        if report:
            return ResearchResponse(
                job_id=job.id,
                status=job.status,
                report=report.report,
                citations=list(report.citations or []),
            )
        logger.warning("Job %s is marked complete but no report exists", job_id)
        return ResearchResponse(
            job_id=job.id,
            status="failed",
            error="Job marked complete but report data is missing",
        )

    return ResearchResponse(
        job_id=job.id,
        status=job.status,
        error=job.error,
    )


@router.get("/research/{job_id}/status", response_model=ResearchJobStatus)
async def get_research_status(job_id: UUID, session: AsyncSession = Depends(get_session)):
    job = await session.get(ResearchJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")

    return ResearchJobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
    )
