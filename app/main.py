# LOCKED BY Worker B

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import research, health
from app.core.config import settings
from app.core.database import get_async_engine, init_db
from app.core.middleware import setup_middleware
from app.services.checkpoint import close_checkpointer

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await init_db()
    yield
    logger.info("Shutting down...")
    await close_checkpointer()
    engine = get_async_engine()
    if engine is not None:
        await engine.dispose()


app = FastAPI(
    title="Autonomous Research Pipeline",
    description="Multi-agent research system using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# Set up middleware (CORS + request logging)
setup_middleware(app)

# Include routers
app.include_router(research.router, prefix="/api/v1", tags=["research"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    return {"message": "Autonomous Research Pipeline API", "version": "0.1.0"}
