"""Researcher agent node.

For each ``ResearchTask`` produced by the Planner:
    1. Run a Serper web search.
    2. Ask the LLM to extract a small list of relevant evidence items from
       the search snippets, referencing each by 1-based index.
    3. Map the LLM output back to fully-typed :class:`Evidence` objects
       (with the real URL, title, and a coarse source-quality heuristic).

Tasks are searched concurrently for latency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.core.config import settings as default_settings
from app.core.langgraph.state import ResearchState
from app.core.langgraph.tools.web_search import SerperClient, SerperSearchResult
from app.core.llm import get_default_llm
from app.core.prompts import load_prompt
from app.schemas.evidence import Evidence, ResearchTask

logger = logging.getLogger(__name__)


# --- LLM extraction schema --------------------------------------------------


class _ResearcherItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_index: int = Field(..., ge=1, description="1-based index into the input results")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    content: str = Field(..., min_length=1)


class _ResearcherOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[_ResearcherItem] = Field(default_factory=list)


# --- Source quality heuristic ----------------------------------------------

_HIGH_QUALITY_TLDS = (".gov", ".edu", ".int")
_HIGH_QUALITY_DOMAINS = {
    "arxiv.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "ieee.org",
    "acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "nasa.gov",
    "nist.gov",
}
_MEDIUM_QUALITY_DOMAINS = {
    "wikipedia.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "economist.com",
    "ft.com",
    "bloomberg.com",
    "wsj.com",
    "techcrunch.com",
    "github.com",
    "stackoverflow.com",
}
_LOW_QUALITY_HINTS = ("blogspot.", "medium.com", "substack.com", "quora.com")


def _domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _source_quality(url: str) -> float:
    host = _domain(url)
    if not host:
        return 0.3
    if any(host.endswith(tld) for tld in _HIGH_QUALITY_TLDS):
        return 0.9
    if any(host == d or host.endswith("." + d) for d in _HIGH_QUALITY_DOMAINS):
        return 0.85
    if any(host == d or host.endswith("." + d) for d in _MEDIUM_QUALITY_DOMAINS):
        return 0.65
    if any(hint in host for hint in _LOW_QUALITY_HINTS):
        return 0.35
    return 0.5


# --- Per-task pipeline ------------------------------------------------------


def _format_results_for_llm(results: list[SerperSearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] title: {r.title}\n    url: {r.link}\n    snippet: {r.snippet or '(no snippet)'}"
        )
    return "\n".join(lines)


async def _research_one_task(
    task: ResearchTask,
    *,
    search_client: SerperClient,
    llm: BaseChatModel,
    system_prompt: str,
    max_sources: int,
    min_relevance: float,
) -> list[Evidence]:
    # 1. Search.
    results = await search_client.search(task.question, num=max(max_sources * 2, 5))
    if not results:
        logger.info("Researcher: no search results for task %s", task.id)
        return []

    # 2. Ask LLM to pick + summarize relevant ones.
    structured = llm.with_structured_output(_ResearcherOutput)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Sub-question:\n{task.question}\n\n"
                f"Search results:\n{_format_results_for_llm(results)}\n\n"
                f"Select up to {max_sources} results and produce evidence items."
            )
        ),
    ]
    try:
        output: _ResearcherOutput = await structured.ainvoke(messages)
    except Exception as exc:
        logger.warning("Researcher LLM call failed for task %s: %s", task.id, exc)
        return []

    # 3. Map back to typed Evidence, filtering by relevance + bounds.
    evidence: list[Evidence] = []
    for item in output.items:
        idx = item.source_index - 1
        if not (0 <= idx < len(results)):
            continue
        if item.relevance_score < min_relevance:
            continue
        result = results[idx]
        try:
            evidence.append(
                Evidence(
                    task_id=task.id,
                    source_url=str(result.link),
                    title=result.title,
                    snippet=result.snippet,
                    content=item.content.strip(),
                    relevance_score=item.relevance_score,
                    source_quality=_source_quality(str(result.link)),
                )
            )
        except Exception as exc:
            logger.debug("Skipping invalid evidence for task %s: %s", task.id, exc)
            continue

    # Cap and sort by relevance.
    evidence.sort(key=lambda e: e.relevance_score, reverse=True)
    return evidence[:max_sources]


# --- Node factory -----------------------------------------------------------


def make_researcher_node(
    llm: BaseChatModel | None = None,
    *,
    search_client: SerperClient | None = None,
    settings: Settings | None = None,
):
    """Build the researcher node with injected dependencies.

    The injected ``search_client`` is reused across tasks (do not close it
    between invocations). If none is given, a new :class:`SerperClient` is
    created per call (slower; intended for ad-hoc use).
    """
    cfg = settings or default_settings
    system_prompt = load_prompt("researcher")

    async def researcher_node(state: ResearchState) -> dict[str, Any]:
        tasks: list[ResearchTask] = state.get("research_tasks") or []
        if not tasks:
            return {
                "status": "error",
                "error": "Researcher invoked with no tasks.",
                "evidence": [],
            }

        max_sources = int(state.get("max_sources_per_task") or cfg.max_sources_per_task)
        min_relevance = cfg.min_source_relevance
        chat = llm or get_default_llm()

        owned_client = False
        client = search_client
        if client is None:
            client = SerperClient()
            owned_client = True

        try:
            coros = [
                _research_one_task(
                    task,
                    search_client=client,
                    llm=chat,
                    system_prompt=system_prompt,
                    max_sources=max_sources,
                    min_relevance=min_relevance,
                )
                for task in tasks
            ]
            results: list[list[Evidence]] = await asyncio.gather(*coros, return_exceptions=False)
        finally:
            if owned_client:
                await client.aclose()

        all_evidence: list[Evidence] = [e for sub in results for e in sub]
        logger.info(
            "Researcher collected %d evidence item(s) across %d task(s)",
            len(all_evidence),
            len(tasks),
        )

        return {
            "evidence": all_evidence,
            "status": "fact_checking",
        }

    return researcher_node


researcher_node = make_researcher_node()
