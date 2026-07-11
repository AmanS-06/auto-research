"""Fact Checker agent node.

Responsibilities:
    1. Deduplicate raw evidence by ``(normalized_url, title)`` fingerprint.
    2. Apply a coarse quality floor (drop empty content, drop sub-threshold
       relevance).
    3. Ask the LLM to re-score each remaining item against the user's
       *overall* question (not just the per-task sub-question) and decide
       which items to keep.
    4. Materialize the final ``verified_evidence`` list, sorted by combined
       quality.

The dedup + thresholding step is deterministic so duplicates never reach the
LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.core.config import settings as default_settings
from app.core.langgraph.state import ResearchState
from app.core.llm import get_default_llm
from app.core.prompts import load_prompt
from app.schemas.evidence import Evidence

logger = logging.getLogger(__name__)


class _Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: str = Field(..., min_length=1)
    keep: bool
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    source_quality: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=400)


class _FactCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[_Assessment] = Field(default_factory=list)


def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
    """Keep the highest-relevance item per fingerprint."""
    best: dict[str, Evidence] = {}
    for item in evidence:
        fp = item.fingerprint
        existing = best.get(fp)
        if existing is None or item.relevance_score > existing.relevance_score:
            best[fp] = item
    return list(best.values())


def _format_evidence_for_llm(evidence: list[Evidence]) -> str:
    lines: list[str] = []
    for e in evidence:
        lines.append(
            f"fingerprint: {e.fingerprint}\n"
            f"task_id: {e.task_id}\n"
            f"url: {e.source_url}\n"
            f"title: {e.title}\n"
            f"content: {e.content}\n"
            f"relevance_score (current): {e.relevance_score:.2f}\n"
            f"source_quality (current): {e.source_quality:.2f}\n"
            "---"
        )
    return "\n".join(lines)


def make_fact_checker_node(
    llm: BaseChatModel | None = None,
    *,
    settings: Settings | None = None,
):
    """Build the fact-checker node with injected dependencies."""
    cfg = settings or default_settings
    system_prompt = load_prompt("fact_checker")

    async def fact_checker_node(state: ResearchState) -> dict[str, Any]:
        raw: list[Evidence] = state.get("evidence") or []
        if not raw:
            return {
                "verified_evidence": [],
                "status": "writing",
            }

        # 1. Dedup + relevance floor.
        deduped = _deduplicate(raw)
        floored = [
            e
            for e in deduped
            if e.content.strip() and e.relevance_score >= cfg.min_source_relevance
        ]
        logger.info(
            "Fact-checker: %d raw -> %d deduped -> %d above floor",
            len(raw),
            len(deduped),
            len(floored),
        )

        if not floored:
            return {
                "verified_evidence": [],
                "status": "writing",
            }

        # 2. LLM assessment.
        question = (state.get("question") or "").strip()
        chat = llm or get_default_llm()
        structured = chat.with_structured_output(_FactCheckOutput)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User's original question:\n{question}\n\n"
                    f"Candidate evidence ({len(floored)} items):\n"
                    f"{_format_evidence_for_llm(floored)}\n\n"
                    "Return an assessment for every item, using its exact fingerprint."
                )
            ),
        ]

        try:
            output: _FactCheckOutput = await structured.ainvoke(messages)
        except Exception as exc:
            logger.warning("Fact-checker LLM call failed, falling back to deduped: %s", exc)
            verified = sorted(
                floored,
                key=lambda e: (e.relevance_score, e.source_quality),
                reverse=True,
            )
            return {"verified_evidence": verified, "status": "writing"}

        # 3. Apply assessments by fingerprint.
        by_fp: dict[str, Evidence] = {e.fingerprint: e for e in floored}
        verified: list[Evidence] = []
        for a in output.assessments:
            base = by_fp.get(a.fingerprint)
            if base is None or not a.keep:
                continue
            verified.append(
                base.model_copy(
                    update={
                        "relevance_score": a.relevance_score,
                        "source_quality": a.source_quality,
                    }
                )
            )

        # Re-apply relevance floor against the *revised* scores.
        verified = [e for e in verified if e.relevance_score >= cfg.min_source_relevance]
        verified.sort(
            key=lambda e: (e.relevance_score, e.source_quality),
            reverse=True,
        )

        logger.info("Fact-checker kept %d of %d candidates", len(verified), len(floored))
        return {"verified_evidence": verified, "status": "writing"}

    return fact_checker_node


fact_checker_node = make_fact_checker_node()
