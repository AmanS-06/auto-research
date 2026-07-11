"""Planner agent node.

Takes the user's question and produces a list of focused, non-overlapping
sub-questions (ResearchTask) for the Researcher to work on.

Uses LangChain's ``with_structured_output`` for a typed Pydantic plan, so
the JSON parsing happens inside the LLM client rather than in fragile
``json.loads`` calls of free-form text.
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
from app.schemas.evidence import ResearchTask

logger = logging.getLogger(__name__)


class PlannerOutput(BaseModel):
    """Structured output the Planner LLM is asked to produce."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[ResearchTask] = Field(..., min_length=1)


def _normalize_tasks(tasks: list[ResearchTask], max_tasks: int) -> list[ResearchTask]:
    """Clip, re-id, and de-dup tasks defensively."""
    seen: set[str] = set()
    normalized: list[ResearchTask] = []
    for task in tasks[:max_tasks]:
        key = task.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            ResearchTask(
                id=f"t{len(normalized) + 1}",
                question=task.question.strip(),
                rationale=task.rationale.strip(),
            )
        )
    return normalized


def make_planner_node(
    llm: BaseChatModel | None = None,
    *,
    settings: Settings | None = None,
):
    """Build the planner node with injected dependencies (test-friendly)."""
    cfg = settings or default_settings
    system_prompt = load_prompt("planner")

    async def planner_node(state: ResearchState) -> dict[str, Any]:
        question = (state.get("question") or "").strip()
        if not question:
            return {
                "status": "error",
                "error": "Planner received empty question.",
                "research_tasks": [],
            }

        max_tasks = int(state.get("max_tasks") or cfg.max_research_tasks)
        max_tasks = max(1, min(max_tasks, cfg.max_research_tasks))

        chat = llm or get_default_llm()
        structured = chat.with_structured_output(PlannerOutput)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User research question:\n{question}\n\n"
                    f"max_tasks = {max_tasks}\n\n"
                    "Return the plan now."
                )
            ),
        ]

        try:
            result: PlannerOutput = await structured.ainvoke(messages)
        except Exception as exc:
            logger.exception("Planner LLM call failed")
            return {
                "status": "error",
                "error": f"Planner failed: {exc}",
                "research_tasks": [],
            }

        tasks = _normalize_tasks(result.tasks, max_tasks=max_tasks)
        if not tasks:
            return {
                "status": "error",
                "error": "Planner produced no usable tasks.",
                "research_tasks": [],
            }

        logger.info("Planner produced %d task(s) for question: %r", len(tasks), question)
        return {"research_tasks": tasks, "status": "researching"}

    return planner_node


# Default node bound to the default LLM; tests should use ``make_planner_node``.
planner_node = make_planner_node()
