"""Writer agent node.

Synthesizes the verified evidence into a Markdown research report with
numbered, deterministic citations.

Citation ids are assigned in Python (1, 2, 3, ...) *before* calling the
LLM, and are presented to the LLM in the prompt so it can reference them
inline as ``[1]``, ``[2]`` etc. This guarantees the ``Sources`` list and
in-body citation ids stay consistent regardless of LLM behavior.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.langgraph.state import ResearchState
from app.core.llm import get_default_llm
from app.core.prompts import load_prompt
from app.schemas.evidence import Citation, Evidence

logger = logging.getLogger(__name__)


class _WriterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1)
    body_markdown: str = Field(..., min_length=1)


def _format_evidence_for_writer(evidence: list[Evidence]) -> str:
    lines: list[str] = []
    for i, e in enumerate(evidence, start=1):
        lines.append(
            f"[{i}] task_id: {e.task_id}\n"
            f"    title: {e.title}\n"
            f"    url: {e.source_url}\n"
            f"    relevance: {e.relevance_score:.2f} | quality: {e.source_quality:.2f}\n"
            f"    content: {e.content}"
        )
    return "\n\n".join(lines)


def _build_citations(evidence: list[Evidence]) -> list[Citation]:
    return [
        Citation(
            id=str(i),
            source_url=str(e.source_url),
            title=e.title,
            snippet=e.snippet or e.content[:200],
            task_id=e.task_id,
        )
        for i, e in enumerate(evidence, start=1)
    ]


def _insufficient_evidence_report(question: str) -> dict[str, Any]:
    summary = (
        "The pipeline could not gather sufficient verified evidence to answer "
        "the question. See the report for what was attempted."
    )
    body = (
        f"# {question}\n\n"
        "## Summary\n"
        f"{summary}\n\n"
        "## Key findings\n"
        "- No verified evidence passed the quality threshold.\n\n"
        "## Detailed analysis\n"
        "The planner generated sub-questions and the researcher queried the web, "
        "but every candidate source was either off-topic, duplicated, or rejected "
        "by the fact-checker. This usually means the topic is too new, too "
        "narrow, or requires non-public sources.\n\n"
        "## Conflicting evidence or open questions\n"
        "None identified — the underlying issue is lack of evidence, not "
        "disagreement.\n\n"
        "## Sources\n"
        "_None_\n"
    )
    return {
        "summary": summary,
        "report_markdown": body,
        "citations": [],
        "status": "complete",
    }


def make_writer_node(llm: BaseChatModel | None = None):
    """Build the writer node with injected dependencies."""
    system_prompt = load_prompt("writer")

    async def writer_node(state: ResearchState) -> dict[str, Any]:
        question = (state.get("question") or "").strip()
        evidence: list[Evidence] = state.get("verified_evidence") or []

        if not evidence:
            logger.info("Writer received no verified evidence; emitting fallback report")
            return _insufficient_evidence_report(question or "(no question provided)")

        citations = _build_citations(evidence)
        chat = llm or get_default_llm()
        structured = chat.with_structured_output(_WriterOutput)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User's original question:\n{question}\n\n"
                    f"Verified evidence (use the bracketed ids as citation ids):\n"
                    f"{_format_evidence_for_writer(evidence)}\n\n"
                    "Write the report now, following the required Markdown structure."
                )
            ),
        ]

        try:
            output: _WriterOutput = await structured.ainvoke(messages)
        except Exception as exc:
            logger.exception("Writer LLM call failed")
            return {
                "status": "error",
                "error": f"Writer failed: {exc}",
                "report_markdown": "",
                "summary": "",
                "citations": citations,
            }

        logger.info(
            "Writer produced report (%d chars) with %d citations",
            len(output.body_markdown),
            len(citations),
        )
        return {
            "summary": output.summary.strip(),
            "report_markdown": output.body_markdown.strip(),
            "citations": citations,
            "status": "complete",
        }

    return writer_node


writer_node = make_writer_node()
