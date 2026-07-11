# LOCKED BY Worker B

"""Research StateGraph builder.

Wires the four agents (planner -> researcher -> fact_checker -> writer)
into a single LangGraph state machine. The graph short-circuits to END
whenever a node sets ``state["status"] == "error"``.

The builder takes optional injected dependencies (LLM, Serper client,
checkpointer) so:
    * Tests can inject mocks.
    * Worker A can plug in their PostgreSQL checkpointer without touching
      this module.
    * Worker C can call :func:`build_research_graph` with no args for the
      default production wiring.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.core.config import Settings
from app.core.langgraph.nodes.fact_checker import make_fact_checker_node
from app.core.langgraph.nodes.planner import make_planner_node
from app.core.langgraph.nodes.researcher import make_researcher_node
from app.core.langgraph.nodes.writer import make_writer_node
from app.core.langgraph.state import ResearchState, extract_report_payload
from app.core.langgraph.tools.web_search import SerperClient

logger = logging.getLogger(__name__)


PLANNER = "planner"
RESEARCHER = "researcher"
FACT_CHECKER = "fact_checker"
WRITER = "writer"


def _route_or_end(target: str):
    """Conditional edge: go to ``target`` unless ``state["status"] == "error"``."""

    def _router(state: ResearchState) -> str:
        if state.get("status") == "error":
            return END
        return target

    return _router


def build_research_graph(
    *,
    llm: BaseChatModel | None = None,
    search_client: SerperClient | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the research graph.

    Args:
        llm: Override the chat model used by every agent. Defaults to the
            cached default LLM (DeepSeek).
        search_client: Override the Serper client used by the Researcher.
            If ``None``, the Researcher creates and closes one per call.
        settings: Settings override (limits, thresholds). Defaults to the
            global :data:`app.core.config.settings`.
        checkpointer: Optional LangGraph checkpointer for state persistence.
            Worker A wires the PostgreSQL saver here.
    """
    planner = make_planner_node(llm=llm, settings=settings)
    researcher = make_researcher_node(llm=llm, search_client=search_client, settings=settings)
    fact_checker = make_fact_checker_node(llm=llm, settings=settings)
    writer = make_writer_node(llm=llm)

    graph: StateGraph = StateGraph(ResearchState)
    graph.add_node(PLANNER, planner)
    graph.add_node(RESEARCHER, researcher)
    graph.add_node(FACT_CHECKER, fact_checker)
    graph.add_node(WRITER, writer)

    graph.set_entry_point(PLANNER)
    graph.add_conditional_edges(PLANNER, _route_or_end(RESEARCHER), [RESEARCHER, END])
    graph.add_conditional_edges(RESEARCHER, _route_or_end(FACT_CHECKER), [FACT_CHECKER, END])
    graph.add_conditional_edges(FACT_CHECKER, _route_or_end(WRITER), [WRITER, END])
    graph.add_edge(WRITER, END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.debug("Research graph compiled (checkpointer=%s)", bool(checkpointer))
    return compiled


__all__ = ["build_research_graph", "extract_report_payload"]
