"""LangGraph StateGraph assembly for the Corporate Filing Intelligence agent."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.ai.agent.nodes.abstention_node import abstention_node, decide_after_abstention
from src.ai.agent.nodes.citation_node import citation_node, synthesis_node
from src.ai.agent.nodes.rag_node import (
    _compact_rag_chunk_text,
    _infer_rag_form,
    get_retriever,
    rag_node,
)
from src.ai.agent.nodes.router_node import route_next_step, router_node
from src.ai.agent.nodes.sql_node import get_sql_tool, sql_node
from src.ai.agent.nodes.web_node import get_web_tool, web_node
from src.ai.agent.state import AgentState
from src.core.financial.company_registry import resolve_filing_form as _resolve_filing_form

__all__ = [
    "_compact_rag_chunk_text",
    "_infer_rag_form",
    "_resolve_filing_form",
    "abstention_node",
    "build_agent_graph",
    "citation_node",
    "decide_after_abstention",
    "get_retriever",
    "get_sql_tool",
    "get_web_tool",
    "rag_node",
    "route_next_step",
    "router_node",
    "sql_node",
    "synthesis_node",
    "web_node",
]


def build_agent_graph() -> Any:
    """Construct and compile the LangGraph StateGraph workflow with parallel tool fan-out."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("web_node", web_node)
    workflow.add_node("abstention_node", abstention_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("citation_node", citation_node)

    workflow.add_edge(START, "router_node")
    workflow.add_conditional_edges(
        "router_node",
        route_next_step,
        ["sql_node", "rag_node", "web_node", "abstention_node"],
    )

    workflow.add_edge("sql_node", "abstention_node")
    workflow.add_edge("rag_node", "abstention_node")
    workflow.add_edge("web_node", "abstention_node")

    workflow.add_conditional_edges(
        "abstention_node",
        decide_after_abstention,
        ["synthesis_node", "citation_node"],
    )

    workflow.add_edge("synthesis_node", "citation_node")
    workflow.add_edge("citation_node", END)
    return workflow.compile()
