"""LangGraph router node and conditional fan-out edge dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.ai.agent.contracts import IntentType
from src.ai.agent.state import AgentState

if TYPE_CHECKING:
    from src.ai.agent.router import IntentRouter


def router_node(state: AgentState, router: IntentRouter | None = None) -> dict[str, Any]:
    """LangGraph node executing autonomous intent planning and parameter synthesis."""
    query = state.get("query", "").strip()
    history = state.get("conversation_history") or []
    if router is None:
        from src.ai.agent.router import IntentRouter

        active_router = IntentRouter()
    else:
        active_router = router

    decision = active_router.route(query, conversation_history=history)

    step_understanding = {
        "step": "Understanding question",
        "status": "completed",
        "description": f"Classified intent as {decision.intent.value} ({decision.execution_strategy.value})",
        "duration_ms": None,
    }

    metrics_desc = (
        f"Resolved tickers: {decision.tickers or ['none']}, "
        f"metrics: {decision.query_plan.metrics if decision.query_plan else []}"
    )
    step_metrics = {
        "step": "Resolving metrics",
        "status": "completed" if decision.intent != IntentType.OUT_OF_SCOPE else "skipped",
        "description": metrics_desc if decision.intent != IntentType.OUT_OF_SCOPE else "Skipped for out-of-scope query",
        "duration_ms": None,
    }

    state_update: dict[str, Any] = {
        "intent": decision.intent.value,
        "routing_decision": decision.model_dump(),
        "query_plan": decision.query_plan.model_dump() if decision.query_plan else None,
        "sql_params": decision.sql_params.model_dump() if decision.sql_params else None,
        "rag_params": decision.rag_params.model_dump() if decision.rag_params else None,
        "web_params": decision.web_params.model_dump() if decision.web_params else None,
        "execution_strategy": decision.execution_strategy.value,
        "activity_timeline": [step_understanding, step_metrics],
    }
    if decision.token_usage:
        state_update["llm_usage"] = [decision.token_usage]
    return state_update


def route_next_step(state: AgentState) -> list[str] | str:
    """LangGraph conditional edge function determining fan-out or direct routing."""
    intent = state.get("intent")

    if intent == IntentType.OUT_OF_SCOPE.value:
        return "abstention_node"

    routing_decision = state.get("routing_decision") or {}
    target_tools = routing_decision.get("target_tools") or []

    query_plan = state.get("query_plan") or {}
    requires_sql = query_plan.get("requires_sql", False) or ("financial_sql_tool" in target_tools)
    requires_rag = query_plan.get("requires_rag", False) or ("hybrid_retriever" in target_tools)
    requires_web = query_plan.get("requires_web", False) or ("web_search_tool" in target_tools)

    selected_nodes: list[str] = []
    if requires_sql:
        selected_nodes.append("sql_node")
    if requires_rag:
        selected_nodes.append("rag_node")
    if requires_web:
        selected_nodes.append("web_node")

    if len(selected_nodes) > 1:
        return selected_nodes
    if len(selected_nodes) == 1:
        return selected_nodes[0]

    if intent == IntentType.FINANCIAL_METRIC.value:
        return "sql_node"

    if intent == IntentType.TEXTUAL_RESEARCH.value:
        return "rag_node"

    if intent in (IntentType.MIXED_RESEARCH.value, IntentType.DEEP_RESEARCH_REPORT.value):
        return ["sql_node", "rag_node"]

    return "abstention_node"
