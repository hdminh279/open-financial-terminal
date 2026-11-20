"""High-level Facade for Corporate Filing Intelligence Research Agent.

Encapsulates the compiled LangGraph StateGraph, provides a clean query() interface,
converts raw state outputs into strongly-typed AgentResponse models, and records
end-to-end telemetry.
"""

import time
from typing import Any, Dict, List, Optional, Union

from src.ai.agent.contracts import (
    AbstentionInfo,
    ActivityTimelineStep,
    AgentRequest,
    AgentResponse,
    ChartData,
    Citation,
    FinancialQueryPlan,
    IntentType,
    ReportSection,
    ToolExecutionResult,
)
from src.ai.agent.graph import build_agent_graph
from src.ai.agent.state import AgentState

CANONICAL_TIMELINE_STEPS = [
    "Understanding question",
    "Resolving metrics",
    "Querying DuckDB",
    "Searching SEC filings",
    "Validating evidence",
    "Preparing visualization",
]


def canonicalize_timeline(
    raw_events: List[Dict[str, Any]],
    intent: IntentType,
    has_charts: bool,
) -> List[ActivityTimelineStep]:
    """Sort and reconcile concurrent timeline events into a deterministic sequence."""
    event_by_step: Dict[str, Dict[str, Any]] = {
        e["step"]: e for e in raw_events if "step" in e
    }

    ordered_steps: List[ActivityTimelineStep] = []
    for step_name in CANONICAL_TIMELINE_STEPS:
        if step_name in event_by_step:
            ev = event_by_step[step_name]
            ordered_steps.append(
                ActivityTimelineStep(
                    step=ev["step"],
                    status=ev.get("status", "completed"),
                    description=ev.get("description"),
                    duration_ms=ev.get("duration_ms"),
                )
            )
        else:
            status = "skipped"
            description = "Not required for this inquiry"

            if step_name == "Querying DuckDB" and intent == IntentType.TEXTUAL_RESEARCH:
                description = "Skipped (pure textual research)"
            elif step_name == "Searching SEC filings" and intent == IntentType.FINANCIAL_METRIC:
                description = "Skipped (structured metric query)"
            elif step_name == "Preparing visualization" and not has_charts:
                description = "Skipped (no visual trend requested)"

            ordered_steps.append(
                ActivityTimelineStep(
                    step=step_name,
                    status=status,
                    description=description,
                    duration_ms=0.0,
                )
            )

    return ordered_steps


class ResearchAgent:
    """Enterprise Research Agent orchestrating multi-tool SEC filing intelligence."""

    def __init__(self, compiled_graph: Optional[Any] = None) -> None:
        """Initialize agent with compiled LangGraph application."""
        self._graph = compiled_graph or build_agent_graph()

    @property
    def graph(self) -> Any:
        """Access the underlying compiled LangGraph app."""
        return self._graph

    def query(
        self,
        request: Union[AgentRequest, str],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """Execute end-to-end query workflow across LangGraph state graph.

        Args:
            request: Natural language query string or validated AgentRequest.
            conversation_history: Optional list of previous chat turns for contextual resolution.

        Returns:
            Validated AgentResponse envelope with grounded answers, citations, and telemetry.
        """
        start_time = time.perf_counter()

        query_text = request.query if isinstance(request, AgentRequest) else str(request).strip()

        # Initialize input state
        initial_state: AgentState = {
            "query": query_text,
            "conversation_history": conversation_history or [],
            "tool_results": [],
            "evidence_items": [],
            "citations": [],
            "charts": [],
            "report_sections": [],
        }

        # Invoke the compiled LangGraph workflow
        final_state: Dict[str, Any] = self._graph.invoke(initial_state)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Unpack intent
        raw_intent = final_state.get("intent", IntentType.OUT_OF_SCOPE.value)
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            intent = IntentType.OUT_OF_SCOPE

        # Unpack citations
        citations = [
            Citation(**c) if isinstance(c, dict) else c
            for c in final_state.get("citations", [])
        ]

        # Unpack charts
        charts = [
            ChartData(**ch) if isinstance(ch, dict) else ch
            for ch in final_state.get("charts", [])
        ]

        # Unpack report sections
        report_sections = [
            ReportSection(**sec) if isinstance(sec, dict) else sec
            for sec in final_state.get("report_sections", [])
        ]

        # Unpack abstention info
        raw_abs = final_state.get("abstention_info")
        abstention = AbstentionInfo(**raw_abs) if raw_abs else None

        # Unpack tool results
        tool_results = [
            ToolExecutionResult(**tr) if isinstance(tr, dict) else tr
            for tr in final_state.get("tool_results", [])
        ]

        # Unpack query plan
        raw_plan = final_state.get("query_plan")
        query_plan = FinancialQueryPlan(**raw_plan) if raw_plan else None

        answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""

        # Unpack and canonicalize activity timeline
        raw_timeline = final_state.get("activity_timeline", [])
        ordered_timeline = canonicalize_timeline(
            raw_events=raw_timeline,
            intent=intent,
            has_charts=len(charts) > 0,
        )

        return AgentResponse(
            query=query_text,
            intent=intent,
            answer=answer,
            citations=citations,
            charts=charts,
            report_sections=report_sections,
            abstention=abstention,
            tool_results=tool_results,
            execution_time_ms=total_latency_ms,
            query_plan=query_plan,
            activity_timeline=ordered_timeline,
        )
