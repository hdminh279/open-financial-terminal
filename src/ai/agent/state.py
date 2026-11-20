"""LangGraph AgentState definitions and reducers for Corporate Filing Intelligence.

Defines the central state dictionary carried across graph nodes, with reducers
for accumulating tool execution logs, evidence records, charts, and report sections.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


def timeline_reducer(
    existing: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Thread-safe event accumulator for LangGraph parallel fan-out/fan-in branches.

    Safely combines timeline events from concurrent nodes (e.g., sql_node and rag_node)
    without race conditions, state clobbering, or event loss.
    """
    accumulated = list(existing or [])
    if incoming:
        accumulated.extend(incoming)
    return accumulated


class AgentState(TypedDict, total=False):
    """Central state passed between nodes in the LangGraph StateGraph."""

    # 1. Inputs & Intent Planning
    query: str
    intent: str
    conversation_history: Optional[List[Dict[str, Any]]]
    routing_decision: Optional[Dict[str, Any]]
    query_plan: Optional[Dict[str, Any]]
    sql_params: Optional[Dict[str, Any]]
    rag_params: Optional[Dict[str, Any]]
    execution_strategy: str

    # 2. Tool Data Stores
    sql_data: Optional[Dict[str, Any]]
    rag_data: Optional[Dict[str, Any]]
    
    # 3. Accumulated Logs & Evidences (Reducers for parallel Fan-Out/Fan-In)
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    evidence_items: Annotated[List[Dict[str, Any]], operator.add]
    activity_timeline: Annotated[List[Dict[str, Any]], timeline_reducer]

    # 4. Abstention & Quality Gate Telemetry
    abstention_info: Optional[Dict[str, Any]]
    is_abstaining: bool
    is_partial: bool

    # 5. Synthesis & Visual Artifacts
    draft_answer: Optional[str]
    final_answer: str
    citations: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    report_sections: List[Dict[str, Any]]

    # 6. Pipeline Execution Metrics
    execution_time_ms: float
    error: Optional[str]
