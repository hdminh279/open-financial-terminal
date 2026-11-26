"""Corporate Filing Intelligence AI Agent Package.

Exports the primary ResearchAgent facade, domain contracts, state schema,
and graph compilation functions.
"""

from backend.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    AgentRequest,
    AgentResponse,
    ChartData,
    ChartType,
    Citation,
    ComparisonType,
    EvidenceItem,
    EvidenceSourceType,
    ExecutionStrategy,
    FinancialQueryPlan,
    IntentType,
    ReportSection,
    RoutingDecision,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from backend.ai.agent.graph import build_agent_graph
from backend.ai.agent.research_agent import ResearchAgent
from backend.ai.agent.state import AgentState

__all__ = [
    "ResearchAgent",
    "AgentState",
    "AgentRequest",
    "AgentResponse",
    "IntentType",
    "ComparisonType",
    "FinancialQueryPlan",
    "ChartType",
    "ChartData",
    "ReportSection",
    "Citation",
    "EvidenceItem",
    "EvidenceSourceType",
    "AbstentionInfo",
    "AbstentionReason",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "ExecutionStrategy",
    "RoutingDecision",
    "build_agent_graph",
]
