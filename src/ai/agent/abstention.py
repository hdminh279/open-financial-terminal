"""Multi-Layer Granular Abstention Gate for LangGraph

Implements 3-tier abstention checks:
1. Scope & Domain Validation (Supported companies catalog).
2. Evidence Health & Completeness (distinguishing tool_failed vs no_data_found).
3. Partial Failure Handling for Mixed / Report queries (answering supported portion
   and explicitly abstaining from unsupported portion without failing the graph).
"""

import logging
from typing import Any, Dict, List, Optional

from src.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    IntentType,
    ToolExecutionStatus,
)
from src.ai.agent.state import AgentState
from src.core.financial.company_registry import get_supported_tickers

logger = logging.getLogger(__name__)


class AbstentionGate:
    """Evaluates query scope, tool results health, and evidence sufficiency (Data-Driven)."""

    def evaluate(self, state: AgentState) -> AbstentionInfo:
        """Perform granular multi-tier data-driven abstention check.

        Args:
            state: Current AgentState with intent, tool_results, and evidence_items.

        Returns:
            AbstentionInfo detailing full, partial, or zero abstention.
        """
        intent_str = state.get("intent", IntentType.OUT_OF_SCOPE.value)
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.OUT_OF_SCOPE

        routing_decision = state.get("routing_decision") or {}
        reasoning = routing_decision.get("reasoning", "")
        tickers = list(routing_decision.get("tickers") or [])
        if not tickers and routing_decision.get("ticker"):
            tickers = [routing_decision.get("ticker")]
        ticker_str = ", ".join(tickers) if tickers else "the target company"

        years = list(routing_decision.get("fiscal_years") or [])
        if not years and routing_decision.get("fiscal_year"):
            years = [routing_decision.get("fiscal_year")]
        year_str = ", ".join(map(str, years)) if years else ""

        # -------------------------------------------------------------
        # Tier 1: Direct Scope Rejection (Dynamic Tickers from Registry)
        # -------------------------------------------------------------
        supported_tickers = get_supported_tickers()
        if intent == IntentType.OUT_OF_SCOPE:
            invalid_tickers = [t for t in tickers if t not in supported_tickers]
            if invalid_tickers:
                t_display = ", ".join(invalid_tickers)
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.UNSUPPORTED_COMPANY,
                    explanation=f"Company '{t_display}' is not currently supported. Corporate Filing Intelligence supports: {sorted(supported_tickers)}.",
                    unsupported_aspects=[f"company:{t}" for t in invalid_tickers],
                )

            default_en = "This inquiry is outside the operational scope of corporate SEC filings."
            return AbstentionInfo(
                is_abstaining=True,
                is_partial=False,
                reason=AbstentionReason.UNSUPPORTED_TOPIC,
                explanation=reasoning or default_en,
            )

        # -------------------------------------------------------------
        # Tier 2: Tool Execution Health Check
        # -------------------------------------------------------------
        tool_results: List[Dict[str, Any]] = state.get("tool_results", [])
        evidence_items: List[Dict[str, Any]] = state.get("evidence_items", [])

        sql_results = [r for r in tool_results if r.get("tool_name") == "financial_sql_tool"]
        rag_results = [r for r in tool_results if r.get("tool_name") == "hybrid_retriever"]
        web_results = [r for r in tool_results if r.get("tool_name") == "web_search_tool"]

        sql_success = any(r.get("status") == ToolExecutionStatus.SUCCESS.value for r in sql_results)
        rag_success = any(r.get("status") == ToolExecutionStatus.SUCCESS.value for r in rag_results)
        web_success = any(r.get("status") == ToolExecutionStatus.SUCCESS.value for r in web_results)

        sql_failed = any(r.get("status") == ToolExecutionStatus.TOOL_FAILED.value for r in sql_results)
        rag_failed = any(r.get("status") == ToolExecutionStatus.TOOL_FAILED.value for r in rag_results)
        web_failed = any(r.get("status") == ToolExecutionStatus.TOOL_FAILED.value for r in web_results)

        # -------------------------------------------------------------
        # Tier 3: Intent-Specific Granular Resolution
        # -------------------------------------------------------------
        # Pure Live Web Search evaluation
        if web_results and not sql_results and not rag_results:
            if web_failed:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.TOOL_ERROR,
                    explanation="A technical error occurred while executing live web search.",
                )
            if not web_success or len(evidence_items) == 0:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    explanation=f"No relevant live web news or articles found for {ticker_str}.",
                )
            return AbstentionInfo(is_abstaining=False, is_partial=False)

        if intent == IntentType.FINANCIAL_METRIC:
            if sql_failed:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.TOOL_ERROR,
                    explanation="A technical database error occurred while querying financial statements.",
                )
            if not sql_success:
                period_str = f" for period {year_str}" if year_str else ""
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    explanation=f"No reported financial records found for {ticker_str}{period_str} in verified SEC filings.",
                )
            return AbstentionInfo(is_abstaining=False, is_partial=False)

        if intent == IntentType.TEXTUAL_RESEARCH:
            if web_success and len(evidence_items) > 0:
                return AbstentionInfo(is_abstaining=False, is_partial=False)
            if rag_failed and not web_success:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.TOOL_ERROR,
                    explanation="A technical retrieval service error occurred while searching filing texts.",
                )
            if (not rag_success and not web_success) or len(evidence_items) == 0:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    explanation=f"No relevant narrative disclosures found in SEC 10-K/10-Q filings for {ticker_str} for this inquiry.",
                )
            return AbstentionInfo(is_abstaining=False, is_partial=False)

        # Mixed & Report Intents: Evaluate Partial Failure
        if intent in (IntentType.MIXED_RESEARCH, IntentType.DEEP_RESEARCH_REPORT):
            if not sql_success and not rag_success and not web_success:
                reason = AbstentionReason.TOOL_ERROR if (sql_failed or rag_failed or web_failed) else AbstentionReason.INSUFFICIENT_EVIDENCE
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=reason,
                    explanation=f"Unable to retrieve verified financial figures, filing disclosures, or live news for {ticker_str}.",
                )

            # SQL succeeded, but RAG failed or found no textual context
            if sql_success and not rag_success:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=True,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    explanation="Financial metrics were verified from official SEC filings, but no narrative explanation or qualitative commentary was found in filing disclosures for this period.",
                    unsupported_aspects=["qualitative_explanation"],
                )

            # RAG succeeded, but SQL failed or found no numerical records
            if rag_success and not sql_success:
                return AbstentionInfo(
                    is_abstaining=True,
                    is_partial=True,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    explanation="Qualitative filing disclosures were retrieved, but exact standardized accounting metrics were not reported for this specific period.",
                    unsupported_aspects=["numerical_metrics"],
                )

            return AbstentionInfo(is_abstaining=False, is_partial=False)

        return AbstentionInfo(is_abstaining=False, is_partial=False)


# ---------------------------------------------------------------------------
# LangGraph Node and Conditional Edge Functions
# ---------------------------------------------------------------------------

def abstention_node(
    state: AgentState,
    gate: Optional[AbstentionGate] = None,
) -> Dict[str, Any]:
    """LangGraph node assessing evidence sufficiency and partial abstention.

    Args:
        state: Current AgentState.
        gate: Injected or default AbstentionGate instance.

    Returns:
        State dictionary update for LangGraph.
    """
    evaluator = gate or AbstentionGate()
    abstention_info = evaluator.evaluate(state)

    final_answer = ""
    if abstention_info.is_abstaining and not abstention_info.is_partial:
        final_answer = abstention_info.explanation

    validation_desc = (
        "Abstention triggered: " + (abstention_info.explanation[:60] + "...")
        if abstention_info.is_abstaining
        else "Sufficient evidence verified"
    )

    timeline_step = {
        "step": "Validating evidence",
        "status": "completed",
        "description": validation_desc,
        "duration_ms": None,
    }

    return {
        "abstention_info": abstention_info.model_dump(),
        "is_abstaining": abstention_info.is_abstaining,
        "is_partial": abstention_info.is_partial,
        "final_answer": final_answer,
        "activity_timeline": [timeline_step],
    }


def decide_after_abstention(state: AgentState) -> str:
    """LangGraph conditional edge deciding whether to synthesize answer or terminate.

    Args:
        state: Current AgentState containing 'is_abstaining' and 'is_partial'.

    Returns:
        Target node name: 'synthesis_node' or 'citation_node'.
    """
    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)

    if is_abstaining and not is_partial:
        return "citation_node"

    return "synthesis_node"
