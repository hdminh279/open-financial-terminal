"""LangGraph abstention evaluation node and post-abstention routing edge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.ai.agent.state import AgentState

if TYPE_CHECKING:
    from src.ai.agent.abstention import AbstentionGate


def abstention_node(
    state: AgentState,
    gate: AbstentionGate | None = None,
) -> dict[str, Any]:
    """LangGraph node assessing evidence sufficiency and partial abstention."""
    if gate is None:
        from src.ai.agent.abstention import AbstentionGate

        evaluator = AbstentionGate()
    else:
        evaluator = gate

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
    """LangGraph conditional edge deciding whether to synthesize answer or short-circuit."""
    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)

    if is_abstaining and not is_partial:
        return "citation_node"

    return "synthesis_node"
