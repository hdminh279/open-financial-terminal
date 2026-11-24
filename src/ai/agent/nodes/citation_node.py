"""LangGraph synthesis and citation finalization nodes."""

import logging
import re
from typing import Any

from src.ai.agent.chart_engine import ChartEngine
from src.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    IntentType,
    ReportSection,
)
from src.ai.agent.state import AgentState
from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.factory import LLMFactory

logger = logging.getLogger(__name__)


def synthesis_node(
    state: AgentState,
    llm_adapter: BaseLLMAdapter | None = None,
) -> dict[str, Any]:
    """LangGraph node synthesizing a grounded response with inline citation markers."""
    from src.ai.agent.citation import build_synthesis_messages

    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)

    # Full abstention short-circuit: answer already formulated by abstention_node
    if is_abstaining and not is_partial:
        return {"draft_answer": state.get("final_answer", "")}

    # Streaming mode defers token generation to ResearchAgent.stream_query()
    if state.get("stream_mode"):
        return {"draft_answer": ""}

    evidence_dicts = state.get("evidence_items", [])
    adapter = llm_adapter or LLMFactory.get_adapter(profile="chat")
    messages = build_synthesis_messages(state)

    try:
        response = adapter.generate(
            messages=messages,
            temperature=0.0,
        )
        result: dict[str, Any] = {"draft_answer": response.content}
        if response.usage is not None:
            result["llm_usage"] = [response.usage.model_dump() | {"model": response.model}]
        return result
    except Exception as exc:
        logger.error("Synthesis node failed: %s", exc, exc_info=True)
        return {
            "draft_answer": (
                f"Error synthesizing response: {exc}. Raw evidence collected: {len(evidence_dicts)} items."
            )
        }


def citation_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node attaching verified citations, visual charts, and report sections."""
    from src.ai.agent.citation import (
        CitationEngine,
        build_fiscal_period_notes,
        detect_insufficient_evidence,
        sanitize_answer_formatting,
    )

    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)
    draft_answer = state.get("draft_answer") or state.get("final_answer") or ""

    # In streaming mode, defer citation finalization until stream_query() finishes streaming tokens
    if state.get("stream_mode") and not draft_answer and not (is_abstaining and not is_partial):
        return {}

    # Full abstention short-circuit: no citations or charts to attach
    if is_abstaining and not is_partial:
        return {
            "citations": [],
            "charts": [],
            "report_sections": [],
            "final_answer": draft_answer,
        }

    evidence_dicts = state.get("evidence_items", [])
    sql_data = state.get("sql_data")
    routing_decision = state.get("routing_decision") or {}
    intent_str = state.get("intent", "")
    query = state.get("query", "")

    citations = CitationEngine.build_citations(evidence_dicts)

    query_plan = state.get("query_plan") or {}
    chart_pref = query_plan.get("chart_preference") or routing_decision.get("chart_preference") or "none"
    chart_engine = ChartEngine()
    charts = chart_engine.extract_financial_charts(
        sql_data=sql_data,
        chart_preference=chart_pref,
        query_plan=query_plan,
    )

    if charts:
        c_type = charts[0].chart_type.value if hasattr(charts[0], "chart_type") else "chart"
        vis_step = {
            "step": "Preparing visualization",
            "status": "completed",
            "description": f"Generated {len(charts)} financial chart(s) ({c_type})",
            "duration_ms": None,
        }
    else:
        vis_step = {
            "step": "Preparing visualization",
            "status": "skipped",
            "description": "No visualization requested or insufficient data points",
            "duration_ms": None,
        }

    clean_draft = re.sub(
        r"\n+(?:#{1,3}\s+)?(Financial Visualizations|Visualizations|Sources & References)[\s\S]*$",
        "",
        draft_answer,
        flags=re.IGNORECASE,
    ).strip()
    clean_draft = re.sub(r"```(?:mermaid)?\s*xychart-beta[\s\S]*?```", "", clean_draft).strip()
    clean_draft = re.sub(r"\bxychart-beta\s+title[\s\S]*?(?=\n\n|\Z)", "", clean_draft).strip()
    raw_clean = clean_draft if clean_draft else draft_answer.strip()

    has_sql_success = any(
        r.get("tool_name") == "financial_sql_tool" and r.get("status") == "success"
        for r in (state.get("tool_results") or [])
        if isinstance(r, dict)
    )
    synthesis_abstained, stripped_answer = detect_insufficient_evidence(
        raw_clean,
        is_already_abstaining=is_abstaining,
        is_partial=is_partial,
        has_sql_success=has_sql_success,
    )
    final_answer = sanitize_answer_formatting(stripped_answer)

    if synthesis_abstained:
        abstention_payload = AbstentionInfo(
            is_abstaining=True,
            is_partial=False,
            reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            explanation=final_answer,
        ).model_dump()
        return {
            "citations": [],
            "charts": [],
            "report_sections": [],
            "final_answer": final_answer,
            "is_abstaining": True,
            "abstention_info": abstention_payload,
            "activity_timeline": [vis_step],
        }

    fiscal_notes = build_fiscal_period_notes(sql_data=sql_data, query=query)
    if fiscal_notes:
        missing_notes = [n for n in fiscal_notes if n not in final_answer]
        if missing_notes:
            final_answer = f"{final_answer}\n\n" + "\n".join(missing_notes)

    report_sections: list[dict[str, Any]] = []
    if intent_str == IntentType.DEEP_RESEARCH_REPORT.value:
        raw_sections = re.split(r"\n(?=#{1,2}\s+)", draft_answer)
        for idx, sec in enumerate(raw_sections):
            lines = sec.strip().split("\n", 1)
            header = lines[0].replace("#", "").strip() if lines else f"Section {idx + 1}"
            body = lines[1].strip() if len(lines) > 1 else sec.strip()

            matching_chart = charts[0] if (idx == 1 and charts) else None
            report_section = ReportSection(
                section_id=f"sec_{idx + 1}",
                title=header,
                content=body,
                chart=matching_chart,
            )
            report_sections.append(report_section.model_dump())

    return {
        "citations": [c.model_dump() for c in citations],
        "charts": [c.model_dump() for c in charts],
        "report_sections": report_sections,
        "final_answer": final_answer,
        "activity_timeline": [vis_step],
    }
