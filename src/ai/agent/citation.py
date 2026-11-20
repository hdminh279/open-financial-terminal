"""Citation, Synthesis, and Visual Chart Generation Engine for LangGraph

Implements grounded response synthesis via DeepSeek Chat profile, assigns sequential
citation IDs [1], [2], links authoritative SEC EDGAR URLs, automatically renders
Mermaid xychart-beta visualizations, and structures deep research report sections.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.ai.agent.contracts import (
    ChartData,
    ChartType,
    Citation,
    EvidenceItem,
    EvidenceSourceType,
    IntentType,
    ReportSection,
)
from src.ai.agent.chart_engine import ChartEngine
from src.ai.agent.state import AgentState
from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.contracts import ChatMessage
from src.ai.llm.factory import LLMFactory
from src.core.financial.metric_registry import MetricRegistry
from src.core.financial.semantic_guardrails import FinancialOntology

logger = logging.getLogger(__name__)

_metric_registry: Optional[MetricRegistry] = None
_financial_ontology: Optional[FinancialOntology] = None


def _get_metric_registry() -> MetricRegistry:
    global _metric_registry
    if _metric_registry is None:
        _metric_registry = MetricRegistry()
    return _metric_registry


def _get_financial_ontology() -> FinancialOntology:
    global _financial_ontology
    if _financial_ontology is None:
        _financial_ontology = FinancialOntology()
    return _financial_ontology

SYNTHESIS_SYSTEM_PROMPT = """You are a senior financial research analyst and SEC filings specialist.
Your mission is to synthesize a factual, clear, and comprehensive answer to the user's question,
grounded STRICTLY and EXCLUSIVELY in the provided SEC EDGAR evidence.

### Financial Semantic Guardrails (Mandatory 4-Layer Rules):
1. Layer 1 (GAAP vs Non-GAAP):
   - Clearly identify GAAP metrics (Revenues, Net Income, Operating Income) versus Non-GAAP or Segment metric disclosures.
2. Layer 2 (Financial Ontology - Geographic vs Product Lines):
   - Strictly separate Geographic Segments (e.g. Americas, Europe, Greater China, Rest of Asia Pacific) from Product Categories (e.g. iPhone, Mac, iPad, Services, Azure).
   - NEVER confuse, mix, or aggregate Geographic revenue with Product Line revenue.
3. Layer 3 (Two-Tier Response Structure):
   Your answer MUST be organized into two distinct sections:
   - `### [Official SEC Disclosures]`:
     * Pure factual figures, standard Markdown tables with pipes (|), and inline citations [1], [2].
   - `### [Analysis & Commentary]`:
     * Growth rates, margins, business dynamics, and strategic commentary directly backed by the evidence.
4. Layer 4 (Mathematical Consistency):
   - Operating Margin = Operating Income / Revenues.
   - Net Margin = Net Income / Revenues.
   - Never hallucinate margins that contradict the raw numbers in the evidence.

### Language Enforcement:
- If the user's inquiry is in another language (such as Vietnamese), you MUST synthesize the entire answer fluently in that language, translating the two-tier section headers accordingly.
- For English queries, use:
  * `### [Official SEC Disclosures]`
  * `### [Analysis & Commentary]`

### Markdown Table Formatting Rule:
- When presenting multiple fiscal periods, product categories, or metric comparisons, you MUST format the data as a clean, standard GitHub Flavored Markdown Table with pipe syntax:
  | Fiscal Period / Product Line | Value | Citation Source |
  | :--- | :---: | :---: |
  | <Period_or_Segment_1> | <Amount_1> | [1] |
  | <Period_or_Segment_2> | <Amount_2> | [2] |
- NEVER output unformatted text blocks where column names and values are separated by spaces, tabs, or lone newlines. Always use markdown table pipes (|).

### Clean Content Rules:
- Do NOT output raw Mermaid `xychart-beta` blocks or raw `### Sources & References` footnote sections in the body. The UI automatically displays interactive charts and collapsible references from structured state.
- Keep the analytical answer focused, clear, and easy to read. Do not use graphical emojis.

### Grounding & Citation Rules:
1. Strict Grounding: Rely ONLY on the facts, numerical metrics, and text excerpts in the provided evidence.
   Do not speculate, extrapolate, or inject outside knowledge.
2. Inline Citations: Every statement containing numbers, dates, strategic claims, or risk disclosures
   MUST cite the corresponding evidence index using square brackets (e.g. [1], [2], or [1][2]).
3. Partial Evidence Handling: If the user asked for both metrics and explanations, but only one type of
   evidence was found, answer the supported part thoroughly and state transparently.
4. Professional Structure: Use clear headings (H2/H3), bullet points, and bold metric highlights.

### Deep Research Report Mode:
If the user requested a full report, multi-year overview, or deep dive, organize your response into standard sections:
# Executive Summary
## Financial Performance & Statement Analysis
## Business Strategy & Operational Disclosures
## Risk Factors & Regulatory Disclosures
"""


class CitationEngine:
    """Manages sequential citation indexing and authoritative SEC URL formatting."""

    @staticmethod
    def build_citations(evidence_dicts: List[Dict[str, Any]]) -> List[Citation]:
        """Convert accumulated evidence items into verified sequential Citation objects.

        Args:
            evidence_dicts: Raw evidence item dictionaries from state.

        Returns:
            List of sequential, validated Citation models.
        """
        citations: List[Citation] = []
        for idx, ev_dict in enumerate(evidence_dicts):
            citation_id = idx + 1
            ticker = ev_dict.get("ticker", "SEC")
            form = ev_dict.get("form", "Filing")
            fy = ev_dict.get("fiscal_year")
            fp = ev_dict.get("fiscal_period")
            period_str = f"({fp} {fy})" if (fp and fy) else f"(FY{fy})" if fy else ""
            item_or_metric = ev_dict.get("item_or_metric", "Filing Reference")
            source_url = ev_dict.get("source_url") or "https://www.sec.gov/edgar/searchedgar/companysearch"

            formatted_ref = (
                f"[{citation_id}] {ticker} Form {form} {period_str} "
                f"{item_or_metric} — [SEC EDGAR Source]({source_url})"
            )

            citations.append(
                Citation(
                    citation_id=citation_id,
                    formatted_reference=formatted_ref,
                    source_url=source_url,
                )
            )

        return citations

    @staticmethod
    def format_references_markdown(citations: List[Citation], draft_answer: str = "") -> str:
        """Render ordered references section in Markdown, prioritizing cited items.

        Args:
            citations: List of formatted Citation models.
            draft_answer: Optional synthesized answer text to filter only cited references.

        Returns:
            Markdown string with ### Sources & References footnote block.
        """
        if not citations:
            return ""

        cited_ids = set()
        if draft_answer:
            cited_ids = {int(m) for m in re.findall(r"\[(\d+)\]", draft_answer)}

        filtered = [c for c in citations if c.citation_id in cited_ids] if cited_ids else citations[:10]

        lines = ["\n\n### Sources & References"]
        for cit in filtered:
            lines.append(cit.formatted_reference)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph Node Functions
# ---------------------------------------------------------------------------

def synthesis_node(
    state: AgentState,
    llm_adapter: Optional[BaseLLMAdapter] = None,
) -> Dict[str, Any]:
    """LangGraph node synthesizing a fully grounded response with [1], [2] citation tags.

    Args:
        state: Current AgentState.
        llm_adapter: Injected or default DeepSeekAdapter instance.

    Returns:
        State dictionary update containing 'draft_answer'.
    """
    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)

    # Full abstention short-circuit: answer already formulated by abstention_node
    if is_abstaining and not is_partial:
        return {"draft_answer": state.get("final_answer", "")}

    query = state.get("query", "")
    intent_str = state.get("intent", IntentType.FINANCIAL_METRIC.value)
    evidence_dicts = state.get("evidence_items", [])
    abstention_info = state.get("abstention_info") or {}

    # Format collected evidence blocks
    evidence_text_blocks: List[str] = []
    for idx, ev in enumerate(evidence_dicts):
        cid = idx + 1
        src = ev.get("source_type", "evidence").upper()
        ticker = ev.get("ticker", "")
        form = ev.get("form", "")
        fy = ev.get("fiscal_year", "")
        item = ev.get("item_or_metric", "")
        summary = ev.get("content_summary", "")

        block = f"[{cid}] (Source: {src}, {ticker} {form} FY{fy} {item})\n{summary}"
        evidence_text_blocks.append(block)

    formatted_context = "\n\n".join(evidence_text_blocks) or "No factual evidence retrieved."

    partial_note = ""
    if is_partial:
        partial_note = (
            f"\n\nIMPORTANT PARTIAL COVERAGE NOTE: {abstention_info.get('explanation', '')}\n"
            "Answer the supported parts accurately, and explicitly disclose the missing aspect."
        )

    # Layer 2: Ontology disambiguation guidance
    ontology = _get_financial_ontology()
    routing_decision = state.get("routing_decision") or {}
    primary_ticker = (routing_decision.get("tickers") or [None])[0]
    disambiguation = ontology.disambiguate_query(query, ticker=primary_ticker)
    ontology_note = f"\n\nONTOLOGY NOTE: {disambiguation['guidance']}" if disambiguation.get("guidance") else ""

    # Dynamic Data Availability: Disclose excluded / future / unfiled periods
    sql_data = state.get("sql_data") or {}
    availability_reports = sql_data.get("availability_reports") or []
    unavail_items = [
        f"- {r.get('ticker')} {r.get('fiscal_period', '')} {r.get('fiscal_year')}: {r.get('reason')}"
        for r in availability_reports
        if isinstance(r, dict) and not r.get("is_available")
    ]
    unavail_note = ""
    if unavail_items:
        unavail_note = (
            f"\n\n=== EXCLUDED / UNPUBLISHED PERIODS ===\n"
            "The following requested periods cannot be reported from verified SEC filings:\n"
            + "\n".join(unavail_items)
            + "\nExplicitly disclose in your answer why these periods are excluded (e.g. fiscal year has not ended or Form 10-K is not published yet).\n"
        )

    user_prompt = (
        f"User Inquiry: {query}\n"
        f"Detected Intent: {intent_str}\n"
        f"{partial_note}{ontology_note}{unavail_note}\n\n"
        f"=== COLLECTED EVIDENCE FROM SEC FILINGS ===\n"
        f"{formatted_context}\n\n"
        f"Please provide a grounded, two-tier structured answer adhering strictly to citation and guardrail rules."
    )

    adapter = llm_adapter or LLMFactory.get_adapter(profile="chat")
    messages = [
        ChatMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]

    try:
        response = adapter.generate(
            messages=messages,
            temperature=0.0,
        )
        return {"draft_answer": response.content}
    except Exception as exc:
        logger.error("Synthesis node failed: %s", exc, exc_info=True)
        return {
            "draft_answer": (
                f"Error synthesizing response: {exc}. "
                f"Raw evidence collected: {len(evidence_dicts)} items."
            )
        }


def citation_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node appending verified references, generating visual charts, and parsing report sections.

    Args:
        state: Current AgentState.

    Returns:
        State dictionary update containing 'citations', 'final_answer', 'charts', and 'report_sections'.
    """
    is_abstaining = state.get("is_abstaining", False)
    is_partial = state.get("is_partial", False)
    draft_answer = state.get("draft_answer") or state.get("final_answer") or ""

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
    tickers = list(routing_decision.get("tickers") or [])
    if not tickers and routing_decision.get("ticker"):
        tickers = [routing_decision.get("ticker")]
    ticker_label = " vs ".join(tickers) if tickers else None
    intent_str = state.get("intent", "")

    query = state.get("query", "")

    # 1. Build Citations & Footnotes Markdown
    citations = CitationEngine.build_citations(evidence_dicts)
    references_md = CitationEngine.format_references_markdown(citations, draft_answer=draft_answer)

    # 2. Extract & Build Charts with Router preference
    query_plan = state.get("query_plan") or {}
    chart_pref = (
        query_plan.get("chart_preference")
        or routing_decision.get("chart_preference")
        or "none"
    )
    chart_engine = ChartEngine()
    charts = chart_engine.extract_financial_charts(
        sql_data=sql_data,
        chart_preference=chart_pref,
        ticker=ticker_label,
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

    # 3. Clean final answer: keep analytical text pristine, avoiding raw markdown code blocks
    clean_draft = re.sub(
        r"\n+(?:#{1,3}\s+)?(Financial Visualizations|Visualizations|Sources & References)[\s\S]*$",
        "",
        draft_answer,
        flags=re.IGNORECASE,
    ).strip()
    clean_draft = re.sub(r"```(?:mermaid)?\s*xychart-beta[\s\S]*?```", "", clean_draft).strip()
    clean_draft = re.sub(r"\bxychart-beta\s+title[\s\S]*?(?=\n\n|\Z)", "", clean_draft).strip()
    final_answer = clean_draft if clean_draft else draft_answer.strip()

    # 4. Parse Report Sections if DEEP_RESEARCH_REPORT
    report_sections: List[Dict[str, Any]] = []
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
