"""Citation, Synthesis, and Visual Chart Generation Engine for LangGraph

Implements grounded response synthesis via DeepSeek Chat profile, assigns sequential
citation IDs [1], [2], links authoritative SEC EDGAR URLs, automatically renders
Mermaid xychart-beta visualizations, and structures deep research report sections.
"""

import logging
import re
from typing import Any

from src.ai.agent.chart_engine import ChartEngine
from src.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    Citation,
    EvidenceSourceType,
    IntentType,
    ReportSection,
)
from src.ai.agent.state import AgentState
from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.contracts import ChatMessage
from src.ai.llm.factory import LLMFactory
from src.core.financial.company_registry import load_company_catalog
from src.core.financial.response_templates import detect_query_language, render_template
from src.core.financial.semantic_guardrails import FinancialOntology
from src.core.financial.year_utils import is_shifted_fiscal_period

logger = logging.getLogger(__name__)

_financial_ontology: FinancialOntology | None = None


def _get_financial_ontology() -> FinancialOntology:
    global _financial_ontology
    if _financial_ontology is None:
        _financial_ontology = FinancialOntology()
    return _financial_ontology


SYNTHESIS_SYSTEM_PROMPT = """You are a senior financial research analyst and corporate intelligence specialist.
Your mission is to synthesize a factual, clear, and comprehensive answer to the user's question,
grounded strictly and objectively in the provided verified evidence (including official SEC EDGAR filings, VAS financial statements, and live web sources).

### Financial Semantic Guardrails:
1. Layer 1 (GAAP / VAS vs Non-GAAP):
   - Clearly distinguish official GAAP/VAS reported metrics (Revenues, Net Income, Operating Income) versus segment, non-GAAP, or news disclosures.
2. Layer 2 (Financial Ontology - Geographic vs Product Lines):
   - Strictly separate Geographic Segments (e.g. Americas, Europe, Greater China) from Product Categories (e.g. iPhone, Mac, Services, Azure).
   - NEVER confuse, mix, or aggregate Geographic revenue with Product Line revenue.
3. Layer 3 (Two-Tier Structured Organization):
   Your answer should be well-organized into two clear sections:
   - `### Official Disclosures`:
     * Core factual figures, standard Markdown tables with pipes (|), and inline citations [1], [2].
   - `### Analysis & Commentary`:
     * Growth rates, margins, business dynamics, and strategic commentary directly backed by the evidence.
4. Layer 4 (Mathematical Consistency):
   - Operating Margin = Operating Income / Revenues.
   - Net Margin = Net Income / Revenues.
   - Never hallucinate margins or figures that contradict the raw evidence.
5. Layer 5 (Multi-Currency & Fiscal Calendar Integrity):
   - Always label the exact reporting currency (e.g. USD, VND, TWD, EUR, INR, JPY) alongside monetary figures.
   - NEVER compare raw monetary figures across different currencies as if they shared the same unit, and NEVER invent exchange rates not present in the evidence.
   - When evidence includes exact period dates (start_date to end_date), state the exact fiscal period and date coverage clearly to avoid calendar-year vs fiscal-year confusion.
   - Clearly separate audited filing facts from unaudited live web news sources when both are present.

### Language Enforcement:
- If the user's inquiry is in Vietnamese (or another language), synthesize the entire answer fluently in that language, translating the two-tier section headers naturally without square brackets (e.g. `### Công bố Chính thức & Bằng chứng`, `### Phân tích & Bình luận`).
- For English queries, use:
  * `### Official Disclosures`
  * `### Analysis & Commentary`

### Markdown Table Formatting Rule:
- When presenting multiple fiscal periods, product categories, or metric comparisons, format the data as a clean, standard GitHub Flavored Markdown Table with pipe syntax:
  | Fiscal Period / Product Line | Value | Citation Source |
  | :--- | :---: | :---: |
  | <Period_or_Segment_1> | <Amount_1> | [1] |
  | <Period_or_Segment_2> | <Amount_2> | [2] |
- NEVER output unformatted text blocks where column names and values are separated by spaces, tabs, or lone newlines. Always use markdown table pipes (|).

### Clean Content Rules:
- Do NOT output raw Mermaid `xychart-beta` blocks or raw `### Sources & References` footnote sections in the body. The UI automatically displays interactive charts and collapsible references from structured state.
- NEVER insert horizontal divider lines (`---`, `***`, `___`) between sections.
- NEVER wrap section headings in square brackets `[...]`.
- Minimize parentheses `(...)` as much as possible across prose, bullet points, and table cells. Write years, currencies, and clarifications naturally inline using colons, commas, or prepositions (e.g. `từ 60,9 tỷ USD năm FY2024 lên 215,9 tỷ USD năm FY2026`, `khoảng 15–18%/năm`, `478,8 tỷ USD trong FY2026`) rather than wrapping them in parentheses.
- Use consistent decimal and thousand separators matching the response language across all figures in the same answer (e.g. in Vietnamese, consistently use `.` for thousands and `,` for decimals; never mix `201.183 tỷ` with `245,122 tỷ`).
- Keep the analytical answer focused, clear, and easy to read. Do not use graphical emojis.

### Grounding & Citation Rules:
1. Strict Grounding: Rely ONLY on the facts, numerical metrics, and text excerpts in the provided evidence.
   Do not speculate, extrapolate, or inject outside knowledge. If the retrieved filing excerpts contain NO disclosure supporting the requested topic or claim, start your response with the exact tag `[INSUFFICIENT_EVIDENCE]` on the first line followed by a transparent explanation of what is missing. If a Yes/No question asks about a missing breakdown (e.g. geographic revenue split when only consolidated revenue exists), NEVER answer 'Yes' or 'No'; state transparently that it cannot be confirmed from the provided evidence.
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
    def build_citations(evidence_dicts: list[dict[str, Any]]) -> list[Citation]:
        """Convert accumulated evidence items into verified sequential Citation objects.

        Args:
            evidence_dicts: Raw evidence item dictionaries from state.

        Returns:
            List of sequential, validated Citation models.
        """
        citations: list[Citation] = []
        for idx, ev_dict in enumerate(evidence_dicts):
            citation_id = idx + 1
            raw_source_type = ev_dict.get("source_type")
            source_type = (
                raw_source_type.value
                if hasattr(raw_source_type, "value")
                else str(raw_source_type)
                if raw_source_type
                else None
            )
            ticker = ev_dict.get("ticker", "SEC")
            form = ev_dict.get("form", "Filing")
            fy = ev_dict.get("fiscal_year")
            fp = ev_dict.get("fiscal_period")
            period_str = f"({fp} {fy})" if (fp and fy) else f"(FY{fy})" if fy else ""
            item_or_metric = ev_dict.get("item_or_metric", "Filing Reference")
            source_url = ev_dict.get("source_url") or "https://www.sec.gov/edgar/searchedgar/companysearch"

            if source_type == EvidenceSourceType.WEB.value or form == "WEB_NEWS":
                formatted_ref = f"[{citation_id}] {item_or_metric} — [Live Web Source]({source_url})"
            elif (
                form in ("BCTC", "VAS", "VAS-BANK", "VAS-SECURITIES", "VAS-INSURANCE")
                or "vndirect" in source_url
                or "vietstock" in source_url
            ):
                formatted_ref = f"[{citation_id}] {ticker} BCTC {period_str} {item_or_metric} — [Financial Statement Source]({source_url})"
            else:
                formatted_ref = (
                    f"[{citation_id}] {ticker} Form {form} {period_str} "
                    f"{item_or_metric} — [SEC EDGAR Source]({source_url})"
                )

            citations.append(
                Citation(
                    citation_id=citation_id,
                    formatted_reference=formatted_ref,
                    source_url=source_url,
                    ticker=ticker,
                    form=form,
                    fiscal_year=int(fy) if fy is not None else None,
                    fiscal_period=str(fp) if fp else None,
                    item_or_metric=item_or_metric,
                    excerpt=ev_dict.get("content_summary") or None,
                    accession_number=ev_dict.get("accession_number") or None,
                    source_type=source_type,
                )
            )

        return citations

    @staticmethod
    def format_references_markdown(citations: list[Citation], draft_answer: str = "") -> str:
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


def build_fiscal_period_notes(
    sql_data: dict[str, Any] | None,
    query: str = "",
) -> list[str]:
    """Generate deterministic localized footnotes for shifted fiscal-year periods.

    Inspects returned SQL metric records and emits at most one explanatory footnote
    per company ticker when the reported period falls predominantly (>= 75% of days)
    in a calendar year different from its SEC/VAS fiscal year label.

    Args:
        sql_data: Serialized FinancialQueryResponse dictionary from AgentState.
        query: Original user query used to select response language (VI/EN).

    Returns:
        List of formatted Markdown footnote strings (one per affected ticker).
    """
    if not sql_data or not isinstance(sql_data, dict):
        return []

    records = sql_data.get("data") or []
    if not records:
        return []

    # Track the latest shifted period per ticker to keep footnotes concise
    shifted_by_ticker: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict) or rec.get("metric_value") is None:
            continue
        ticker = str(rec.get("ticker") or "").strip().upper()
        fy = rec.get("fiscal_year")
        start_date = rec.get("start_date")
        end_date = rec.get("end_date")
        if not ticker or fy is None or not start_date or not end_date:
            continue

        is_shifted, dom_year = is_shifted_fiscal_period(
            fiscal_year=int(fy),
            start_date=start_date,
            end_date=end_date,
        )
        if not is_shifted or dom_year is None:
            continue

        end_str = str(end_date)[:10]
        prev = shifted_by_ticker.get(ticker)
        if prev is None or end_str > prev["end_date"]:
            shifted_by_ticker[ticker] = {
                "ticker": ticker,
                "fiscal_year": int(fy),
                "fiscal_period": str(rec.get("fiscal_period") or "FY").upper(),
                "calendar_year": dom_year,
                "start_date": str(start_date)[:10],
                "end_date": end_str,
            }

    if not shifted_by_ticker:
        return []

    catalog = load_company_catalog()
    name_map = {
        str(c.get("ticker", "")).upper(): str(c.get("name", c.get("ticker", ""))) for c in catalog if c.get("ticker")
    }
    lang = detect_query_language(query)

    notes: list[str] = []
    for ticker in sorted(shifted_by_ticker.keys()):
        info = shifted_by_ticker[ticker]
        fp = info["fiscal_period"]
        fy = info["fiscal_year"]
        period_label = f"FY{fy}" if fp == "FY" else f"{fp} FY{fy}"
        note = render_template(
            "fiscal_period_footnote",
            lang=lang,
            company_name=name_map.get(ticker, ticker),
            ticker=ticker,
            period_label=period_label,
            calendar_year=info["calendar_year"],
            start_date=info["start_date"],
            end_date=info["end_date"],
        )
        if note:
            notes.append(note)

    return notes


_INSUFFICIENT_TAG_RE = re.compile(r"^\s*\[INSUFFICIENT_EVIDENCE\]\s*", re.IGNORECASE)
_NEGATIVE_DISCLOSURE_RE = re.compile(
    r"(?:contains?\s+no\s+(?:mention|disclosure|reference|information|record)"
    r"|does\s+not\s+(?:contain|disclose|mention|include)\s+(?:any|a)"
    r"|do\s+not\s+(?:contain|disclose|mention|include)\s+(?:any|a)"
    r"|there\s+(?:is|are)\s+no\s+(?:disclosure|mention|reference|information)"
    r"|no\s+disclosure\s+of\b"
    r"|không\s+(?:có|tìm\s+thấy)\s+(?:thông\s+tin|thuyết\s+minh|tài\s+liệu|bằng\s+chứng)"
    r"|không\s+đề\s+cập\s+(?:đến|tới))",
    re.IGNORECASE,
)


def detect_insufficient_evidence(
    answer: str,
    is_already_abstaining: bool = False,
    is_partial: bool = False,
    has_sql_success: bool = False,
) -> tuple[bool, str]:
    """Detect synthesis-level abstention when retrieved filing chunks lack requested disclosures."""
    if not answer:
        return False, ""

    has_tag = bool(_INSUFFICIENT_TAG_RE.match(answer))
    cleaned = _INSUFFICIENT_TAG_RE.sub("", answer).strip()

    if is_already_abstaining or is_partial:
        return False, cleaned

    # Keep citations intact when the answer contains substantive disclosures
    # (successful SQL metrics, Markdown tables, or multi-paragraph cited analysis).
    has_substantive_content = has_sql_success or "|---" in cleaned or "| :---" in cleaned or len(cleaned) >= 1200
    if has_substantive_content:
        return False, cleaned

    if has_tag:
        return True, cleaned

    # Fallback phrase check for short RAG-only answers that explicitly report zero matching disclosures
    if _NEGATIVE_DISCLOSURE_RE.search(cleaned):
        return True, cleaned

    return False, cleaned


def sanitize_answer_formatting(text: str) -> str:
    """Remove horizontal dividers and square brackets around section headings."""
    if not text:
        return ""
    cleaned = _INSUFFICIENT_TAG_RE.sub("", text)
    cleaned = re.sub(r"(?m)^\s*(?:---+|\*\*\*+|___+)\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^(#{1,4}\s+)\[([^\]]+)\]\s*$", r"\1\2", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_synthesis_messages(state: AgentState) -> list[ChatMessage]:
    """Construct system and user messages for response synthesis from AgentState."""
    query = state.get("query", "")
    intent_str = state.get("intent", IntentType.FINANCIAL_METRIC.value)
    evidence_dicts = state.get("evidence_items", [])
    abstention_info = state.get("abstention_info") or {}
    is_partial = state.get("is_partial", False)

    evidence_text_blocks: list[str] = []
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

    ontology = _get_financial_ontology()
    routing_decision = state.get("routing_decision") or {}
    primary_ticker = (routing_decision.get("tickers") or [None])[0]
    disambiguation = ontology.disambiguate_query(query, ticker=primary_ticker)
    ontology_note = f"\n\nONTOLOGY NOTE: {disambiguation['guidance']}" if disambiguation.get("guidance") else ""

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
            "\n\n=== EXCLUDED / UNPUBLISHED PERIODS ===\n"
            "The following requested periods cannot be reported from verified SEC filings:\n"
            + "\n".join(unavail_items)
            + "\nExplicitly disclose in your answer why these periods are excluded (e.g. fiscal year has not ended or Form 10-K is not published yet).\n"
        )

    query_lang = detect_query_language(query)
    lang_directive = (
        "Response Language: VIETNAMESE (write 100% in Vietnamese, including headers '### Công bố Chính thức & Bằng chứng' and '### Phân tích & Bình luận', with no square brackets, no --- dividers, and minimal parentheses)."
        if query_lang == "vi"
        else "Response Language: ENGLISH (write 100% in English, including headers '### Official Disclosures' and '### Analysis & Commentary', with no square brackets, no --- dividers, and minimal parentheses)."
    )

    user_prompt = (
        f"User Inquiry: {query}\n"
        f"Detected Intent: {intent_str}\n"
        f"{lang_directive}\n"
        f"{partial_note}{ontology_note}{unavail_note}\n\n"
        f"=== COLLECTED EVIDENCE FROM SEC FILINGS ===\n"
        f"{formatted_context}\n\n"
        f"Please provide a grounded, two-tier structured answer adhering strictly to citation and guardrail rules."
    )

    return [
        ChatMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


# ---------------------------------------------------------------------------
# LangGraph Node Functions
# ---------------------------------------------------------------------------


def synthesis_node(
    state: AgentState,
    llm_adapter: BaseLLMAdapter | None = None,
) -> dict[str, Any]:
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
    """LangGraph node appending verified references, generating visual charts, and parsing report sections.

    Args:
        state: Current AgentState.

    Returns:
        State dictionary update containing 'citations', 'final_answer', 'charts', and 'report_sections'.
    """
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
    tickers = list(routing_decision.get("tickers") or [])
    if not tickers and routing_decision.get("ticker"):
        tickers = [routing_decision.get("ticker")]
    ticker_label = " vs ".join(tickers) if tickers else None
    intent_str = state.get("intent", "")

    query = state.get("query", "")

    # 1. Build Citations
    citations = CitationEngine.build_citations(evidence_dicts)

    # 2. Extract & Build Charts with Router preference
    query_plan = state.get("query_plan") or {}
    chart_pref = query_plan.get("chart_preference") or routing_decision.get("chart_preference") or "none"
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

    # 3b. Deterministically append localized fiscal-calendar footnotes when periods are shifted
    fiscal_notes = build_fiscal_period_notes(sql_data=sql_data, query=query)
    if fiscal_notes:
        missing_notes = [n for n in fiscal_notes if n not in final_answer]
        if missing_notes:
            final_answer = f"{final_answer}\n\n" + "\n".join(missing_notes)

    # 4. Parse Report Sections if DEEP_RESEARCH_REPORT
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
