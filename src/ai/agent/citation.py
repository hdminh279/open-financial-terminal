"""Citation, Synthesis, and Visual Chart Generation Engine for LangGraph

Implements grounded response synthesis via DeepSeek Chat profile, assigns sequential
citation IDs [1], [2], links authoritative SEC EDGAR URLs, automatically renders
Mermaid xychart-beta visualizations, and structures deep research report sections.
"""

import logging
import re
from typing import Any

from src.ai.agent.contracts import (
    Citation,
    EvidenceSourceType,
    IntentType,
)
from src.ai.agent.nodes.citation_node import citation_node, synthesis_node
from src.ai.agent.prompts import load_prompt
from src.ai.agent.state import AgentState
from src.ai.llm.contracts import ChatMessage
from src.ai.llm.factory import LLMFactory
from src.core.financial.company_registry import load_company_catalog
from src.core.financial.response_templates import detect_query_language, render_template
from src.core.financial.semantic_guardrails import FinancialOntology
from src.core.financial.year_utils import is_shifted_fiscal_period

__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "CitationEngine",
    "LLMFactory",
    "build_fiscal_period_notes",
    "build_synthesis_messages",
    "citation_node",
    "detect_insufficient_evidence",
    "sanitize_answer_formatting",
    "synthesis_node",
]

logger = logging.getLogger(__name__)

_financial_ontology: FinancialOntology | None = None


def _get_financial_ontology() -> FinancialOntology:
    global _financial_ontology
    if _financial_ontology is None:
        _financial_ontology = FinancialOntology()
    return _financial_ontology


SYNTHESIS_SYSTEM_PROMPT = load_prompt("synthesis_prompt.txt")


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
