"""LangGraph StateGraph assembly and Tool Execution Nodes

Defines sql_node and rag_node, manages evidence extraction with SEC provenance URLs,
and houses the LangGraph workflow structure.
"""

import logging
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.ai.agent.abstention import abstention_node, decide_after_abstention
from src.ai.agent.citation import citation_node, synthesis_node
from src.ai.agent.contracts import (
    EvidenceItem,
    EvidenceSourceType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.router import route_next_step, router_node
from src.ai.agent.state import AgentState
from src.ai.agent.tools.duckdb_repository import DatabaseExecutionError
from src.ai.agent.tools.financial_sql_tool import (
    RATIO_METRICS,
    FinancialQueryInput,
    FinancialSQLTool,
)
from src.ai.agent.tools.web_search_tool import WebSearchTool
from src.ai.retrieval.retriever import HybridRetriever, RetrievalFilter
from src.core.financial.company_registry import get_supported_tickers, load_company_catalog
from src.core.financial.metric_registry import get_metric_registry
from src.core.financial.response_templates import get_routing_markers
from src.core.financial.year_utils import extract_years_from_text, parse_fiscal_years

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Execution Nodes
# ---------------------------------------------------------------------------

_SHARED_SQL_TOOL: FinancialSQLTool | None = None
_SHARED_RETRIEVER: HybridRetriever | None = None
_SHARED_WEB_TOOL: WebSearchTool | None = None
_COMPANY_FORM_MAP: dict[str, str] | None = None
_TOOL_INIT_LOCK = threading.Lock()


def _load_company_form_map() -> dict[str, str]:
    """Load cached mapping of ticker -> annual filing form ('10-K', '20-F', etc.)."""
    global _COMPANY_FORM_MAP
    if _COMPANY_FORM_MAP is None:
        with _TOOL_INIT_LOCK:
            if _COMPANY_FORM_MAP is None:
                _COMPANY_FORM_MAP = {
                    str(c["ticker"]).upper(): str(c.get("form_type", "10-K"))
                    for c in load_company_catalog()
                    if c.get("ticker")
                }
    return _COMPANY_FORM_MAP


def _resolve_filing_form(ticker: str, cik: str, fiscal_period: str) -> str:
    """Resolve authoritative filing form label (BCTC, 20-F, 10-K, 10-Q) for a record."""
    if cik.strip().upper().startswith("VN:"):
        return "BCTC"
    annual_form = _load_company_form_map().get(ticker.upper(), "10-K")
    # SEC Q4 figures are derived from annual 10-K / 20-F filings (no separate Q4 10-Q exists)
    if annual_form == "20-F":
        return "20-F" if fiscal_period in ("FY", "Q4") else "6-K"
    return "10-K" if fiscal_period in ("FY", "Q4") else "10-Q"


def _compact_rag_chunk_text(text: str, max_chars: int = 3000) -> str:
    """Strip blank Markdown table spacer columns and preserve full chunk text up to max_chars."""
    if not text:
        return ""
    lines = text.splitlines()
    out_lines: list[str] = []
    table_buf: list[list[str]] = []

    def _flush_table() -> None:
        if not table_buf:
            return
        max_cols = max(len(r) for r in table_buf)
        keep = [
            c
            for c in range(max_cols)
            if any(c < len(r) and r[c].strip() and not set(r[c].strip()) <= {"-"} for r in table_buf)
        ]
        if not keep:
            keep = list(range(max_cols))
        for r in table_buf:
            cells = [r[c].strip() if c < len(r) else "" for c in keep]
            # Drop completely blank spacer rows while preserving Markdown separator rows
            if any(cells):
                out_lines.append("| " + " | ".join(cells) + " |")
        table_buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            table_buf.append(cells)
        else:
            _flush_table()
            out_lines.append(line)
    _flush_table()

    compacted = "\n".join(out_lines).strip()
    if len(compacted) > max_chars:
        return compacted[:max_chars].rstrip() + "..."
    return compacted


def _infer_rag_form(
    query: str,
    raw_params: dict[str, Any] | None,
    valid_tickers: list[str],
) -> str | None:
    """Infer target SEC form filter ('10-K' or '20-F') when annual filings or risks are requested."""
    if raw_params and raw_params.get("form"):
        return str(raw_params["form"]).strip().upper()
    q_lower = (query or "").lower()
    quarterly_markers = get_routing_markers("quarterly_form_markers") or ("10-q", "6-k", "quarter")
    if any(m in q_lower for m in quarterly_markers):
        return None
    annual_markers = get_routing_markers("annual_form_markers") or (
        "10-k",
        "20-f",
        "annual report",
        "risk factor",
        "risks",
    )
    asks_annual_form = any(m in q_lower for m in annual_markers)
    if not asks_annual_form or not valid_tickers:
        return None
    form_map = _load_company_form_map()
    forms = {form_map.get(t, "10-K") for t in valid_tickers}
    if len(forms) == 1:
        return next(iter(forms))
    return None


def get_sql_tool() -> FinancialSQLTool:
    """Retrieve or lazily initialize the shared FinancialSQLTool instance."""
    global _SHARED_SQL_TOOL
    if _SHARED_SQL_TOOL is None:
        with _TOOL_INIT_LOCK:
            if _SHARED_SQL_TOOL is None:
                _SHARED_SQL_TOOL = FinancialSQLTool()
    return _SHARED_SQL_TOOL


def get_retriever() -> HybridRetriever:
    """Retrieve or lazily initialize the shared HybridRetriever instance."""
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        with _TOOL_INIT_LOCK:
            if _SHARED_RETRIEVER is None:
                _SHARED_RETRIEVER = HybridRetriever()
    return _SHARED_RETRIEVER


def get_web_tool() -> WebSearchTool:
    """Retrieve or lazily initialize the shared WebSearchTool instance."""
    global _SHARED_WEB_TOOL
    if _SHARED_WEB_TOOL is None:
        with _TOOL_INIT_LOCK:
            if _SHARED_WEB_TOOL is None:
                _SHARED_WEB_TOOL = WebSearchTool(
                    timeout=10,
                    max_retries=3,
                    cache_ttl_seconds=300,
                    enable_rss_fallback=True,
                )
    return _SHARED_WEB_TOOL


def sql_node(
    state: AgentState,
    sql_tool: FinancialSQLTool | None = None,
) -> dict[str, Any]:
    """LangGraph node executing parameterized DuckDB queries for financial metrics.

    Args:
        state: Current AgentState containing 'sql_params'.
        sql_tool: Injected or default FinancialSQLTool instance.

    Returns:
        State dictionary update containing 'sql_data', 'tool_results', and 'evidence_items'.
    """
    start_time = time.perf_counter()
    tool = sql_tool or get_sql_tool()
    raw_params = state.get("sql_params")

    # Graceful inference only for explicit tickers or confirmed follow-up turns
    if not raw_params:
        routing_decision = state.get("routing_decision") or {}
        is_followup = bool(routing_decision.get("is_followup", False))
        tickers = list(routing_decision.get("tickers") or [])
        if not tickers and routing_decision.get("ticker"):
            tickers = [routing_decision.get("ticker")]

        q_lower = (state.get("query") or "").lower()
        history = state.get("conversation_history") or []
        if not tickers:
            found_tickers = []
            company_catalog = load_company_catalog()
            for comp in company_catalog:
                tick = comp.get("ticker", "").upper()
                name = comp.get("name", "").lower()
                aliases = [a.lower() for a in comp.get("aliases", []) if isinstance(a, str)]
                check_terms = [tick.lower(), name] + aliases
                if any(re.search(rf"\b{re.escape(term)}\b", q_lower) for term in check_terms if term):
                    found_tickers.append(tick)

            # Never scan prior conversation history unless the router classified this turn as a follow-up
            if not found_tickers and is_followup:
                for prev in reversed(history):
                    prev_content = (prev.get("content") or "").lower()
                    for comp in company_catalog:
                        tick = comp.get("ticker", "").upper()
                        name = comp.get("name", "").lower()
                        aliases = [a.lower() for a in comp.get("aliases", []) if isinstance(a, str)]
                        check_terms = [tick.lower(), name] + aliases
                        if any(re.search(rf"\b{re.escape(term)}\b", prev_content) for term in check_terms if term):
                            found_tickers.append(tick)
                    if found_tickers:
                        break
            tickers = list(dict.fromkeys(found_tickers))

        inferred_metric = None
        metric_reg = get_metric_registry()
        metric_search_texts = [q_lower] + (
            [(prev.get("content") or "").lower() for prev in reversed(history)] if is_followup else []
        )
        for text in metric_search_texts:
            for syn, canon in metric_reg._synonym_index.items():
                if len(syn) >= 3 and re.search(rf"\b{re.escape(syn)}\b", text):
                    inferred_metric = canon
                    break
            if inferred_metric:
                break

        if not inferred_metric:
            inferred_metric = "revenues"

        # Check if years are present in current query
        q_str = state.get("query") or ""
        years_found = extract_years_from_text(q_str)
        target_years = list(routing_decision.get("fiscal_years") or [])
        if not target_years and routing_decision.get("fiscal_year"):
            target_years = [routing_decision.get("fiscal_year")]
        if not target_years and years_found:
            target_years = years_found

        periods = list(routing_decision.get("fiscal_periods") or [])
        if not periods and routing_decision.get("fiscal_period"):
            periods = [routing_decision.get("fiscal_period")]
        if not periods and target_years:
            periods = ["FY"]

        if tickers:
            raw_params = {
                "tickers": tickers,
                "metrics": [inferred_metric],
                "fiscal_years": target_years,
                "fiscal_periods": periods,
            }

    if raw_params:
        # Prioritize sql_metrics from query_plan if available
        query_plan = state.get("query_plan")
        if query_plan and query_plan.get("sql_metrics"):
            raw_params["metrics"] = query_plan["sql_metrics"]

        # Default metric if omitted in dictionary
        metrics_in_params = raw_params.get("metrics") or (
            [raw_params.get("metric")] if raw_params.get("metric") else []
        )
        if not metrics_in_params:
            q_lower = (state.get("query") or "").lower()
            metric_reg = get_metric_registry()
            found_metric = None
            for syn, canon in metric_reg._synonym_index.items():
                if len(syn) >= 3 and re.search(rf"\b{re.escape(syn)}\b", q_lower):
                    found_metric = canon
                    break
            raw_params["metrics"] = [found_metric or "revenues"]

        # Multi-year handling:
        # 1. Expand explicit year ranges when specified.
        # 2. Respect Router's intentional fiscal_years=[] for multi-year comparisons instead of
        #    letting single-year regex extraction clamp a relative comparison back to 1 year.
        q_str = state.get("query") or ""
        explicit_fy = raw_params.get("fiscal_years")
        if explicit_fy:
            raw_params["fiscal_years"] = parse_fiscal_years(q_str, explicit_years=explicit_fy)
        elif "fiscal_years" in raw_params and explicit_fy == []:
            regex_years = parse_fiscal_years(q_str, explicit_years=None)
            raw_params["fiscal_years"] = regex_years if len(regex_years) >= 2 else []
        else:
            parsed_years = parse_fiscal_years(q_str, explicit_years=None)
            raw_params["fiscal_years"] = parsed_years or []

    if not raw_params:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result = ToolExecutionResult(
            tool_name="financial_sql_tool",
            status=ToolExecutionStatus.NO_DATA_FOUND,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Missing sql_params in state",
        )
        return {
            "sql_data": None,
            "tool_results": [result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Querying DuckDB",
                    "status": "completed",
                    "description": "Missing sql_params in state",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }

    try:
        query_input = FinancialQueryInput(**raw_params)
        response = tool.execute(query_input)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        has_valid_values = any(r.metric_value is not None for r in response.data)
        status = (
            ToolExecutionStatus.SUCCESS
            if response.record_count > 0 and has_valid_values
            else ToolExecutionStatus.NO_DATA_FOUND
        )

        tool_result = ToolExecutionResult(
            tool_name="financial_sql_tool",
            status=status,
            data=response.model_dump(),
            execution_time_ms=elapsed_ms,
        )

        evidence_items: list[dict[str, Any]] = []
        for idx, record in enumerate(response.data):
            if record.metric_value is not None:
                form_type = _resolve_filing_form(record.ticker, record.cik, record.fiscal_period)
                date_span = ""
                if record.start_date and record.end_date:
                    date_span = f" ({record.start_date} to {record.end_date})"
                elif record.end_date:
                    date_span = f" (as of {record.end_date})"

                unit_suffix = ""
                if record.metric_name in RATIO_METRICS:
                    unit_suffix = " (ratio)"
                elif record.currency:
                    unit_suffix = f" {record.currency}"

                summary = (
                    f"{record.ticker} {record.fiscal_period} {record.fiscal_year}{date_span} "
                    f"{record.metric_name}: {record.metric_value}{unit_suffix}"
                )
                fallback_url = (
                    f"https://api-finfo.vndirect.com.vn/v4/financial_statements?q=code:{record.ticker}"
                    if record.cik.strip().upper().startswith("VN:")
                    else "https://www.sec.gov/edgar/searchedgar/companysearch"
                )
                evidence = EvidenceItem(
                    citation_id=idx + 1,
                    source_type=EvidenceSourceType.SQL,
                    ticker=record.ticker,
                    form=form_type,
                    fiscal_year=record.fiscal_year,
                    fiscal_period=record.fiscal_period,
                    item_or_metric=record.metric_name,
                    content_summary=summary,
                    source_url=record.provenance_url or fallback_url,
                    accession_number=record.source_accession_number,
                )
                evidence_items.append(evidence.model_dump())

        sql_step = {
            "step": "Querying DuckDB",
            "status": "completed",
            "description": (
                f"Retrieved {response.record_count} financial statement records"
                if status == ToolExecutionStatus.SUCCESS
                else "No financial records found in Gold marts"
            ),
            "duration_ms": round(elapsed_ms, 2),
        }

        return {
            "sql_data": response.model_dump(),
            "tool_results": [tool_result.model_dump()],
            "evidence_items": evidence_items,
            "activity_timeline": [sql_step],
        }

    except DatabaseExecutionError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Database query execution error: %s", exc, exc_info=True)
        tool_result = ToolExecutionResult(
            tool_name="financial_sql_tool",
            status=ToolExecutionStatus.TOOL_FAILED,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Database service temporarily unavailable",
        )
        return {
            "sql_data": None,
            "tool_results": [tool_result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Querying DuckDB",
                    "status": "failed",
                    "description": "Database query interrupted or unavailable",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Unexpected error executing financial_sql_tool: %s", exc, exc_info=True)
        tool_result = ToolExecutionResult(
            tool_name="financial_sql_tool",
            status=ToolExecutionStatus.TOOL_FAILED,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Financial SQL tool encountered an unexpected error",
        )
        return {
            "sql_data": None,
            "tool_results": [tool_result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Querying DuckDB",
                    "status": "failed",
                    "description": "Query execution failed",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }


def rag_node(
    state: AgentState,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """LangGraph node executing hybrid vector/sparse retrieval over filing text.

    Args:
        state: Current AgentState containing 'rag_params'.
        retriever: Injected or default HybridRetriever instance.

    Returns:
        State dictionary update containing 'rag_data', 'tool_results', and 'evidence_items'.
    """
    start_time = time.perf_counter()
    active_retriever = retriever or get_retriever()
    raw_params = state.get("rag_params")
    original_query = state.get("query", "")

    search_query = raw_params.get("search_query") if raw_params and raw_params.get("search_query") else original_query

    if not search_query:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result = ToolExecutionResult(
            tool_name="hybrid_retriever",
            status=ToolExecutionStatus.NO_DATA_FOUND,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Missing search_query in state",
        )
        return {
            "rag_data": None,
            "tool_results": [result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Searching SEC filings",
                    "status": "completed",
                    "description": "Missing search_query in state",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }

    try:
        routing_decision = state.get("routing_decision") or {}
        query_plan = state.get("query_plan") or {}
        raw_tickers = list(
            routing_decision.get("tickers")
            or query_plan.get("tickers")
            or (state.get("sql_params") or {}).get("tickers")
            or []
        )
        if not raw_tickers and raw_params and raw_params.get("ticker"):
            raw_tickers = [raw_params.get("ticker")]

        supported_tickers = get_supported_tickers()
        valid_tickers = [t.strip().upper() for t in raw_tickers if t and t.strip().upper() in supported_tickers]

        raw_item = raw_params.get("item") if raw_params else None
        normalized_item = raw_item.upper() if isinstance(raw_item, str) else None

        # Multi-Year and Zero Year Ceiling: If multiple fiscal years are in scope, do not restrict RAG to a single year
        target_years = list(
            routing_decision.get("fiscal_years")
            or query_plan.get("fiscal_years")
            or (state.get("sql_params") or {}).get("fiscal_years")
            or []
        )
        rag_fy = raw_params.get("fiscal_year") if raw_params else None
        if len(target_years) > 1:
            rag_fy = None

        min_fy: int | None = None
        if rag_fy is None:
            if target_years:
                min_fy = min(target_years)
            elif any(k in original_query.lower() for k in ("gần nhất", "mới nhất", "recent", "latest")):
                min_fy = datetime.now(UTC).year - 3

        inferred_form = _infer_rag_form(original_query, raw_params, valid_tickers)
        filter_params = RetrievalFilter(
            ticker=valid_tickers[0] if len(valid_tickers) == 1 else None,
            form=inferred_form,
            fiscal_year=rag_fy,
            min_fiscal_year=min_fy,
            item=normalized_item,
        )

        # Multi-Entity RAG Fan-Out: When comparing multiple companies (tickers >= 2), fan out per company to ensure balanced context
        if len(valid_tickers) >= 2:
            per_ticker_limit = 6 if len(valid_tickers) <= 3 else 4
            response = active_retriever.retrieve_multi_entity(
                query=search_query,
                tickers=valid_tickers,
                filter_params=filter_params,
                per_ticker_limit=per_ticker_limit,
                candidate_limit_per_ticker=25,
                mode="hybrid_rerank",
            )
            if response.total_found == 0 and (inferred_form is not None or min_fy is not None):
                fallback_filter = filter_params.model_copy(update={"form": None, "min_fiscal_year": None})
                response = active_retriever.retrieve_multi_entity(
                    query=search_query,
                    tickers=valid_tickers,
                    filter_params=fallback_filter,
                    per_ticker_limit=per_ticker_limit,
                    candidate_limit_per_ticker=25,
                    mode="hybrid_rerank",
                )
        else:
            response = active_retriever.retrieve(
                query=search_query,
                filter_params=filter_params,
                limit=10,
                candidate_limit=25,
                mode="hybrid_rerank",
            )
            if response.total_found == 0 and (inferred_form is not None or min_fy is not None):
                fallback_filter = filter_params.model_copy(update={"form": None, "min_fiscal_year": None})
                response = active_retriever.retrieve(
                    query=search_query,
                    filter_params=fallback_filter,
                    limit=10,
                    candidate_limit=25,
                    mode="hybrid_rerank",
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        status = ToolExecutionStatus.SUCCESS if response.total_found > 0 else ToolExecutionStatus.NO_DATA_FOUND

        tool_result = ToolExecutionResult(
            tool_name="hybrid_retriever",
            status=status,
            data=response.model_dump(),
            execution_time_ms=elapsed_ms,
        )

        evidence_items: list[dict[str, Any]] = []
        for idx, chunk in enumerate(response.chunks):
            evidence = EvidenceItem(
                citation_id=idx + 1,
                source_type=EvidenceSourceType.RAG,
                ticker=chunk.ticker,
                form=chunk.form,
                fiscal_year=chunk.fiscal_year,
                fiscal_period=None,
                item_or_metric=chunk.item or "Filing Text",
                content_summary=_compact_rag_chunk_text(chunk.text, max_chars=3000),
                source_url=chunk.source_url or "https://www.sec.gov/edgar/searchedgar/companysearch",
                accession_number=chunk.accession_number,
            )
            evidence_items.append(evidence.model_dump())

        rag_step = {
            "step": "Searching SEC filings",
            "status": "completed",
            "description": (
                f"Retrieved {len(response.chunks)} relevant narrative chunks"
                if status == ToolExecutionStatus.SUCCESS
                else "No matching filing disclosures located"
            ),
            "duration_ms": round(elapsed_ms, 2),
        }

        return {
            "rag_data": response.model_dump(),
            "tool_results": [tool_result.model_dump()],
            "evidence_items": evidence_items,
            "activity_timeline": [rag_step],
        }

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Error executing hybrid_retriever: %s", exc, exc_info=True)
        tool_result = ToolExecutionResult(
            tool_name="hybrid_retriever",
            status=ToolExecutionStatus.TOOL_FAILED,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Filing retrieval service failed",
        )
        return {
            "rag_data": None,
            "tool_results": [tool_result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Searching SEC filings",
                    "status": "failed",
                    "description": "Filing retrieval service failed",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }


def web_node(
    state: AgentState,
    web_tool: WebSearchTool | None = None,
) -> dict[str, Any]:
    """LangGraph node executing real-time web searches via DuckDuckGo Search.

    Args:
        state: Current AgentState containing 'web_params' or 'query'.
        web_tool: Injected or default WebSearchTool instance.

    Returns:
        State dictionary update containing 'web_data', 'tool_results', 'evidence_items', and 'activity_timeline'.
    """
    start_time = time.perf_counter()
    tool = web_tool or get_web_tool()
    raw_params = state.get("web_params") or {}
    query_str = raw_params.get("search_query") or state.get("query", "").strip()
    max_results = raw_params.get("max_results", 5)

    try:
        response = tool.search(query=query_str, max_results=max_results)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if response.error:
            status = ToolExecutionStatus.TOOL_FAILED
        elif response.total_found > 0:
            status = ToolExecutionStatus.SUCCESS
        else:
            status = ToolExecutionStatus.NO_DATA_FOUND

        tool_result = ToolExecutionResult(
            tool_name="web_search_tool",
            status=status,
            data=response.model_dump(),
            execution_time_ms=elapsed_ms,
            error_message=response.error,
        )

        routing_decision = state.get("routing_decision") or {}
        tickers = routing_decision.get("tickers") or []
        primary_ticker = tickers[0] if tickers else "WEB"

        evidence_items: list[dict[str, Any]] = []
        for idx, item in enumerate(response.results):
            summary_with_date = f"[{item.date}] {item.snippet}" if item.date else item.snippet
            evidence = EvidenceItem(
                citation_id=idx + 1,
                source_type=EvidenceSourceType.WEB,
                ticker=primary_ticker,
                form="WEB_NEWS",
                fiscal_year=None,
                fiscal_period=None,
                item_or_metric=f"Web: {item.title}",
                content_summary=summary_with_date,
                source_url=item.url,
                accession_number=None,
            )
            evidence_items.append(evidence.model_dump())

        web_step = {
            "step": "Searching live web",
            "status": "completed",
            "description": (
                f"Retrieved {response.total_found} live web result(s)"
                if status == ToolExecutionStatus.SUCCESS
                else "No matching live web results found"
            ),
            "duration_ms": round(elapsed_ms, 2),
        }

        return {
            "web_data": response.model_dump(),
            "tool_results": [tool_result.model_dump()],
            "evidence_items": evidence_items,
            "activity_timeline": [web_step],
        }

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Error executing web_search_tool: %s", exc, exc_info=True)
        tool_result = ToolExecutionResult(
            tool_name="web_search_tool",
            status=ToolExecutionStatus.TOOL_FAILED,
            data=None,
            execution_time_ms=elapsed_ms,
            error_message="Live web search service failed",
        )
        return {
            "web_data": None,
            "tool_results": [tool_result.model_dump()],
            "evidence_items": [],
            "activity_timeline": [
                {
                    "step": "Searching live web",
                    "status": "failed",
                    "description": "Live web search service failed",
                    "duration_ms": round(elapsed_ms, 2),
                }
            ],
        }


def build_agent_graph() -> Any:
    """Construct and compile the enterprise LangGraph StateGraph workflow.
    Workflow topology:
    1. START -> router_node
    2. router_node -> route_next_step (Conditional Edge):
       - 'OUT_OF_SCOPE' -> abstention_node
       - 'FINANCIAL_METRIC' -> sql_node
       - 'TEXTUAL_RESEARCH' -> rag_node
       - 'web_search_tool' / breaking news -> web_node
       - Parallel Fan-Out (e.g. ['sql_node', 'web_node'], ['sql_node', 'rag_node', 'web_node'])
    3. sql_node -> abstention_node (Fan-In)
    4. rag_node -> abstention_node (Fan-In)
    5. web_node -> abstention_node (Fan-In)
    6. abstention_node -> decide_after_abstention (Conditional Edge):
       - Full Abstention -> citation_node (short-circuits synthesis)
       - Supported / Partial -> synthesis_node
    7. synthesis_node -> citation_node
    8. citation_node -> END
    Returns:
        Compiled LangGraph application.
    """
    workflow = StateGraph(AgentState)
    # 1. Register Nodes
    workflow.add_node("router_node", router_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("web_node", web_node)
    workflow.add_node("abstention_node", abstention_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("citation_node", citation_node)
    # 2. Entry Edge
    workflow.add_edge(START, "router_node")
    # 3. Router Conditional Branching (Supports Parallel Fan-Out)
    workflow.add_conditional_edges(
        "router_node",
        route_next_step,
        ["sql_node", "rag_node", "web_node", "abstention_node"],
    )
    # 4. Tool Convergence to Abstention Gate (Fan-In)
    workflow.add_edge("sql_node", "abstention_node")
    workflow.add_edge("rag_node", "abstention_node")
    workflow.add_edge("web_node", "abstention_node")
    # 5. Abstention Gate Conditional Branching
    workflow.add_conditional_edges(
        "abstention_node",
        decide_after_abstention,
        ["synthesis_node", "citation_node"],
    )
    # 6. Synthesis -> Citation -> END
    workflow.add_edge("synthesis_node", "citation_node")
    workflow.add_edge("citation_node", END)
    return workflow.compile()
