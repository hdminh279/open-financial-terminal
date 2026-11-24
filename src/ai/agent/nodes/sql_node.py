"""LangGraph SQL execution node querying Gold DuckDB financial marts."""

import logging
import re
import threading
import time
from typing import Any

from src.ai.agent.contracts import (
    EvidenceItem,
    EvidenceSourceType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.state import AgentState
from src.ai.agent.tools.duckdb_repository import DatabaseExecutionError
from src.ai.agent.tools.financial_sql_tool import (
    RATIO_METRICS,
    FinancialQueryInput,
    FinancialSQLTool,
)
from src.core.financial.company_registry import load_company_catalog, resolve_filing_form
from src.core.financial.metric_registry import get_metric_registry
from src.core.financial.year_utils import extract_years_from_text, parse_fiscal_years

logger = logging.getLogger(__name__)

_SHARED_SQL_TOOL: FinancialSQLTool | None = None
_SQL_TOOL_LOCK = threading.Lock()


def get_sql_tool() -> FinancialSQLTool:
    """Retrieve or lazily initialize the shared FinancialSQLTool instance."""
    global _SHARED_SQL_TOOL
    if _SHARED_SQL_TOOL is None:
        with _SQL_TOOL_LOCK:
            if _SHARED_SQL_TOOL is None:
                _SHARED_SQL_TOOL = FinancialSQLTool()
    return _SHARED_SQL_TOOL


def sql_node(
    state: AgentState,
    sql_tool: FinancialSQLTool | None = None,
) -> dict[str, Any]:
    """LangGraph node executing parameterized DuckDB queries for financial metrics."""
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
        query_plan = state.get("query_plan")
        if query_plan and query_plan.get("sql_metrics"):
            raw_params["metrics"] = query_plan["sql_metrics"]

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

        # Expand explicit year ranges while respecting intentional fiscal_years=[] for multi-year comparisons
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
                form_type = resolve_filing_form(record.ticker, record.cik, record.fiscal_period)
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
