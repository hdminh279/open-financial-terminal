"""LangGraph StateGraph assembly and Tool Execution Nodes

Defines sql_node and rag_node, manages evidence extraction with SEC provenance URLs,
and houses the LangGraph workflow structure.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

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
    FinancialQueryInput,
    FinancialSQLTool,
)
from src.ai.agent.tools.web_search_tool import WebSearchTool
from src.ai.retrieval.retriever import HybridRetriever, RetrievalFilter
from src.core.financial.company_registry import get_supported_tickers, load_company_catalog
from src.core.financial.metric_registry import MetricRegistry
from src.core.financial.year_utils import extract_years_from_text, parse_fiscal_years

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Execution Nodes
# ---------------------------------------------------------------------------

_SHARED_SQL_TOOL: Optional[FinancialSQLTool] = None
_SHARED_RETRIEVER: Optional[HybridRetriever] = None
_SHARED_WEB_TOOL: Optional[WebSearchTool] = None


def get_sql_tool() -> FinancialSQLTool:
    """Retrieve or lazily initialize the shared FinancialSQLTool instance."""
    global _SHARED_SQL_TOOL
    if _SHARED_SQL_TOOL is None:
        _SHARED_SQL_TOOL = FinancialSQLTool()
    return _SHARED_SQL_TOOL


def get_retriever() -> HybridRetriever:
    """Retrieve or lazily initialize the shared HybridRetriever instance."""
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        _SHARED_RETRIEVER = HybridRetriever()
    return _SHARED_RETRIEVER


def get_web_tool() -> WebSearchTool:
    """Retrieve or lazily initialize the shared WebSearchTool instance."""
    global _SHARED_WEB_TOOL
    if _SHARED_WEB_TOOL is None:
        _SHARED_WEB_TOOL = WebSearchTool()
    return _SHARED_WEB_TOOL


def sql_node(
    state: AgentState,
    sql_tool: Optional[FinancialSQLTool] = None,
) -> Dict[str, Any]:
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

    # Graceful inference for conversational queries (follow-up turns)
    if not raw_params:
        routing_decision = state.get("routing_decision") or {}
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

            if not found_tickers:
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
        metric_reg = MetricRegistry()
        for text in [q_lower] + [(prev.get("content") or "").lower() for prev in reversed(history)]:
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
        metrics_in_params = raw_params.get("metrics") or ([raw_params.get("metric")] if raw_params.get("metric") else [])
        if not metrics_in_params:
            q_lower = (state.get("query") or "").lower()
            metric_reg = MetricRegistry()
            found_metric = None
            for syn, canon in metric_reg._synonym_index.items():
                if len(syn) >= 3 and re.search(rf"\b{re.escape(syn)}\b", q_lower):
                    found_metric = canon
                    break
            raw_params["metrics"] = [found_metric or "revenues"]

        # Multi-year handling:
        # 1. Expand year ranges or preserve explicit years so SQL and charts are scoped.
        # 2. Only unconstrained trend/compare queries without specified years fetch all periods (fiscal_years=[]).
        q_str = state.get("query") or ""
        parsed_years = parse_fiscal_years(q_str, explicit_years=raw_params.get("fiscal_years"))
        if parsed_years:
            raw_params["fiscal_years"] = parsed_years
        elif not raw_params.get("fiscal_years") and any(
            k in q_str.lower()
            for k in ["compare", "comparison", "trend", "chart", "over time"]
        ):
            raw_params["fiscal_years"] = []

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
            "activity_timeline": [{
                "step": "Querying DuckDB",
                "status": "completed",
                "description": "Missing sql_params in state",
                "duration_ms": round(elapsed_ms, 2),
            }],
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

        evidence_items: List[Dict[str, Any]] = []
        for idx, record in enumerate(response.data):
            if record.metric_value is not None:
                form_type = "10-K" if record.fiscal_period == "FY" else "10-Q"
                summary = (
                    f"{record.ticker} {record.fiscal_period} {record.fiscal_year} "
                    f"{record.metric_name}: {record.metric_value}"
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
                    source_url=record.provenance_url or "https://www.sec.gov/edgar/searchedgar/companysearch",
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
            "activity_timeline": [{
                "step": "Querying DuckDB",
                "status": "failed",
                "description": "Database query interrupted or unavailable",
                "duration_ms": round(elapsed_ms, 2),
            }],
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
            "activity_timeline": [{
                "step": "Querying DuckDB",
                "status": "failed",
                "description": "Query execution failed",
                "duration_ms": round(elapsed_ms, 2),
            }],
        }


def rag_node(
    state: AgentState,
    retriever: Optional[HybridRetriever] = None,
) -> Dict[str, Any]:
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

    search_query = (
        raw_params.get("search_query")
        if raw_params and raw_params.get("search_query")
        else original_query
    )

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
            "activity_timeline": [{
                "step": "Searching SEC filings",
                "status": "completed",
                "description": "Missing search_query in state",
                "duration_ms": round(elapsed_ms, 2),
            }],
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

        filter_params = RetrievalFilter(
            ticker=valid_tickers[0] if len(valid_tickers) == 1 else None,
            fiscal_year=rag_fy,
            item=normalized_item,
        )

        # Multi-Entity RAG Fan-Out: When comparing multiple companies (tickers >= 2), fan out per company to ensure balanced context
        if len(valid_tickers) >= 2:
            per_ticker_limit = 4 if len(valid_tickers) == 2 else max(2, 9 // len(valid_tickers))
            response = active_retriever.retrieve_multi_entity(
                query=search_query,
                tickers=valid_tickers,
                filter_params=filter_params,
                per_ticker_limit=per_ticker_limit,
                candidate_limit_per_ticker=15,
                mode="hybrid_rerank",
            )
        else:
            response = active_retriever.retrieve(
                query=search_query,
                filter_params=filter_params,
                limit=8,
                candidate_limit=20,
                mode="hybrid_rerank",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        status = (
            ToolExecutionStatus.SUCCESS
            if response.total_found > 0
            else ToolExecutionStatus.NO_DATA_FOUND
        )

        tool_result = ToolExecutionResult(
            tool_name="hybrid_retriever",
            status=status,
            data=response.model_dump(),
            execution_time_ms=elapsed_ms,
        )

        evidence_items: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(response.chunks):
            evidence = EvidenceItem(
                citation_id=idx + 1,
                source_type=EvidenceSourceType.RAG,
                ticker=chunk.ticker,
                form=chunk.form,
                fiscal_year=chunk.fiscal_year,
                fiscal_period=None,
                item_or_metric=chunk.item or "Filing Text",
                content_summary=chunk.text[:250] + "..." if len(chunk.text) > 250 else chunk.text,
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
            "activity_timeline": [{
                "step": "Searching SEC filings",
                "status": "failed",
                "description": "Filing retrieval service failed",
                "duration_ms": round(elapsed_ms, 2),
            }],
        }


def web_node(
    state: AgentState,
    web_tool: Optional[WebSearchTool] = None,
) -> Dict[str, Any]:
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

        status = (
            ToolExecutionStatus.SUCCESS
            if response.total_found > 0
            else ToolExecutionStatus.NO_DATA_FOUND
        )

        tool_result = ToolExecutionResult(
            tool_name="web_search_tool",
            status=status,
            data=response.model_dump(),
            execution_time_ms=elapsed_ms,
        )

        routing_decision = state.get("routing_decision") or {}
        tickers = routing_decision.get("tickers") or []
        primary_ticker = tickers[0] if tickers else "WEB"

        evidence_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(response.results):
            evidence = EvidenceItem(
                citation_id=idx + 1,
                source_type=EvidenceSourceType.WEB,
                ticker=primary_ticker,
                form="WEB_NEWS",
                fiscal_year=None,
                fiscal_period=None,
                item_or_metric=f"Web: {item.title}",
                content_summary=item.snippet,
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
            "activity_timeline": [{
                "step": "Searching live web",
                "status": "failed",
                "description": "Live web search service failed",
                "duration_ms": round(elapsed_ms, 2),
            }],
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
