"""LangGraph live web search node querying DuckDuckGo / RSS feeds."""

import logging
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
from src.ai.agent.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)

_SHARED_WEB_TOOL: WebSearchTool | None = None
_WEB_TOOL_LOCK = threading.Lock()


def get_web_tool() -> WebSearchTool:
    """Retrieve or lazily initialize the shared WebSearchTool instance."""
    global _SHARED_WEB_TOOL
    if _SHARED_WEB_TOOL is None:
        with _WEB_TOOL_LOCK:
            if _SHARED_WEB_TOOL is None:
                _SHARED_WEB_TOOL = WebSearchTool(
                    timeout=10,
                    max_retries=3,
                    cache_ttl_seconds=300,
                    enable_rss_fallback=True,
                )
    return _SHARED_WEB_TOOL


def web_node(
    state: AgentState,
    web_tool: WebSearchTool | None = None,
) -> dict[str, Any]:
    """LangGraph node executing real-time web searches via DuckDuckGo Search."""
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
