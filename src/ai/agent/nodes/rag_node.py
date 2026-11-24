"""LangGraph RAG retrieval node querying Qdrant hybrid dense/BM25 filing chunks."""

import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

from src.ai.agent.contracts import (
    EvidenceItem,
    EvidenceSourceType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.state import AgentState
from src.ai.retrieval.retriever import HybridRetriever, RetrievalFilter
from src.core.financial.company_registry import get_company_form_map, get_supported_tickers
from src.core.financial.response_templates import get_routing_markers

logger = logging.getLogger(__name__)

_SHARED_RETRIEVER: HybridRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_retriever() -> HybridRetriever:
    """Retrieve or lazily initialize the shared HybridRetriever instance."""
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _SHARED_RETRIEVER is None:
                _SHARED_RETRIEVER = HybridRetriever()
    return _SHARED_RETRIEVER


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
    form_map = get_company_form_map()
    forms = {form_map.get(t, "10-K") for t in valid_tickers}
    if len(forms) == 1:
        return next(iter(forms))
    return None


def rag_node(
    state: AgentState,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """LangGraph node executing hybrid vector/sparse retrieval over filing text."""
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

        # Multi-entity fan-out keeps narrative context balanced when comparing multiple companies
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
