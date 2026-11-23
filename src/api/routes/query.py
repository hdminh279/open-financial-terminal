"""Query router handling natural language research requests.

Orchestrates input validation, session management, LangGraph agent execution,
and dual-layer persistence of conversation history to MinIO S3 Lakehouse.
"""

import json
import logging
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.ai.agent.contracts import AgentResponse
from src.ai.agent.research_agent import ResearchAgent
from src.api.chat_history import ChatHistoryManager
from src.api.contracts import QueryRequest, QueryResponse
from src.api.deps import get_chat_history, get_research_agent
from src.api.security import get_current_user_id
from src.core.observability.telemetry import log_run_telemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


def _finalize_query_turn(
    *,
    cleaned_query: str,
    request_id: str,
    session_id: str,
    user_id: str,
    agent_resp: AgentResponse,
    history: ChatHistoryManager,
) -> QueryResponse:
    """Persist assistant turn, record run telemetry, and build QueryResponse."""
    metadata = {
        "request_id": request_id,
        "intent": agent_resp.intent.value,
        "execution_time_ms": round(agent_resp.execution_time_ms, 2),
        "citations_count": len(agent_resp.citations),
        "charts_count": len(agent_resp.charts),
        "citations": [c.model_dump() for c in agent_resp.citations],
        "charts": [ch.model_dump() for ch in agent_resp.charts],
        "activity_timeline": [s.model_dump() for s in agent_resp.activity_timeline],
    }
    history.add_assistant_message(
        session_id=session_id,
        content=agent_resp.answer,
        metadata=metadata,
        user_id=user_id,
    )

    plan = agent_resp.query_plan
    tickers = plan.tickers if plan else []
    metrics = plan.metrics if plan else []
    periods = [f"FY{y}" for y in plan.fiscal_years] if plan else []
    tools_called = [r.tool_name for r in agent_resp.tool_results]
    tool_status = {r.tool_name: r.status.value for r in agent_resp.tool_results}

    error_code = None
    if agent_resp.abstention and agent_resp.abstention.is_abstaining:
        error_code = agent_resp.abstention.reason.value if agent_resp.abstention.reason else "ABSTENTION"

    usage = agent_resp.token_usage
    log_run_telemetry(
        request_id=request_id,
        session_id=session_id,
        query=cleaned_query,
        intent=agent_resp.intent.value,
        latency_ms=agent_resp.execution_time_ms,
        tickers=tickers,
        metrics=metrics,
        periods=periods,
        tools_called=tools_called,
        tool_status=tool_status,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cache_hit_tokens=usage.cache_hit_tokens if usage else 0,
        estimated_cost_usd=usage.estimated_cost_usd if usage else 0.0,
        error_code=error_code,
    )

    return QueryResponse(
        query=cleaned_query,
        session_id=session_id,
        intent=agent_resp.intent,
        answer=agent_resp.answer,
        citations=agent_resp.citations,
        charts=agent_resp.charts,
        report_sections=agent_resp.report_sections,
        abstention=agent_resp.abstention,
        tool_results=agent_resp.tool_results,
        request_id=request_id,
        activity_timeline=agent_resp.activity_timeline,
        execution_time_ms=agent_resp.execution_time_ms,
    )


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-tool SEC research query",
    description="Processes user question through LangGraph agent (SQL, RAG, Citations, Charts) and logs to chat history.",
)
def execute_query(
    request: QueryRequest,
    agent: ResearchAgent = Depends(get_research_agent),
    history: ChatHistoryManager = Depends(get_chat_history),
    user_id: str = Depends(get_current_user_id),
) -> QueryResponse:
    """Execute end-to-end research query with grounded citations and session persistence."""
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Query string cannot be empty or whitespace.",
        )

    request_id = str(uuid.uuid4())
    session = history.get_or_create_session(request.session_id, user_id=user_id)
    session_id = session.session_id
    previous_history = [{"role": m.role, "content": m.content, "metadata": m.metadata} for m in session.messages[-6:]]
    history.add_user_message(session_id=session_id, content=cleaned_query, user_id=user_id)

    try:
        agent_resp = agent.query(cleaned_query, conversation_history=previous_history)
        return _finalize_query_turn(
            cleaned_query=cleaned_query,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            agent_resp=agent_resp,
            history=history,
        )
    except Exception as exc:
        logger.error("Error executing query '%s' [request_id=%s]: %s", cleaned_query, request_id, exc, exc_info=True)
        log_run_telemetry(
            request_id=request_id,
            session_id=session_id,
            query=cleaned_query,
            intent="ERROR",
            latency_ms=0.0,
            error_code="INTERNAL_SERVER_ERROR",
        )
        history.add_assistant_message(
            session_id=session_id,
            content="An error occurred while analyzing the filing data.",
            metadata={"request_id": request_id, "error": str(exc), "status": "failed"},
            user_id=user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed for request {request_id}. Please try again later.",
        ) from exc


def _sse_frame(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream multi-tool SEC research query via Server-Sent Events (SSE)",
    description="Streams LangGraph node status events, live synthesis token deltas, and final citations/charts.",
)
def stream_query_endpoint(
    request: QueryRequest,
    agent: ResearchAgent = Depends(get_research_agent),
    history: ChatHistoryManager = Depends(get_chat_history),
    user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    """Stream end-to-end research workflow as Server-Sent Events."""
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Query string cannot be empty or whitespace.",
        )

    request_id = str(uuid.uuid4())
    session = history.get_or_create_session(request.session_id, user_id=user_id)
    session_id = session.session_id
    previous_history = [{"role": m.role, "content": m.content, "metadata": m.metadata} for m in session.messages[-6:]]
    history.add_user_message(session_id=session_id, content=cleaned_query, user_id=user_id)

    def _event_stream() -> Iterator[str]:
        try:
            for item in agent.stream_query(cleaned_query, conversation_history=previous_history):
                ev_type = item.get("event", "status")
                ev_data = item.get("data")
                if ev_type == "done" and isinstance(ev_data, AgentResponse):
                    final_payload = _finalize_query_turn(
                        cleaned_query=cleaned_query,
                        request_id=request_id,
                        session_id=session_id,
                        user_id=user_id,
                        agent_resp=ev_data,
                        history=history,
                    )
                    yield _sse_frame("done", final_payload.model_dump(mode="json"))
                else:
                    yield _sse_frame(ev_type, ev_data if isinstance(ev_data, dict) else {"value": ev_data})
        except Exception as exc:
            logger.error(
                "Error streaming query '%s' [request_id=%s]: %s", cleaned_query, request_id, exc, exc_info=True
            )
            log_run_telemetry(
                request_id=request_id,
                session_id=session_id,
                query=cleaned_query,
                intent="ERROR",
                latency_ms=0.0,
                error_code="INTERNAL_SERVER_ERROR",
            )
            history.add_assistant_message(
                session_id=session_id,
                content="An error occurred while analyzing the filing data.",
                metadata={"request_id": request_id, "error": str(exc), "status": "failed"},
                user_id=user_id,
            )
            yield _sse_frame(
                "error",
                {"detail": f"Query execution failed for request {request_id}. Please try again later."},
            )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
