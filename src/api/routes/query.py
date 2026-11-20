"""Query router handling natural language research requests.

Orchestrates input validation, session management, LangGraph agent execution,
and dual-layer persistence of conversation history to MinIO S3 Lakehouse.
"""

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from src.ai.agent.research_agent import ResearchAgent
from src.api.chat_history import ChatHistoryManager
from src.api.contracts import QueryRequest, QueryResponse
from src.api.deps import get_chat_history, get_research_agent
from src.core.observability.telemetry import log_run_telemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-tool SEC research query",
    description="Processes user question through LangGraph agent (SQL, RAG, Citations, Charts) and logs to chat history.",
)
async def execute_query(
    request: QueryRequest,
    agent: ResearchAgent = Depends(get_research_agent),
    history: ChatHistoryManager = Depends(get_chat_history),
) -> QueryResponse:
    """Execute end-to-end research query with grounded citations and session persistence."""
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Query string cannot be empty or whitespace.",
        )

    # 1. Initialize or resolve session and correlation ID
    request_id = str(uuid.uuid4())
    session = history.get_or_create_session(request.session_id)
    session_id = session.session_id

    # Extract previous conversation history for contextual follow-up queries
    previous_history = [
        {"role": m.role, "content": m.content, "metadata": m.metadata}
        for m in session.messages[-6:]
    ]

    # 2. Append user turn to chat history
    history.add_user_message(session_id=session_id, content=cleaned_query)

    try:
        # 3. Invoke LangGraph ResearchAgent with conversation history
        agent_resp = agent.query(cleaned_query, conversation_history=previous_history)

        # 4. Prepare assistant response metadata for chat history
        metadata = {
            "request_id": request_id,
            "intent": agent_resp.intent.value,
            "execution_time_ms": round(agent_resp.execution_time_ms, 2),
            "citations_count": len(agent_resp.citations),
            "charts_count": len(agent_resp.charts),
            "citations": [c.model_dump() for c in agent_resp.citations],
            "charts": [ch.model_dump() for ch in agent_resp.charts],
        }

        # 5. Append assistant turn to chat history (persisted to MinIO)
        history.add_assistant_message(
            session_id=session_id,
            content=agent_resp.answer,
            metadata=metadata,
        )

        # 6. Record execution telemetry (append-only JSONL)
        plan = agent_resp.query_plan
        tickers = plan.tickers if plan else []
        metrics = plan.metrics if plan else []
        periods = [f"FY{y}" for y in plan.fiscal_years] if plan else []
        tools_called = [r.tool_name for r in agent_resp.tool_results]
        tool_status = {r.tool_name: r.status.value for r in agent_resp.tool_results}

        error_code = None
        if agent_resp.abstention and agent_resp.abstention.is_abstaining:
            error_code = (
                agent_resp.abstention.reason.value
                if agent_resp.abstention.reason
                else "ABSTENTION"
            )

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
            error_code=error_code,
        )

        # 7. Construct and return API response envelope
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

    except Exception as exc:
        logger.error("Error executing query '%s' [request_id=%s]: %s", cleaned_query, request_id, exc, exc_info=True)

        # Record error telemetry
        log_run_telemetry(
            request_id=request_id,
            session_id=session_id,
            query=cleaned_query,
            intent="ERROR",
            latency_ms=0.0,
            error_code="INTERNAL_SERVER_ERROR",
        )

        # Record error state in chat history for auditability
        history.add_assistant_message(
            session_id=session_id,
            content="An error occurred while analyzing the filing data.",
            metadata={"request_id": request_id, "error": str(exc), "status": "failed"},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed for request {request_id}. Please try again later.",
        )
