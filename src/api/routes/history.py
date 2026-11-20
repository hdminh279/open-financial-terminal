"""Chat history and session persistence router.

Provides endpoints to list saved sessions, retrieve conversation turn history,
and remove archived sessions synchronized with MinIO S3 Lakehouse.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.chat_history import ChatHistoryManager
from src.api.contracts import ChatSession, ChatSessionListResponse
from src.api.deps import get_chat_history

router = APIRouter(prefix="/history", tags=["Chat History"])


@router.get(
    "",
    response_model=ChatSessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List saved chat sessions",
    description="Returns chronological summary list of all active and archived research sessions.",
)
async def list_chat_sessions(
    history: ChatHistoryManager = Depends(get_chat_history),
) -> ChatSessionListResponse:
    """Return all conversation sessions."""
    summaries = history.list_sessions()
    return ChatSessionListResponse(sessions=summaries)


@router.get(
    "/{session_id}",
    response_model=ChatSession,
    status_code=status.HTTP_200_OK,
    summary="Get conversation session detail",
    description="Retrieves full conversation history with message payloads, telemetry, and citations.",
)
async def get_chat_session(
    session_id: str,
    history: ChatHistoryManager = Depends(get_chat_history),
) -> ChatSession:
    """Return specific session or 404 if not found."""
    session = history.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return session


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete conversation session",
    description="Removes session from memory and deletes backing JSON object from MinIO S3 bucket sec-silver.",
)
async def delete_chat_session(
    session_id: str,
    history: ChatHistoryManager = Depends(get_chat_history),
) -> dict:
    """Remove conversation session."""
    deleted = history.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return {"status": "deleted", "session_id": session_id}
