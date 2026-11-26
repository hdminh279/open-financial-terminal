"""Chat history and session persistence router.

Provides endpoints to list saved sessions, retrieve conversation turn history,
and remove archived sessions synchronized with MinIO S3 Lakehouse.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.contracts import ChatSession, ChatSessionListResponse
from backend.deps import get_chat_history
from backend.security import get_current_user_id
from backend.services.chat_history import ChatHistoryManager

router = APIRouter(prefix="/history", tags=["Chat History"])


@router.get(
    "",
    response_model=ChatSessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List saved chat sessions",
    description="Returns chronological summary list of all active and archived research sessions.",
)
def list_chat_sessions(
    history: ChatHistoryManager = Depends(get_chat_history),
    user_id: str = Depends(get_current_user_id),
) -> ChatSessionListResponse:
    """Return all conversation sessions for the active user."""
    summaries = history.list_sessions(user_id=user_id)
    return ChatSessionListResponse(sessions=summaries)


@router.get(
    "/{session_id}",
    response_model=ChatSession,
    status_code=status.HTTP_200_OK,
    summary="Get conversation session detail",
    description="Retrieves full conversation history with message payloads, telemetry, and citations.",
)
def get_chat_session(
    session_id: str,
    history: ChatHistoryManager = Depends(get_chat_history),
    user_id: str = Depends(get_current_user_id),
) -> ChatSession:
    """Return specific user-scoped session or 404 if not found."""
    try:
        session = history.get_session(session_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
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
    description="Removes session from memory and deletes backing JSON object from MinIO S3 bucket cfi-app-state.",
)
def delete_chat_session(
    session_id: str,
    history: ChatHistoryManager = Depends(get_chat_history),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """Remove user-scoped conversation session."""
    try:
        deleted = history.delete_session(session_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return {"status": "deleted", "session_id": session_id}
