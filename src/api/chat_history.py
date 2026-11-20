"""Chat history management and MinIO S3 persistence engine.

Provides session lifecycle tracking, conversation appending, and dual-layer persistence:
in-memory caching for zero-latency retrieval and MinIO S3 object storage
(bucket: sec-silver, key: chat_history/{session_id}.json) for immutable,
reproducible data lakehouse audit trails.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.api.contracts import (
    ChatMessage,
    ChatSession,
    ChatSessionSummary,
)
from src.data.storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat sessions with local memory caching and MinIO S3 synchronization."""

    BUCKET_NAME = "sec-silver"
    KEY_PREFIX = "chat_history"

    def __init__(self, s3_storage: Optional[S3Storage] = None) -> None:
        """Initialize chat history manager with optional injected S3Storage adapter.

        Args:
            s3_storage: Optional S3Storage instance. If None, instantiates default.
        """
        self._sessions: Dict[str, ChatSession] = {}
        self._s3: Optional[S3Storage] = s3_storage
        self._s3_enabled: bool = os.getenv("ENABLE_S3_CHAT_HISTORY", "true").lower() == "true"

        if self._s3 is None and self._s3_enabled:
            self._init_s3()

    def _init_s3(self) -> Optional[S3Storage]:
        """Lazy-initialize or retrieve the S3Storage adapter."""
        if self._s3 is not None:
            return self._s3
        if not self._s3_enabled:
            return None
        try:
            self._s3 = S3Storage()
            return self._s3
        except Exception as exc:
            logger.warning("MinIO S3Storage initialization failed (%s); operating in memory-only mode.", exc)
            self._s3 = None
            return None

    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Retrieve an existing session or initialize a new one.

        Args:
            session_id: Optional existing session UUID.

        Returns:
            Active ChatSession instance.
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from S3 if session_id is provided but not in memory
        s3 = self._init_s3()
        if session_id and s3 is not None:
            persisted = self._load_from_s3(session_id)
            if persisted:
                self._sessions[session_id] = persisted
                return persisted

        # Create brand-new session (lazy persistence: only saved to S3 when first message is added)
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        session = ChatSession(
            session_id=new_id,
            title="New Research Session",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self._sessions[new_id] = session
        return session

    def add_user_message(self, session_id: str, content: str) -> ChatMessage:
        """Append a user message to the session.

        Args:
            session_id: Target session UUID.
            content: Query string submitted by user.

        Returns:
            Constructed ChatMessage instance.
        """
        session = self.get_or_create_session(session_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Update title from first user query if still default
        if session.title == "New Research Session" and content.strip():
            # Truncate clean title
            cleaned = content.strip().replace("\n", " ")
            session.title = cleaned[:60] + "..." if len(cleaned) > 60 else cleaned

        msg = ChatMessage(
            message_id=msg_id,
            role="user",
            content=content,
            timestamp=now,
        )
        session.messages.append(msg)
        session.updated_at = now
        self._persist_to_s3(session)
        return msg

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Append an assistant response to the session.

        Args:
            session_id: Target session UUID.
            content: Synthesized answer text.
            metadata: Structured telemetry, citations, intent, or charts.

        Returns:
            Constructed ChatMessage instance.
        """
        session = self.get_or_create_session(session_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        msg = ChatMessage(
            message_id=msg_id,
            role="assistant",
            content=content,
            timestamp=now,
            metadata=metadata,
        )
        session.messages.append(msg)
        session.updated_at = now
        self._persist_to_s3(session)
        return msg

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Fetch session by ID with lazy loading from S3."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        s3 = self._init_s3()
        if s3 is not None:
            persisted = self._load_from_s3(session_id)
            if persisted:
                self._sessions[session_id] = persisted
                return persisted

        return None

    def list_sessions(self) -> List[ChatSessionSummary]:
        """Return summaries of all available sessions ordered by recency.

        Reconciles memory cache with authoritative MinIO S3 Lakehouse state,
        pruning empty or orphaned zombie files.
        """
        s3 = self._init_s3()
        if self._s3_enabled and s3 is not None:
            try:
                keys = s3.list_objects(self.BUCKET_NAME, f"{self.KEY_PREFIX}/")
                active_s3_ids = set()
                for key in keys:
                    if not key.endswith(".json"):
                        continue
                    sid = key.split("/")[-1].replace(".json", "")
                    active_s3_ids.add(sid)
                    if sid not in self._sessions:
                        persisted = self._load_from_s3(sid)
                        if persisted:
                            self._sessions[sid] = persisted

                # Reconcile: remove memory sessions that no longer exist in S3
                for mem_id in list(self._sessions.keys()):
                    if mem_id not in active_s3_ids and len(self._sessions[mem_id].messages) > 0:
                        del self._sessions[mem_id]
            except Exception as exc:
                logger.warning("Failed to sync chat sessions from MinIO S3: %s", exc)

        summaries: List[ChatSessionSummary] = []
        for session in list(self._sessions.values()):
            # Prune empty sessions from both storage and summaries
            if len(session.messages) == 0:
                if self._s3_enabled and s3 is not None:
                    self.delete_session(session.session_id)
                continue

            summaries.append(
                ChatSessionSummary(
                    session_id=session.session_id,
                    title=session.title,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=len(session.messages),
                )
            )

        # Sort descending by updated_at
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from memory and MinIO storage."""
        existed = False
        if session_id in self._sessions:
            del self._sessions[session_id]
            existed = True

        s3 = self._init_s3()
        if s3 is not None:
            key = f"{self.KEY_PREFIX}/{session_id}.json"
            try:
                if s3.object_exists(self.BUCKET_NAME, key):
                    s3.delete_object(self.BUCKET_NAME, key)
                    existed = True
            except Exception as exc:
                logger.warning("Failed to delete session %s from S3: %s", session_id, exc)

        return existed

    def _persist_to_s3(self, session: ChatSession) -> None:
        """Upload serialized session JSON to MinIO S3 bucket sec-silver."""
        if not self._s3_enabled:
            return
        # Do not persist empty sessions
        if not session.messages:
            return

        s3 = self._init_s3()
        if s3 is None:
            return

        key = f"{self.KEY_PREFIX}/{session.session_id}.json"
        try:
            payload = session.model_dump_json(indent=2).encode("utf-8")
            # Sanitize metadata to pure ASCII to prevent S3 HTTP header validation failure on non-ASCII titles
            safe_title = session.title.encode("ascii", "ignore").decode("ascii").strip() or "Chat Session"
            s3.upload_bytes(
                bucket=self.BUCKET_NAME,
                key=key,
                data=payload,
                content_type="application/json",
                metadata={
                    "session_id": session.session_id,
                    "title_ascii": safe_title[:50],
                    "message_count": str(len(session.messages)),
                    "updated_at": session.updated_at,
                },
            )
            logger.debug("Persisted chat session %s to s3://%s/%s", session.session_id, self.BUCKET_NAME, key)
        except Exception as exc:
            logger.warning("Could not persist chat session %s to MinIO S3: %s", session.session_id, exc)

    def _load_from_s3(self, session_id: str) -> Optional[ChatSession]:
        """Download and deserialize session JSON from MinIO S3 bucket sec-silver."""
        s3 = self._init_s3()
        if s3 is None:
            return None

        key = f"{self.KEY_PREFIX}/{session_id}.json"
        try:
            if not s3.object_exists(self.BUCKET_NAME, key):
                return None

            raw_bytes = s3.download_bytes(self.BUCKET_NAME, key)
            data = json.loads(raw_bytes.decode("utf-8"))
            return ChatSession(**data)
        except Exception as exc:
            logger.warning("Failed to retrieve chat session %s from S3: %s", session_id, exc)
            return None
