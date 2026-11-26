"""Chat history management and MinIO S3 persistence engine.

Provides user-isolated session lifecycle tracking, conversation appending, and
dual-layer persistence: bounded TTL LRU caching and MinIO S3 object storage
(bucket: cfi-app-state, key: sessions/{user_id}/{session_id}.json).
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from backend.contracts import (
    ChatMessage,
    ChatSession,
    ChatSessionSummary,
)
from backend.core.security import validate_user_id
from backend.core.storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
BENCHMARK_SESSION_PREFIXES: tuple[str, ...] = (
    "prod50-",
    "test-session-",
    "adversarial-",
    "edge-50-",
)


def is_benchmark_session_id(session_id: str) -> bool:
    """Return True for automated benchmark/test session IDs that must not pollute user history."""
    sid = (session_id or "").strip().lower()
    return any(sid.startswith(prefix) for prefix in BENCHMARK_SESSION_PREFIXES)


def validate_session_id(session_id: str) -> str:
    """Ensure session_id is safe for S3 object keys and free of path traversal."""
    if not session_id or not SESSION_ID_RE.match(session_id):
        raise ValueError("Invalid session_id: must be 1-64 alphanumeric, hyphen, or underscore characters.")
    return session_id


class ChatHistoryManager:
    """Manages per-user chat sessions with thread-safe TTL LRU cache and MinIO S3 synchronization."""

    BUCKET_NAME = "cfi-app-state"
    KEY_PREFIX = "sessions"
    MAX_CACHED_SESSIONS = 500
    DEFAULT_CACHE_TTL_SECONDS = 60.0
    DEFAULT_USER_ID = "default_user"

    def __init__(
        self,
        s3_storage: S3Storage | None = None,
        max_sessions: int = MAX_CACHED_SESSIONS,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Initialize chat history manager with optional injected S3Storage adapter.

        Args:
            s3_storage: Optional S3Storage instance. If None, instantiates default.
            max_sessions: Maximum number of sessions kept in local memory cache.
            cache_ttl_seconds: Freshness window before refreshing cached sessions from S3.
        """
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._session_cached_at: dict[str, float] = {}
        self._max_sessions = max(1, int(max_sessions))
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._lock = threading.RLock()
        self._s3: S3Storage | None = s3_storage
        self._s3_enabled: bool = (
            True if s3_storage is not None else os.getenv("ENABLE_S3_CHAT_HISTORY", "true").lower() == "true"
        )

        if self._s3 is None and self._s3_enabled:
            self._init_s3()

    @staticmethod
    def _cache_key(user_id: str, session_id: str) -> str:
        uid = validate_user_id(user_id)
        sid = validate_session_id(session_id)
        return f"{uid}:{sid}"

    def _s3_key(self, user_id: str, session_id: str) -> str:
        uid = validate_user_id(user_id)
        sid = validate_session_id(session_id)
        return f"{self.KEY_PREFIX}/{uid}/{sid}.json"

    def _is_cache_fresh_locked(self, c_key: str, s3_active: bool) -> bool:
        """Return True if cached session is valid (memory-only never expires; S3 cache uses TTL)."""
        if c_key not in self._sessions:
            return False
        if not s3_active:
            return True
        age = time.monotonic() - self._session_cached_at.get(c_key, 0.0)
        return age < self._cache_ttl_seconds

    def _put_cached_session(self, user_id: str, session_id: str, session: ChatSession) -> None:
        """Store session in bounded LRU OrderedDict under lock."""
        c_key = self._cache_key(user_id, session_id)
        with self._lock:
            self._sessions[c_key] = session
            self._session_cached_at[c_key] = time.monotonic()
            self._sessions.move_to_end(c_key)
            while len(self._sessions) > self._max_sessions:
                evicted_key, _ = self._sessions.popitem(last=False)
                self._session_cached_at.pop(evicted_key, None)

    def _init_s3(self) -> S3Storage | None:
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

    def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> ChatSession:
        """Retrieve an existing user-scoped session or initialize a new one."""
        uid = validate_user_id(user_id)
        s3 = self._init_s3()
        s3_active = self._s3_enabled and s3 is not None

        if session_id is not None:
            validate_session_id(session_id)
            c_key = self._cache_key(uid, session_id)
            with self._lock:
                if self._is_cache_fresh_locked(c_key, s3_active):
                    self._sessions.move_to_end(c_key)
                    return self._sessions[c_key]

            if s3_active and not is_benchmark_session_id(session_id):
                persisted = self._load_from_s3(session_id, user_id=uid)
                if persisted:
                    self._put_cached_session(uid, session_id, persisted)
                    return persisted

            # Preserve unpersisted in-memory empty session if S3 has nothing yet
            with self._lock:
                if c_key in self._sessions:
                    self._session_cached_at[c_key] = time.monotonic()
                    self._sessions.move_to_end(c_key)
                    return self._sessions[c_key]

        # Create brand-new session (lazy persistence: only saved to S3 when first message is added)
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        session = ChatSession(
            session_id=new_id,
            title="New Research Session",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self._put_cached_session(uid, new_id, session)
        return session

    def add_user_message(
        self,
        session_id: str,
        content: str,
        user_id: str = DEFAULT_USER_ID,
    ) -> ChatMessage:
        """Append a user message to the user-scoped session atomically under lock."""
        uid = validate_user_id(user_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        msg = ChatMessage(
            message_id=msg_id,
            role="user",
            content=content,
            timestamp=now,
        )

        with self._lock:
            session = self.get_or_create_session(session_id, user_id=uid)
            if session.title == "New Research Session" and content.strip():
                cleaned = content.strip().replace("\n", " ")
                session.title = cleaned[:60] + "..." if len(cleaned) > 60 else cleaned
            session.messages.append(msg)
            session.updated_at = now
            self._put_cached_session(uid, session.session_id, session)
            snapshot = session.model_copy(deep=True)

        self._persist_to_s3(snapshot, user_id=uid)
        return msg

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> ChatMessage:
        """Append an assistant response to the user-scoped session atomically under lock."""
        uid = validate_user_id(user_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        msg = ChatMessage(
            message_id=msg_id,
            role="assistant",
            content=content,
            timestamp=now,
            metadata=metadata,
        )

        with self._lock:
            session = self.get_or_create_session(session_id, user_id=uid)
            session.messages.append(msg)
            session.updated_at = now
            self._put_cached_session(uid, session.session_id, session)
            snapshot = session.model_copy(deep=True)

        self._persist_to_s3(snapshot, user_id=uid)
        return msg

    def get_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> ChatSession | None:
        """Fetch user-scoped session by ID with TTL cache and lazy loading from S3."""
        uid = validate_user_id(user_id)
        validate_session_id(session_id)
        c_key = self._cache_key(uid, session_id)
        s3 = self._init_s3()
        s3_active = self._s3_enabled and s3 is not None

        with self._lock:
            if self._is_cache_fresh_locked(c_key, s3_active):
                self._sessions.move_to_end(c_key)
                return self._sessions[c_key]

        if s3_active and not is_benchmark_session_id(session_id):
            persisted = self._load_from_s3(session_id, user_id=uid)
            if persisted:
                self._put_cached_session(uid, session_id, persisted)
                return persisted

        with self._lock:
            return self._sessions.get(c_key)

    def list_sessions(self, user_id: str = DEFAULT_USER_ID) -> list[ChatSessionSummary]:
        """Return summaries of user-scoped sessions ordered by recency, excluding benchmark sessions."""
        uid = validate_user_id(user_id)
        prefix = f"{self.KEY_PREFIX}/{uid}/"
        cache_prefix = f"{uid}:"

        s3 = self._init_s3()
        if self._s3_enabled and s3 is not None:
            try:
                keys = s3.list_objects(self.BUCKET_NAME, prefix)
                active_s3_ids: set[str] = set()
                for key in keys:
                    if not key.endswith(".json"):
                        continue
                    sid = key.split("/")[-1].replace(".json", "")
                    if is_benchmark_session_id(sid):
                        continue
                    active_s3_ids.add(sid)
                    c_key = self._cache_key(uid, sid)
                    with self._lock:
                        is_fresh = self._is_cache_fresh_locked(c_key, True)
                    if not is_fresh:
                        persisted = self._load_from_s3(sid, user_id=uid)
                        if persisted:
                            self._put_cached_session(uid, sid, persisted)

                with self._lock:
                    for c_key in list(self._sessions.keys()):
                        if not c_key.startswith(cache_prefix):
                            continue
                        sid = c_key[len(cache_prefix) :]
                        if sid not in active_s3_ids and len(self._sessions[c_key].messages) > 0:
                            del self._sessions[c_key]
                            self._session_cached_at.pop(c_key, None)
            except Exception as exc:
                logger.warning("Failed to sync chat sessions from MinIO S3: %s", exc)

        summaries: list[ChatSessionSummary] = []
        with self._lock:
            user_entries = [
                (c_key[len(cache_prefix) :], sess)
                for c_key, sess in self._sessions.items()
                if c_key.startswith(cache_prefix)
            ]

        for sid, session in user_entries:
            if is_benchmark_session_id(sid):
                continue
            if len(session.messages) == 0:
                if self._s3_enabled and s3 is not None:
                    self.delete_session(session.session_id, user_id=uid)
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

        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    def delete_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Remove a user-scoped session from memory and MinIO storage."""
        uid = validate_user_id(user_id)
        validate_session_id(session_id)
        c_key = self._cache_key(uid, session_id)
        existed = False
        with self._lock:
            if c_key in self._sessions:
                del self._sessions[c_key]
                self._session_cached_at.pop(c_key, None)
                existed = True

        s3 = self._init_s3()
        if s3 is not None:
            key = self._s3_key(uid, session_id)
            try:
                if s3.object_exists(self.BUCKET_NAME, key):
                    s3.delete_object(self.BUCKET_NAME, key)
                    existed = True
            except Exception as exc:
                logger.warning("Failed to delete session %s from S3: %s", session_id, exc)

        return existed

    def _persist_to_s3(self, session: ChatSession, user_id: str = DEFAULT_USER_ID) -> None:
        """Upload serialized session JSON to MinIO S3 bucket cfi-app-state."""
        if not self._s3_enabled or not session.messages or is_benchmark_session_id(session.session_id):
            return

        s3 = self._init_s3()
        if s3 is None:
            return

        uid = validate_user_id(user_id)
        key = self._s3_key(uid, session.session_id)
        try:
            s3.ensure_buckets_exist([self.BUCKET_NAME])
            payload = session.model_dump_json(indent=2).encode("utf-8")
            # Sanitize metadata to pure ASCII to prevent S3 HTTP header validation failure on non-ASCII titles
            safe_title = session.title.encode("ascii", "ignore").decode("ascii").strip() or "Chat Session"
            s3.upload_bytes(
                bucket=self.BUCKET_NAME,
                key=key,
                data=payload,
                content_type="application/json",
                metadata={
                    "user_id": uid,
                    "session_id": session.session_id,
                    "title_ascii": safe_title[:50],
                    "message_count": str(len(session.messages)),
                    "updated_at": session.updated_at,
                },
            )
            logger.debug("Persisted chat session %s to s3://%s/%s", session.session_id, self.BUCKET_NAME, key)
        except Exception as exc:
            logger.warning("Could not persist chat session %s to MinIO S3: %s", session.session_id, exc)

    def _load_from_s3(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> ChatSession | None:
        """Download and deserialize user-scoped session JSON from MinIO S3 without redundant HEAD checks."""
        s3 = self._init_s3()
        if s3 is None:
            return None

        key = self._s3_key(user_id, session_id)
        try:
            raw_bytes = s3.download_bytes(self.BUCKET_NAME, key)
            if not isinstance(raw_bytes, (bytes, bytearray)):
                return None
            data = json.loads(raw_bytes.decode("utf-8"))
            return ChatSession(**data)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                return None
            logger.warning("Failed to retrieve chat session %s from S3: %s", session_id, exc)
            return None
        except Exception as exc:
            logger.warning("Failed to retrieve chat session %s from S3: %s", session_id, exc)
            return None
