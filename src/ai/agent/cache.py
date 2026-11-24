"""Thread-safe bounded LRU TTL cache for standalone ResearchAgent responses."""

import re
import threading
import time
from collections import OrderedDict

from src.ai.agent.contracts import AgentResponse


def normalize_cache_key(query: str) -> str:
    """Normalize a standalone query string into a canonical cache key."""
    return re.sub(r"\s+", " ", query.strip().lower())


class TTLQueryCache:
    """Thread-safe, bounded LRU in-memory TTL cache for standalone AgentResponse objects."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 256) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, tuple[float, AgentResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> AgentResponse | None:
        """Retrieve a non-expired cached response or None on miss."""
        if self.ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, response = entry
            if now >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return response

    def set(self, key: str, response: AgentResponse) -> None:
        """Store a response in the bounded LRU cache with TTL."""
        if self.ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._store[key] = (expires_at, response)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Return current number of entries stored in cache."""
        with self._lock:
            return len(self._store)
