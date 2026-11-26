"""Factory for instantiating and pooling LLM adapters by operational profile.

Provides a centralized construction point allowing downstream agents to request
'chat' (thinking disabled) or 'reasoner' (thinking enabled) profiles while reusing
underlying HTTP connection pools across requests.
"""

import contextlib
import threading
from typing import Any, Literal

from backend.ai.llm.base import BaseLLMAdapter
from backend.ai.llm.deepseek_adapter import DeepSeekAdapter


class LLMFactory:
    """Thread-safe factory and connection-pool cache for LLM adapters by profile."""

    VALID_PROFILES = ("chat", "reasoner")
    _ADAPTER_CACHE: dict[str, BaseLLMAdapter] = {}
    _CACHE_LOCK = threading.Lock()

    @classmethod
    def get_adapter(
        cls,
        profile: Literal["chat", "reasoner"] = "chat",
        **kwargs: Any,
    ) -> BaseLLMAdapter:
        """Retrieve a pooled adapter (when no overrides are passed) or build a custom one.

        Args:
            profile: 'chat' (thinking disabled for fast routing/extraction) or
                     'reasoner' (thinking enabled for financial reasoning).
            **kwargs: Direct parameter overrides forwarded to DeepSeekAdapter
                      (e.g. client, timeout, max_attempts). Overrides bypass the shared pool.

        Returns:
            Configured BaseLLMAdapter instance.

        Raises:
            ValueError: If an unrecognized profile name is passed.
        """
        if profile not in cls.VALID_PROFILES:
            valid_str = ", ".join(repr(p) for p in cls.VALID_PROFILES)
            raise ValueError(f"Unknown profile {profile!r}. Valid profiles are: {valid_str}.")

        if kwargs:
            return DeepSeekAdapter(thinking=(profile == "reasoner"), **kwargs)

        cached = cls._ADAPTER_CACHE.get(profile)
        if cached is not None:
            return cached

        with cls._CACHE_LOCK:
            cached = cls._ADAPTER_CACHE.get(profile)
            if cached is None:
                cached = DeepSeekAdapter(thinking=(profile == "reasoner"))
                cls._ADAPTER_CACHE[profile] = cached
            return cached

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached adapter instances without closing externally mocked clients."""
        with cls._CACHE_LOCK:
            cls._ADAPTER_CACHE.clear()

    @classmethod
    def close_all(cls) -> None:
        """Close all pooled adapter HTTP clients and clear the cache."""
        with cls._CACHE_LOCK:
            adapters = list(cls._ADAPTER_CACHE.values())
            cls._ADAPTER_CACHE.clear()
        for adapter in adapters:
            with contextlib.suppress(Exception):
                adapter.close()
