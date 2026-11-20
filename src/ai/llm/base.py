"""Abstract base contract for LLM adapters

Provides a provider-agnostic interface supporting synchronous generation
and real-time streaming without exposing vendor SDKs to upstream callers.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.ai.llm.contracts import ChatMessage, LLMResponse, LLMStreamChunk


class BaseLLMAdapter(ABC):
    """Abstract interface for LLM provider adapters."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Execute a synchronous completion request.

        Args:
            messages: Conversation messages history.
            temperature: Sampling temperature (ignored in reasoning mode).
            max_tokens: Maximum completion tokens to generate.
            json_mode: Whether to enforce valid JSON object output.

        Returns:
            Immutable LLMResponse containing content, reasoning, usage, and latency.

        Raises:
            LLMError: On unrecoverable request, authentication, balance, or provider failures.
        """

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream response tokens chunk-by-chunk.

        Args:
            messages: Conversation messages history.
            temperature: Sampling temperature (ignored in reasoning mode).
            max_tokens: Maximum completion tokens to generate.

        Yields:
            LLMStreamChunk instances with content deltas or final usage metrics.

        Raises:
            LLMError: On network, rate limit, or provider failures.
        """
