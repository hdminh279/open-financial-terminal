"""Data contracts and exception hierarchy for LLM adapters.

Provides immutable domain models for chat messages, token usage accounting,
and model responses alongside a structured exception hierarchy preserving
original causes via chained exceptions.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Represents a single message in an LLM conversation context."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"] = Field(description="Role of the message sender.")
    content: str = Field(description="Text content of the message.")
    reasoning_content: str | None = Field(
        default=None,
        description="Chain-of-thought reasoning content preserved for multi-turn assistant messages.",
    )


class TokenUsage(BaseModel):
    """Tracks token consumption and context caching metrics.

    All metrics are populated directly from the API response without estimation.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(description="Number of tokens in the prompt.")
    completion_tokens: int = Field(description="Number of tokens generated in the completion.")
    total_tokens: int = Field(description="Total token count for prompt plus completion.")
    reasoning_tokens: int | None = Field(
        default=None,
        description="Tokens consumed by internal reasoning / chain-of-thought.",
    )
    cache_hit_tokens: int | None = Field(
        default=None,
        description="Prompt tokens that hit the context cache (discounted rate).",
    )
    cache_miss_tokens: int | None = Field(
        default=None,
        description="Prompt tokens that missed the context cache.",
    )


class LLMResponse(BaseModel):
    """Encapsulates a complete non-streaming response from an LLM."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="Final generated answer content.")
    reasoning_content: str | None = Field(
        default=None,
        description="Separate chain-of-thought reasoning steps when thinking is enabled.",
    )
    model: str = Field(description="Model identifier that generated the response.")
    finish_reason: str = Field(description="Completion termination reason (e.g. 'stop', 'length').")
    usage: TokenUsage | None = Field(
        default=None,
        description="Verified token usage details if returned by the provider.",
    )
    latency_ms: float = Field(description="Total roundtrip request duration in milliseconds.")


class LLMStreamChunk(BaseModel):
    """Represents an incremental chunk emitted during streaming generation."""

    model_config = ConfigDict(frozen=True)

    content_delta: str = Field(
        default="",
        description="Incremental text generated in this streaming chunk.",
    )
    reasoning_delta: str | None = Field(
        default=None,
        description="Incremental chain-of-thought reasoning delta if thinking is enabled.",
    )
    is_final: bool = Field(
        default=False,
        description="True if this is the terminating chunk of the stream.",
    )
    finish_reason: str | None = Field(
        default=None,
        description="Termination reason emitted in the final chunk.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Cumulative token usage statistics emitted in the final stream chunk.",
    )


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM client, adapter, and provider errors."""


class LLMAuthError(LLMError):
    """Raised when authentication fails due to invalid or missing credentials (HTTP 401)."""


class LLMInsufficientBalanceError(LLMError):
    """Raised when account credit balance is exhausted (HTTP 402)."""


class LLMInvalidRequestError(LLMError):
    """Raised when request payload or parameters are rejected (HTTP 400 / 422)."""


class LLMRateLimitError(LLMError):
    """Raised when concurrency or requests-per-minute limit is exceeded (HTTP 429)."""


class LLMProviderError(LLMError):
    """Raised when upstream provider experiences server error or overload (HTTP 500 / 503)."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds configured timeout duration."""
