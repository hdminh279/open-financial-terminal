"""DeepSeek LLM Adapter implementation using OpenAI-compatible SDK

Provides resilient generation and streaming via Tenacity retries, strict parameter
filtering across thinking and non-thinking profiles, and zero-leakage security.
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

import tenacity
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.contracts import (
    ChatMessage,
    LLMAuthError,
    LLMError,
    LLMInsufficientBalanceError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    """Determine whether an exception qualifies for exponential retry.

    Retryable errors:
        - Rate limit exhaustion (HTTP 429 / RateLimitError)
        - Internal server errors & overloaded state (HTTP 500, 502, 503, 504 / InternalServerError)
        - Connection drops (APIConnectionError)
        - Request timeouts (APITimeoutError)

    Fail-fast errors (never retried):
        - Authentication failure (HTTP 401 / AuthenticationError)
        - Insufficient balance (HTTP 402)
        - Bad request or unprocessable entity (HTTP 400, 422)
    """
    if isinstance(
        exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    ):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in (429, 500, 502, 503, 504)


class DeepSeekAdapter(BaseLLMAdapter):
    """Adapter for DeepSeek API using OpenAI-compatible client interface."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-flash"
    DEFAULT_TIMEOUT_SECONDS = 30.0
    DEFAULT_MAX_ATTEMPTS = 3

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_attempts: int | None = None,
        thinking: bool = False,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize DeepSeekAdapter.

        Args:
            api_key: DeepSeek API key. If omitted, loaded from DEEPSEEK_API_KEY.
            base_url: API base URL. If omitted, loaded from DEEPSEEK_BASE_URL.
            model: Model name. If omitted, loaded from LLM_MODEL.
            timeout: Request timeout in seconds. If omitted, loaded from LLM_TIMEOUT_SECONDS.
            max_attempts: TOTAL number of call attempts (including initial call).
                If omitted, loaded from LLM_MAX_ATTEMPTS (defaults to 3).
            thinking: If True, enables chain-of-thought thinking mode (reasoner profile).
                If False, thinking is disabled (chat profile).
            client: Optional pre-configured OpenAI client (useful for mocking in unit tests).

        Raises:
            LLMAuthError: If no API key is provided and client is not injected.
        """
        self.thinking = thinking
        self.model = model or os.getenv("LLM_MODEL") or self.DEFAULT_MODEL

        timeout_env = os.getenv("LLM_TIMEOUT_SECONDS")
        self.timeout = (
            timeout
            if timeout is not None
            else (float(timeout_env) if timeout_env else self.DEFAULT_TIMEOUT_SECONDS)
        )

        attempts_env = os.getenv("LLM_MAX_ATTEMPTS")
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else (int(attempts_env) if attempts_env else self.DEFAULT_MAX_ATTEMPTS)
        )

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not resolved_key or not resolved_key.strip():
                raise LLMAuthError(
                    "DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable "
                    "or pass api_key to DeepSeekAdapter."
                )
            resolved_base_url = (
                base_url or os.getenv("DEEPSEEK_BASE_URL") or self.DEFAULT_BASE_URL
            )
            # max_retries=0 avoids duplicate internal SDK retries; tenacity handles all retries
            self.client = OpenAI(
                api_key=resolved_key,
                base_url=resolved_base_url,
                timeout=self.timeout,
                max_retries=0,
            )

    def _log_retry(self, retry_state: tenacity.RetryCallState) -> None:
        """Log a warning when an execution attempt fails and enters backoff."""
        attempt = retry_state.attempt_number
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying LLM call (attempt %d/%d) due to %s: %s",
            attempt,
            self.max_attempts,
            type(exc).__name__ if exc else "UnknownException",
            exc,
        )

    def _map_exception(self, exc: Exception) -> LLMError:
        """Map SDK and network exceptions to project LLMError hierarchy."""
        if isinstance(exc, LLMError):
            return exc

        status_code = getattr(exc, "status_code", None)

        if isinstance(exc, AuthenticationError) or status_code == 401:
            return LLMAuthError(f"Authentication failed (HTTP 401): {exc}")
        if status_code == 402:
            return LLMInsufficientBalanceError(
                f"Insufficient account balance (HTTP 402): {exc}"
            )
        if isinstance(
            exc, (BadRequestError, UnprocessableEntityError)
        ) or status_code in (400, 422):
            return LLMInvalidRequestError(
                f"Invalid request ({status_code or 'bad_request'}): {exc}"
            )
        if isinstance(exc, RateLimitError) or status_code == 429:
            return LLMRateLimitError(
                f"Rate limit exceeded after {self.max_attempts} attempts: {exc}"
            )
        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError(
                f"Request timed out after {self.max_attempts} attempts: {exc}"
            )
        if isinstance(
            exc, (InternalServerError, APIConnectionError)
        ) or status_code in (500, 502, 503, 504):
            return LLMProviderError(
                f"Provider error ({status_code or 'connection'}) after {self.max_attempts} attempts: {exc}"
            )

        return LLMError(f"Unexpected LLM error: {exc}")

    def _format_message(self, msg: ChatMessage) -> dict[str, Any]:
        """Convert domain ChatMessage to OpenAI API payload format."""
        payload: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
        }
        # Multi-turn requirement: preserve assistant reasoning_content if present
        if msg.role == "assistant" and msg.reasoning_content is not None:
            payload["reasoning_content"] = msg.reasoning_content
        return payload

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Construct the request parameters dictionary adhering to verified specs."""
        if json_mode:
            has_json_kw = any("json" in (m.content or "").lower() for m in messages)
            if not has_json_kw:
                raise LLMInvalidRequestError(
                    "JSON mode requires the conversation prompt to explicitly contain the word 'json' or 'JSON'."
                )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._format_message(m) for m in messages],
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if self.thinking:
            # When thinking is enabled, temperature, presence_penalty, frequency_penalty are omitted
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            payload["temperature"] = temperature
            payload["extra_body"] = {"thinking": {"type": "disabled"}}

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _parse_usage(self, raw_usage: Any) -> TokenUsage | None:
        """Extract verified token usage fields without estimation."""
        if raw_usage is None:
            return None

        prompt_tokens = getattr(raw_usage, "prompt_tokens", 0)
        completion_tokens = getattr(raw_usage, "completion_tokens", 0)
        total_tokens = getattr(
            raw_usage, "total_tokens", prompt_tokens + completion_tokens
        )

        reasoning_tokens = None
        details = getattr(raw_usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

        cache_hit_tokens = getattr(raw_usage, "prompt_cache_hit_tokens", None)
        cache_miss_tokens = getattr(raw_usage, "prompt_cache_miss_tokens", None)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Execute synchronous completion request with retry backoff."""
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=False,
        )

        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_random_exponential(min=0.5, max=10.0),
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
            before_sleep=self._log_retry,
        )

        start_time = time.perf_counter()
        raw_response = None
        try:
            for attempt in retrying:
                with attempt:
                    raw_response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise self._map_exception(exc) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000

        if not raw_response or not raw_response.choices:
            raise LLMProviderError("Provider returned empty choices array in response.")

        choice = raw_response.choices[0]
        content = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        finish_reason = choice.finish_reason or "unknown"
        usage = self._parse_usage(raw_response.usage)

        if json_mode:
            if not content.strip():
                raise LLMError("Model returned empty content in JSON mode.")
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"Model returned malformed JSON (finish_reason='{finish_reason}'): {exc}"
                ) from exc

        logger.info(
            "LLM call completed: model=%s, latency_ms=%.1f, usage=%s",
            self.model,
            latency_ms,
            usage,
        )

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=raw_response.model or self.model,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )

    def _parse_stream_chunk(self, raw_chunk: Any) -> LLMStreamChunk | None:
        """Parse raw stream chunk into domain LLMStreamChunk or None if empty."""
        usage = self._parse_usage(getattr(raw_chunk, "usage", None))

        content_delta = ""
        reasoning_delta = None
        finish_reason = None

        choices = getattr(raw_chunk, "choices", None)
        if choices and len(choices) > 0:
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content_delta = getattr(delta, "content", None) or ""
                reasoning_delta = getattr(delta, "reasoning_content", None)

        is_final = bool(finish_reason is not None or usage is not None)

        # Skip empty chunks (no content delta, no reasoning, no finish, no usage)
        if (
            not content_delta
            and reasoning_delta is None
            and finish_reason is None
            and usage is None
        ):
            return None

        return LLMStreamChunk(
            content_delta=content_delta,
            reasoning_delta=reasoning_delta,
            is_final=is_final,
            finish_reason=finish_reason,
            usage=usage,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream completion tokens with TTFT logging and safe pre-yield retries."""
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            stream=True,
        )

        has_yielded_content = False
        start_time = time.perf_counter()
        ttft_logged = False
        attempt = 0

        while attempt < self.max_attempts:
            attempt += 1
            try:
                raw_stream = self.client.chat.completions.create(**payload)
                for raw_chunk in raw_stream:
                    chunk = self._parse_stream_chunk(raw_chunk)
                    if chunk is None:
                        continue

                    if chunk.content_delta:
                        if not ttft_logged:
                            ttft_ms = (time.perf_counter() - start_time) * 1000
                            logger.info(
                                "LLM stream TTFT: model=%s, ttft_ms=%.1f",
                                self.model,
                                ttft_ms,
                            )
                            ttft_logged = True
                        has_yielded_content = True

                    yield chunk
                # Stream completed successfully
                return

            except Exception as exc:
                if has_yielded_content:
                    # Content was already emitted to consumer: do NOT restart stream
                    logger.error(
                        "Error during streaming after content was yielded: %s", exc
                    )
                    raise self._map_exception(exc) from exc

                if not _is_retryable_error(exc) or attempt >= self.max_attempts:
                    raise self._map_exception(exc) from exc

                logger.warning(
                    "Retrying LLM stream (attempt %d/%d) due to %s: %s",
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                    exc,
                )
                # Sleep using tenacity nap sleep helper to allow zero-delay patching in tests
                delay = min(0.5 * (2 ** (attempt - 1)), 10.0)
                tenacity.nap.sleep(delay)
