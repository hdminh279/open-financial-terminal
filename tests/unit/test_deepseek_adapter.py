"""Unit tests for DeepSeekAdapter and LLMFactory.

Verifies 100% offline:
1. Request payload construction: model, thinking mode, parameter filtering, JSON mode.
2. Response mapping: content, finish_reason, reasoning_content separation, token usage & caching.
3. Streaming behavior: empty chunk skipping, final usage chunk yielding, TTFT logging.
4. Stream retry boundaries: retry before content vs. immediate fail without restart after content.
5. Transient error recovery: retry on 429/500/timeout with correct attempt counts.
6. Fail-fast error handling: immediate rejection on 400, 401, 402, 422.
7. Attempt exhaustion: mapping to LLMRateLimitError, LLMProviderError, LLMTimeoutError.
8. Zero-leakage security: key validation on init, no API key or prompt logged.
9. Factory profile instantiation: chat vs reasoner, rejection of invalid profiles.
"""

import json
import logging
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)

from src.ai.llm.contracts import (
    ChatMessage,
    LLMAuthError,
    LLMError,
    LLMInsufficientBalanceError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.ai.llm.deepseek_adapter import DeepSeekAdapter
from src.ai.llm.factory import LLMFactory


@pytest.fixture(autouse=True)
def patch_tenacity_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantly skip any tenacity backoff sleep delays across all tests."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr("tenacity.nap.sleep", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def make_mock_response(
    content: str = "Test response",
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
    model: str = "deepseek-flash",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int | None = None,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
    has_usage: bool = True,
) -> MagicMock:
    """Construct a mock OpenAI ChatCompletion response."""
    resp = MagicMock()
    resp.model = model

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.reasoning_content = reasoning_content
    resp.choices = [choice]

    if has_usage:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = prompt_tokens + completion_tokens
        usage.prompt_cache_hit_tokens = cache_hit_tokens
        usage.prompt_cache_miss_tokens = cache_miss_tokens
        if reasoning_tokens is not None:
            usage.completion_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)
        else:
            usage.completion_tokens_details = None
        resp.usage = usage
    else:
        resp.usage = None

    return resp


def make_mock_chunk(
    content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Construct a mock OpenAI ChatCompletionChunk for streaming."""
    chunk = MagicMock()
    chunk.usage = usage

    if content is not None or reasoning is not None or finish_reason is not None:
        choice = MagicMock()
        choice.finish_reason = finish_reason
        choice.delta = MagicMock()
        choice.delta.content = content
        choice.delta.reasoning_content = reasoning
        chunk.choices = [choice]
    else:
        chunk.choices = []

    return chunk


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


class TestPayloadConstructionAndFiltering:
    """1. Verify request payload building, thinking parameters, and JSON mode."""

    def test_chat_profile_payload(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response()
        adapter = DeepSeekAdapter(client=mock_client, thinking=False, model="deepseek-flash")

        adapter.generate(
            [ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=256,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-flash"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_reasoner_profile_payload(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response()
        adapter = DeepSeekAdapter(client=mock_client, thinking=True, model="deepseek-flash")

        adapter.generate(
            [ChatMessage(role="user", content="Analyze filing")],
            temperature=0.7,
            max_tokens=512,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-flash"
        assert call_kwargs["max_tokens"] == 512
        assert "temperature" not in call_kwargs
        assert "presence_penalty" not in call_kwargs
        assert "frequency_penalty" not in call_kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_json_mode_payload_success(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(content='{"revenue": 1000}')
        adapter = DeepSeekAdapter(client=mock_client, thinking=False)

        res = adapter.generate(
            [ChatMessage(role="user", content="Return json formatted numbers")],
            json_mode=True,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert json.loads(res.content) == {"revenue": 1000}

    def test_json_mode_missing_keyword_raises(self) -> None:
        mock_client = MagicMock()
        adapter = DeepSeekAdapter(client=mock_client)

        with pytest.raises(LLMInvalidRequestError) as exc_info:
            adapter.generate(
                [ChatMessage(role="user", content="Return structured output")],
                json_mode=True,
            )
        assert "explicitly contain the word 'json'" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 0

    def test_multi_turn_reasoning_content_preserved(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response()
        adapter = DeepSeekAdapter(client=mock_client, thinking=True)

        messages = [
            ChatMessage(role="user", content="What was Apple's FY23 revenue?"),
            ChatMessage(
                role="assistant",
                content="$383,285 million",
                reasoning_content="Examining Form 10-K Item 8 Consolidated Statements of Operations.",
            ),
            ChatMessage(role="user", content="And for FY22?"),
        ]

        adapter.generate(messages)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        assert len(sent_messages) == 3
        assert sent_messages[1]["role"] == "assistant"
        assert sent_messages[1]["reasoning_content"] == (
            "Examining Form 10-K Item 8 Consolidated Statements of Operations."
        )


class TestResponseMappingAndSemantics:
    """2. Verify mapping of content, separate reasoning_content, and token usage metrics."""

    def test_map_chat_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            content="Answer text",
            reasoning_content=None,
            finish_reason="stop",
            model="deepseek-flash",
        )
        adapter = DeepSeekAdapter(client=mock_client)

        res = adapter.generate([ChatMessage(role="user", content="Query")])

        assert res.content == "Answer text"
        assert res.reasoning_content is None
        assert res.finish_reason == "stop"
        assert res.model == "deepseek-flash"
        assert res.latency_ms > 0

    def test_map_reasoner_response_separate_thinking(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            content="Final answer",
            reasoning_content="Step 1: Parse financial notes.\nStep 2: Compare GAAP vs non-GAAP.",
        )
        adapter = DeepSeekAdapter(client=mock_client, thinking=True)

        res = adapter.generate([ChatMessage(role="user", content="Explain reconciliation")])

        assert res.content == "Final answer"
        assert res.reasoning_content == ("Step 1: Parse financial notes.\nStep 2: Compare GAAP vs non-GAAP.")
        assert "Step 1" not in res.content

    def test_map_token_usage_full(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            prompt_tokens=150,
            completion_tokens=60,
            reasoning_tokens=40,
            cache_hit_tokens=100,
            cache_miss_tokens=50,
        )
        adapter = DeepSeekAdapter(client=mock_client)

        res = adapter.generate([ChatMessage(role="user", content="Query")])

        assert res.usage is not None
        assert res.usage.prompt_tokens == 150
        assert res.usage.completion_tokens == 60
        assert res.usage.total_tokens == 210
        assert res.usage.reasoning_tokens == 40
        assert res.usage.cache_hit_tokens == 100
        assert res.usage.cache_miss_tokens == 50

    def test_map_token_usage_partial(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            prompt_tokens=80,
            completion_tokens=20,
            reasoning_tokens=None,
            cache_hit_tokens=None,
            cache_miss_tokens=None,
        )
        adapter = DeepSeekAdapter(client=mock_client)

        res = adapter.generate([ChatMessage(role="user", content="Query")])

        assert res.usage is not None
        assert res.usage.prompt_tokens == 80
        assert res.usage.completion_tokens == 20
        assert res.usage.total_tokens == 100
        assert res.usage.reasoning_tokens is None
        assert res.usage.cache_hit_tokens is None
        assert res.usage.cache_miss_tokens is None

    def test_map_token_usage_none(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(has_usage=False)
        adapter = DeepSeekAdapter(client=mock_client)

        res = adapter.generate([ChatMessage(role="user", content="Query")])
        assert res.usage is None

    def test_json_mode_malformed_raises_llm_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            content="Not valid json {",
            finish_reason="length",
        )
        adapter = DeepSeekAdapter(client=mock_client)

        with pytest.raises(LLMError) as exc_info:
            adapter.generate(
                [ChatMessage(role="user", content="Return json")],
                json_mode=True,
            )
        assert "malformed JSON" in str(exc_info.value)
        assert "length" in str(exc_info.value)

    def test_json_mode_empty_raises_llm_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response(
            content="   ",
            finish_reason="stop",
        )
        adapter = DeepSeekAdapter(client=mock_client)

        with pytest.raises(LLMError) as exc_info:
            adapter.generate(
                [ChatMessage(role="user", content="Return json")],
                json_mode=True,
            )
        assert "empty content in JSON mode" in str(exc_info.value)


class TestStreamHandlingAndTTFT:
    """3. Verify stream chunk parsing, empty chunk filtering, final usage chunk, and TTFT."""

    def test_stream_normal_flow(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_client = MagicMock()
        usage_mock = MagicMock(
            prompt_tokens=40,
            completion_tokens=15,
            total_tokens=55,
            prompt_cache_hit_tokens=20,
            prompt_cache_miss_tokens=20,
            completion_tokens_details=None,
        )

        chunks_stream = [
            make_mock_chunk(content=None),  # Empty ping chunk -> must be skipped
            make_mock_chunk(content="Hello", reasoning=None),
            make_mock_chunk(content=" world", reasoning=None),
            make_mock_chunk(content="", finish_reason="stop"),
            make_mock_chunk(content=None, usage=usage_mock),  # Final usage chunk
        ]
        mock_client.chat.completions.create.return_value = chunks_stream
        adapter = DeepSeekAdapter(client=mock_client)

        with caplog.at_level(logging.INFO):
            emitted = list(adapter.stream([ChatMessage(role="user", content="Hi")]))

        assert len(emitted) == 4
        assert emitted[0].content_delta == "Hello"
        assert emitted[0].is_final is False
        assert emitted[1].content_delta == " world"
        assert emitted[2].finish_reason == "stop"
        assert emitted[2].is_final is True
        assert emitted[3].usage is not None
        assert emitted[3].usage.total_tokens == 55
        assert emitted[3].is_final is True

        assert any("LLM stream TTFT" in record.message for record in caplog.records)


class TestStreamRetryBoundary:
    """4. Verify stream retries before content yield vs immediate failure after content yield."""

    def test_stream_retry_before_content_success(self) -> None:
        mock_client = MagicMock()
        c1 = make_mock_chunk(content="Success token")

        # First call raises RateLimitError; second call succeeds
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(message="Rate limited", response=MagicMock(status_code=429), body=None),
            [c1],
        ]
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        chunks = list(adapter.stream([ChatMessage(role="user", content="Hi")]))

        assert len(chunks) == 1
        assert chunks[0].content_delta == "Success token"
        assert mock_client.chat.completions.create.call_count == 2

    def test_stream_no_restart_after_content_yielded(self) -> None:
        mock_client = MagicMock()
        c1 = make_mock_chunk(content="Partial content")

        def broken_generator() -> Generator[MagicMock, None, None]:
            yield c1
            raise APIConnectionError(request=MagicMock())

        mock_client.chat.completions.create.return_value = broken_generator()
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        chunks = []
        with pytest.raises(LLMProviderError) as exc_info:
            for chunk in adapter.stream([ChatMessage(role="user", content="Hi")]):
                chunks.append(chunk)  # noqa: PERF402

        assert len(chunks) == 1
        assert chunks[0].content_delta == "Partial content"
        assert "Provider error (connection)" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1  # Never restarted!


class TestRetryRecoveryAndAttempts:
    """5. Verify retry recovery after transient errors (429, 500, timeout)."""

    def test_retry_recovery_after_transient_failures(self) -> None:
        mock_client = MagicMock()
        mock_success = make_mock_response(content="Recovered answer")

        mock_client.chat.completions.create.side_effect = [
            RateLimitError(message="Rate limited", response=MagicMock(status_code=429), body=None),
            InternalServerError(message="Server error", response=MagicMock(status_code=500), body=None),
            mock_success,
        ]
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        res = adapter.generate([ChatMessage(role="user", content="Query")])

        assert res.content == "Recovered answer"
        assert mock_client.chat.completions.create.call_count == 3


class TestFailFastHandling:
    """6. Verify fail-fast errors (400, 401, 402, 422) execute exactly once."""

    def test_fail_fast_400_bad_request(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = BadRequestError(
            message="Invalid parameter format",
            response=MagicMock(status_code=400),
            body=None,
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMInvalidRequestError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Invalid request" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1

    def test_fail_fast_401_authentication(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401),
            body=None,
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMAuthError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Authentication failed (HTTP 401)" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1

    def test_fail_fast_402_insufficient_balance(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Insufficient Balance",
            response=MagicMock(status_code=402),
            body=None,
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMInsufficientBalanceError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Insufficient account balance (HTTP 402)" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1

    def test_fail_fast_422_unprocessable_entity(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = UnprocessableEntityError(
            message="Unprocessable entity",
            response=MagicMock(status_code=422),
            body=None,
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMInvalidRequestError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Invalid request (422)" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 1


class TestMaxAttemptsExhaustion:
    """7. Verify correct exception mapping when all retry attempts fail."""

    def test_persistent_429_raises_rate_limit_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit 429", response=MagicMock(status_code=429), body=None
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMRateLimitError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Rate limit exceeded after 3 attempts" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3

    def test_persistent_500_raises_provider_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = InternalServerError(
            message="Server overloaded 500",
            response=MagicMock(status_code=500),
            body=None,
        )
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMProviderError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Provider error (500) after 3 attempts" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3

    def test_persistent_timeout_raises_timeout_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMTimeoutError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Request timed out after 3 attempts" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3

    def test_persistent_connection_error_raises_provider_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        adapter = DeepSeekAdapter(client=mock_client, max_attempts=3)

        with pytest.raises(LLMProviderError) as exc_info:
            adapter.generate([ChatMessage(role="user", content="Query")])

        assert "Provider error (connection) after 3 attempts" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3


class TestSecurityAndKeyValidation:
    """8. Verify API key handling, validation timing, and zero log leakage."""

    def test_missing_key_raises_auth_error_on_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        with pytest.raises(LLMAuthError) as exc_info:
            DeepSeekAdapter()

        assert "DeepSeek API key is required" in str(exc_info.value)

    def test_injected_client_requires_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        adapter = DeepSeekAdapter(client=MagicMock())
        assert adapter is not None

    def test_key_and_prompt_not_leaked_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        secret_key = "sk-super-secret-key-123456789"
        secret_prompt = "confidential-user-prompt-987654"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_mock_response()
        adapter = DeepSeekAdapter(client=mock_client, api_key=secret_key)

        with caplog.at_level(logging.DEBUG):
            adapter.generate([ChatMessage(role="user", content=secret_prompt)])

        combined_logs = " ".join(record.message for record in caplog.records)
        assert secret_key not in combined_logs
        assert secret_prompt not in combined_logs


class TestLLMFactory:
    """9. Verify factory profile resolution and invalid profile error reporting."""

    def test_factory_chat_profile(self) -> None:
        mock_client = MagicMock()
        adapter = LLMFactory.get_adapter("chat", client=mock_client)
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.thinking is False

    def test_factory_reasoner_profile(self) -> None:
        mock_client = MagicMock()
        adapter = LLMFactory.get_adapter("reasoner", client=mock_client)
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.thinking is True

    def test_factory_invalid_profile_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.get_adapter("fast_pro")  # type: ignore[arg-type]

        msg = str(exc_info.value)
        assert "Unknown profile 'fast_pro'" in msg
        assert "'chat'" in msg
        assert "'reasoner'" in msg
