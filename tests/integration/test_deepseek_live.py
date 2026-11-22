"""Live smoke tests for DeepSeekAdapter against production API.

Only runs when explicitly invoked with `-m live` and DEEPSEEK_API_KEY is present in environment.
Validates:
1. Live chat profile synchronous generation.
2. Live chat profile streaming and chunk reconstruction.
3. Live reasoner profile chain-of-thought generation with separate reasoning_content.
"""

import os

import pytest
from dotenv import load_dotenv

from src.ai.llm.contracts import ChatMessage
from src.ai.llm.factory import LLMFactory

load_dotenv()

# Skip all tests in module if no API key is available
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("DEEPSEEK_API_KEY"),
        reason="DEEPSEEK_API_KEY not configured in environment.",
    ),
]


def test_live_chat_generate() -> None:
    """Smoke test: execute live non-thinking completion with low token limit."""
    adapter = LLMFactory.get_adapter("chat")
    messages = [
        ChatMessage(
            role="user",
            content="Answer in one short sentence: What is the primary purpose of an SEC Form 10-K?",
        )
    ]

    response = adapter.generate(messages, max_tokens=64)

    assert response.content is not None
    assert len(response.content.strip()) > 0
    assert response.reasoning_content is None
    assert response.model is not None
    assert response.latency_ms > 0
    assert response.usage is not None
    assert response.usage.total_tokens > 0


def test_live_chat_stream() -> None:
    """Smoke test: execute live streaming completion and verify chunk continuity."""
    adapter = LLMFactory.get_adapter("chat")
    messages = [
        ChatMessage(
            role="user",
            content="Count from 1 to 5 separated by commas.",
        )
    ]

    chunks = list(adapter.stream(messages, max_tokens=64))

    assert len(chunks) > 0
    full_text = "".join(c.content_delta for c in chunks)
    assert any(char.isdigit() for char in full_text)

    # Final chunk carries usage
    final_chunk = chunks[-1]
    assert final_chunk.is_final is True
    assert final_chunk.usage is not None
    assert final_chunk.usage.total_tokens > 0


def test_live_reasoner_generate() -> None:
    """Smoke test: execute live reasoning completion and verify reasoning_content extraction."""
    adapter = LLMFactory.get_adapter("reasoner")
    messages = [
        ChatMessage(
            role="user",
            content="If a company's revenue is $100M and operating expenses are $70M, what is its operating margin?",
        )
    ]

    response = adapter.generate(messages, max_tokens=256)

    assert response.content is not None
    assert len(response.content.strip()) > 0
    # On DeepSeek V4 reasoner profile, reasoning_content should be captured
    assert response.reasoning_content is not None
    assert len(response.reasoning_content.strip()) > 0
    assert "30%" in response.content or "0.3" in response.content
    assert response.latency_ms > 0
    assert response.usage is not None
    assert response.usage.reasoning_tokens is not None or response.usage.completion_tokens > 0
