"""Factory for instantiating LLM adapters by operational profile

Provides a centralized construction point allowing downstream agents to request
'chat' (thinking disabled) or 'reasoner' (thinking enabled) profiles without registry bloat.
"""

from typing import Any, Literal

from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.deepseek_adapter import DeepSeekAdapter


class LLMFactory:
    """Factory for creating LLM adapters configured by profile."""

    VALID_PROFILES = ("chat", "reasoner")

    @classmethod
    def get_adapter(
        cls,
        profile: Literal["chat", "reasoner"] = "chat",
        **kwargs: Any,
    ) -> BaseLLMAdapter:
        """Instantiate and return an adapter for the requested profile.

        Args:
            profile: 'chat' (thinking disabled for fast routing/extraction) or
                     'reasoner' (thinking enabled for financial reasoning).
            **kwargs: Direct parameter overrides forwarded to DeepSeekAdapter
                      (e.g. client, timeout, max_attempts).

        Returns:
            Configured BaseLLMAdapter instance.

        Raises:
            ValueError: If an unrecognized profile name is passed.
        """
        if profile == "chat":
            return DeepSeekAdapter(thinking=False, **kwargs)
        elif profile == "reasoner":
            return DeepSeekAdapter(thinking=True, **kwargs)
        else:
            valid_str = ", ".join(repr(p) for p in cls.VALID_PROFILES)
            raise ValueError(f"Unknown profile {profile!r}. Valid profiles are: {valid_str}.")
