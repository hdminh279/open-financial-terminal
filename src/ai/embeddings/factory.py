"""Factory for instantiating embedding strategies based on configuration."""

import os

from src.ai.embeddings.base import EmbeddingStrategy
from src.ai.embeddings.fastembed_strategy import FastEmbedStrategy


class EmbeddingFactory:
    """Factory responsible for providing the appropriate EmbeddingStrategy instance."""

    @staticmethod
    def create(
        provider: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> EmbeddingStrategy:
        """Create and return an EmbeddingStrategy instance.

        Args:
            provider: Provider name ('fastembed'). Defaults to EMBEDDING_PROVIDER env var.
            model_name: Model identifier. Defaults to EMBEDDING_MODEL env var or provider default.
            kwargs: Extra parameters passed to the strategy constructor.

        Returns:
            Configured EmbeddingStrategy instance.
        """
        selected_provider = (provider or os.getenv("EMBEDDING_PROVIDER") or "fastembed").lower().strip()

        selected_model = model_name or os.getenv("EMBEDDING_MODEL")

        if selected_provider == "fastembed":
            return FastEmbedStrategy(model_name=selected_model, **kwargs)

        raise ValueError(f"Unsupported embedding provider: '{selected_provider}'. Supported: ['fastembed']")
