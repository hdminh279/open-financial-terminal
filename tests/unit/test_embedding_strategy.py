"""Unit tests for Embedding Strategy and Factory."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ai.embeddings.factory import EmbeddingFactory
from src.ai.embeddings.fastembed_strategy import FastEmbedStrategy


class TestEmbeddingStrategyContract:
    """Verify abstract contracts and FastEmbedStrategy implementation."""

    @patch("src.ai.embeddings.fastembed_strategy.TextEmbedding")
    def test_fastembed_strategy_lazy_loading(self, mock_text_embedding_cls):
        """Ensure TextEmbedding is not loaded until an embed operation is requested."""
        strategy = FastEmbedStrategy(model_name="BAAI/bge-large-en-v1.5")
        mock_text_embedding_cls.assert_not_called()

        # Trigger loading
        _ = strategy.dimension
        mock_text_embedding_cls.assert_called_once()

    @patch("src.ai.embeddings.fastembed_strategy.TextEmbedding")
    def test_embed_documents_batching(self, mock_text_embedding_cls):
        """Test document embedding correctly converts numpy arrays to float lists."""
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        mock_text_embedding_cls.return_value = mock_instance

        strategy = FastEmbedStrategy()
        docs = ["Doc 1 content", "Doc 2 content"]
        results = strategy.embed_documents(docs, batch_size=2)

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]
        mock_instance.embed.assert_called_once_with(docs, batch_size=2)

    def test_embed_documents_empty_input(self):
        """Test empty input returns empty list without error."""
        strategy = FastEmbedStrategy()
        assert strategy.embed_documents([]) == []

    @patch("src.ai.embeddings.fastembed_strategy.TextEmbedding")
    def test_embed_query_success(self, mock_text_embedding_cls):
        """Test query embedding returns single 1D list."""
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [np.array([0.7, 0.8, 0.9])]
        mock_text_embedding_cls.return_value = mock_instance

        strategy = FastEmbedStrategy()
        query_vec = strategy.embed_query("What are the revenue growth drivers?")

        assert query_vec == [0.7, 0.8, 0.9]

    def test_embed_query_empty_string_raises(self):
        """Test empty or whitespace query raises ValueError."""
        strategy = FastEmbedStrategy()
        with pytest.raises(ValueError, match="cannot be empty"):
            strategy.embed_query("")

        with pytest.raises(ValueError, match="cannot be empty"):
            strategy.embed_query("   ")


class TestEmbeddingFactory:
    """Verify EmbeddingFactory creates expected strategies."""

    def test_factory_default_provider(self):
        """Factory should instantiate FastEmbedStrategy by default."""
        engine = EmbeddingFactory.create()
        assert isinstance(engine, FastEmbedStrategy)
        assert engine.model_name == FastEmbedStrategy.DEFAULT_MODEL

    def test_factory_custom_model(self):
        """Factory should respect custom model name parameter."""
        engine = EmbeddingFactory.create(provider="fastembed", model_name="BAAI/bge-small-en-v1.5")
        assert isinstance(engine, FastEmbedStrategy)
        assert engine.model_name == "BAAI/bge-small-en-v1.5"

    def test_factory_unsupported_provider_raises(self):
        """Factory should raise ValueError for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingFactory.create(provider="unknown_provider")
