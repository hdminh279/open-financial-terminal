"""Embedding Engine package exposing strategies and factory."""

from src.ai.embeddings.base import EmbeddingStrategy
from src.ai.embeddings.fastembed_strategy import FastEmbedStrategy
from src.ai.embeddings.factory import EmbeddingFactory

__all__ = ["EmbeddingStrategy", "FastEmbedStrategy", "EmbeddingFactory"]
