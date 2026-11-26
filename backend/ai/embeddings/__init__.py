"""Embedding Engine package exposing strategies and factory."""

from backend.ai.embeddings.base import EmbeddingStrategy
from backend.ai.embeddings.factory import EmbeddingFactory
from backend.ai.embeddings.fastembed_strategy import FastEmbedStrategy

__all__ = ["EmbeddingStrategy", "FastEmbedStrategy", "EmbeddingFactory"]
