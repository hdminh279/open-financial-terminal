"""Abstract Base Strategy for Embedding Generation."""

from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):
    """Abstract interface defining the contract for all embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Compute dense vector embeddings for a list of document chunks.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            List of float lists representing dense vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Compute dense vector embedding for a single search query.

        Args:
            text: Query text string.

        Returns:
            Dense vector as a list of floats.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Authoritative identifier/name of the embedding model."""
        pass
