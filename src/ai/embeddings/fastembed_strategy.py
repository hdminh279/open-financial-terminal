"""FastEmbed ONNX CPU Embedding Strategy Implementation."""

import logging
import os
from typing import List, Optional
from fastembed import TextEmbedding

from src.ai.embeddings.base import EmbeddingStrategy

logger = logging.getLogger(__name__)


class FastEmbedStrategy(EmbeddingStrategy):
    """Concrete embedding strategy using FastEmbed with optimized ONNX Runtime on CPU."""

    # Default to BAAI 384-dim high-performance CPU-optimized English dense model
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Initialize FastEmbed strategy with lazy model loading.

        Args:
            model_name: HuggingFace model identifier supported by FastEmbed.
            cache_dir: Optional custom local directory for model weights.
            threads: Optional number of CPU worker threads for ONNX Runtime.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Optional[TextEmbedding] = None
        self._dimension: Optional[int] = None

    def _ensure_loaded(self) -> None:
        """Lazily initialize the TextEmbedding model upon first request."""
        if self._model is None:
            env_threads = os.getenv("FASTEMBED_THREADS")
            cpu_threads = self._threads or (int(env_threads) if env_threads else max(1, (os.cpu_count() or 1) - 1))
            env_cuda = os.getenv("EMBEDDING_CUDA", "").lower() in ("true", "1", "yes")
            env_device = os.getenv("EMBEDDING_DEVICE", "cpu").lower()
            use_cuda = env_cuda or (env_device in ("cuda", "gpu"))
            logger.info(
                "Initializing FastEmbed TextEmbedding model '%s' (threads=%d, cuda=%s)...",
                self._model_name,
                cpu_threads,
                use_cuda,
            )
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                threads=cpu_threads,
                cuda=use_cuda,
                lazy_load=False,
            )
            # Resolve actual model dimension from FastEmbed registry metadata
            for model_meta in TextEmbedding.list_supported_models():
                if model_meta["model"].lower() == self._model_name.lower():
                    self._dimension = model_meta.get("dim", self.DEFAULT_DIMENSION)
                    break
            if self._dimension is None:
                self._dimension = self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        self._ensure_loaded()
        return self._dimension or self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute dense vector embeddings in batches using ONNX runtime."""
        if not texts:
            return []

        self._ensure_loaded()
        assert self._model is not None

        embeddings_generator = self._model.embed(texts, batch_size=batch_size)
        return [embedding.tolist() for embedding in embeddings_generator]

    def embed_query(self, text: str) -> List[float]:
        """Compute dense vector embedding for a single search query."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace.")

        embeddings = self.embed_documents([text.strip()], batch_size=1)
        return embeddings[0]
