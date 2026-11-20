"""Cross-Encoder Reranker using FlashRank ONNX CPU runtime.

Reranks candidate chunks retrieved from Dense, Sparse BM25, or Hybrid RRF
to elevate exact factual and regulatory answers to Rank 1.
"""

import logging
from typing import List, Optional
from flashrank import Ranker, RerankRequest

from src.ai.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """CPU-optimized Cross-Encoder Reranker powered by FlashRank."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize FlashRank Ranker with an ONNX cross-encoder model."""
        self.model_name = model_name
        logger.info("Initializing FlashRankReranker model '%s' on CPU...", model_name)
        kwargs = {"model_name": model_name, "max_length": max_length}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.ranker = Ranker(**kwargs)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        """Rerank a list of candidate RetrievedChunk objects.

        Args:
            query: The user query string.
            chunks: Candidate chunks retrieved from the vector/lexical search.
            top_k: Number of reranked chunks to return.

        Returns:
            List of reranked RetrievedChunk instances updated with new score and rank.
        """
        if not chunks:
            return []

        passages = [{"id": c.chunk_id, "text": c.text, "meta": c} for c in chunks]
        req = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(req)

        reranked_chunks: List[RetrievedChunk] = []
        for rank_idx, item in enumerate(results[:top_k], start=1):
            original_chunk: RetrievedChunk = item["meta"]
            updated_chunk = original_chunk.model_copy(
                update={
                    "score": float(item["score"]),
                    "rank": rank_idx,
                    "retrieval_mode": f"{original_chunk.retrieval_mode}+rerank",
                }
            )
            reranked_chunks.append(updated_chunk)

        return reranked_chunks
