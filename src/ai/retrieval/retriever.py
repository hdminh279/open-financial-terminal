"""Hybrid Retrieval Engine combining BGE Dense Vector and Sparse BM25.

Supports Dense Semantic, Sparse BM25, and Reciprocal Rank Fusion (RRF k=60)
with shared metadata filtering on Qdrant Vector Database, coupled with an optional
FlashRank Cross-Encoder Reranker.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from qdrant_client.http import models
from fastembed import SparseTextEmbedding

from src.ai.embeddings.base import EmbeddingStrategy
from src.ai.embeddings.factory import EmbeddingFactory
from src.ai.vectordb.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)


class RetrievalFilter(BaseModel):
    """Normalized metadata filter contract applied across all retrieval streams."""
    ticker: Optional[str] = None
    form: Optional[str] = None
    fiscal_year: Optional[int] = None
    item: Optional[str] = None
    accession_number: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Standardized retrieval output unit with full provenance citations."""
    chunk_id: str
    score: float
    rank: int
    retrieval_mode: str
    ticker: str
    form: str
    fiscal_year: int
    filed_date: str
    period_end: str
    item: str
    part: Optional[str] = None
    subsection: Optional[str] = None
    content_type: str
    text: str
    source_url: str
    accession_number: str


class RetrievalResponse(BaseModel):
    """Container holding query results and execution metadata."""
    query: str
    mode: str
    total_found: int
    latency_ms: float
    chunks: List[RetrievedChunk] = Field(default_factory=list)


class HybridRetriever:
    """Orchestrates Dense, Sparse BM25, Hybrid RRF, and Reranked retrieval on Qdrant."""

    def __init__(
        self,
        qdrant_storage: Optional[QdrantStorage] = None,
        dense_engine: Optional[EmbeddingStrategy] = None,
        sparse_engine: Optional[SparseTextEmbedding] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        """Initialize retriever with storage, embedding engines, and optional reranker."""
        self.qdrant = qdrant_storage or QdrantStorage()
        self.dense_engine = dense_engine or EmbeddingFactory.create()
        self.sparse_engine = sparse_engine or SparseTextEmbedding(model_name="Qdrant/bm25")
        self._reranker = reranker

    @property
    def reranker(self) -> Any:
        """Lazy initialization of FlashRank Cross-Encoder on CPU."""
        if self._reranker is None:
            from src.ai.retrieval.reranker import FlashRankReranker
            self._reranker = FlashRankReranker()
        return self._reranker

    def _build_qdrant_filter(self, filter_params: Optional[RetrievalFilter]) -> Optional[models.Filter]:
        """Convert RetrievalFilter to Qdrant models.Filter."""
        if not filter_params:
            return None

        must_conditions = []
        if filter_params.ticker:
            must_conditions.append(
                models.FieldCondition(key="ticker", match=models.MatchValue(value=filter_params.ticker))
            )
        if filter_params.form:
            must_conditions.append(
                models.FieldCondition(key="form", match=models.MatchValue(value=filter_params.form))
            )
        if filter_params.fiscal_year:
            must_conditions.append(
                models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=filter_params.fiscal_year))
            )
        if filter_params.item:
            normalized_item = filter_params.item.upper() if isinstance(filter_params.item, str) else filter_params.item
            must_conditions.append(
                models.FieldCondition(key="item", match=models.MatchValue(value=normalized_item))
            )
        if filter_params.accession_number:
            must_conditions.append(
                models.FieldCondition(key="accession_number", match=models.MatchValue(value=filter_params.accession_number))
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def _map_point_to_chunk(self, point: Any, rank: int, mode: str) -> RetrievedChunk:
        """Map Qdrant ScoredPoint to RetrievedChunk."""
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=str(payload.get("chunk_id", point.id)),
            score=float(point.score),
            rank=rank,
            retrieval_mode=mode,
            ticker=str(payload.get("ticker", "")),
            form=str(payload.get("form", "")),
            fiscal_year=int(payload.get("fiscal_year", 0)),
            filed_date=str(payload.get("filed_date", "")),
            period_end=str(payload.get("period_end", "")),
            item=str(payload.get("item", "")),
            part=payload.get("part"),
            subsection=payload.get("subsection"),
            content_type=str(payload.get("content_type", "text")),
            text=str(payload.get("text", "")),
            source_url=str(payload.get("source_url", "")),
            accession_number=str(payload.get("accession_number", "")),
        )

    def retrieve_dense(
        self,
        query: str,
        filter_params: Optional[RetrievalFilter] = None,
        limit: int = 10,
    ) -> RetrievalResponse:
        """Execute semantic dense vector search."""
        t0 = time.perf_counter()
        dense_vec = self.dense_engine.embed_query(query)
        qdrant_filter = self._build_qdrant_filter(filter_params)

        res = self.qdrant.client.query_points(
            collection_name=self.qdrant.collection_name,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=limit,
        )
        latency = (time.perf_counter() - t0) * 1000

        chunks = [self._map_point_to_chunk(p, rank=i + 1, mode="dense") for i, p in enumerate(res.points)]
        return RetrievalResponse(
            query=query,
            mode="dense",
            total_found=len(chunks),
            latency_ms=round(latency, 2),
            chunks=chunks,
        )

    def retrieve_bm25(
        self,
        query: str,
        filter_params: Optional[RetrievalFilter] = None,
        limit: int = 10,
    ) -> RetrievalResponse:
        """Execute lexical sparse BM25 search."""
        t0 = time.perf_counter()
        s_vec = list(self.sparse_engine.embed([query]))[0]
        sparse_model_vec = models.SparseVector(
            indices=s_vec.indices.tolist(),
            values=s_vec.values.tolist(),
        )
        qdrant_filter = self._build_qdrant_filter(filter_params)

        res = self.qdrant.client.query_points(
            collection_name=self.qdrant.collection_name,
            query=sparse_model_vec,
            using="bm25",
            query_filter=qdrant_filter,
            limit=limit,
        )
        latency = (time.perf_counter() - t0) * 1000

        chunks = [self._map_point_to_chunk(p, rank=i + 1, mode="bm25") for i, p in enumerate(res.points)]
        return RetrievalResponse(
            query=query,
            mode="bm25",
            total_found=len(chunks),
            latency_ms=round(latency, 2),
            chunks=chunks,
        )

    def retrieve_hybrid(
        self,
        query: str,
        filter_params: Optional[RetrievalFilter] = None,
        limit: int = 10,
        prefetch_limit: int = 25,
    ) -> RetrievalResponse:
        """Execute server-side hybrid search fused via Reciprocal Rank Fusion (RRF k=60)."""
        t0 = time.perf_counter()
        dense_vec = self.dense_engine.embed_query(query)
        s_vec = list(self.sparse_engine.embed([query]))[0]
        sparse_model_vec = models.SparseVector(
            indices=s_vec.indices.tolist(),
            values=s_vec.values.tolist(),
        )
        qdrant_filter = self._build_qdrant_filter(filter_params)

        res = self.qdrant.client.query_points(
            collection_name=self.qdrant.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_model_vec,
                    using="bm25",
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        latency = (time.perf_counter() - t0) * 1000

        chunks = [self._map_point_to_chunk(p, rank=i + 1, mode="hybrid_rrf") for i, p in enumerate(res.points)]
        return RetrievalResponse(
            query=query,
            mode="hybrid_rrf",
            total_found=len(chunks),
            latency_ms=round(latency, 2),
            chunks=chunks,
        )

    def retrieve_hybrid_rerank(
        self,
        query: str,
        filter_params: Optional[RetrievalFilter] = None,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> RetrievalResponse:
        """Execute Hybrid RRF retrieval followed by Cross-Encoder reranking with top candidate pool optimization."""
        t0 = time.perf_counter()
        # Retrieve candidate pool from Hybrid RRF
        hybrid_resp = self.retrieve_hybrid(
            query=query,
            filter_params=filter_params,
            limit=candidate_limit,
        )

        # Rerank candidate chunks using Cross-Encoder
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=hybrid_resp.chunks,
            top_k=limit,
        )
        latency = (time.perf_counter() - t0) * 1000

        return RetrievalResponse(
            query=query,
            mode="hybrid_rerank",
            total_found=len(reranked_chunks),
            latency_ms=round(latency, 2),
            chunks=reranked_chunks,
        )

    def retrieve_multi_entity(
        self,
        query: str,
        tickers: List[str],
        filter_params: Optional[RetrievalFilter] = None,
        per_ticker_limit: int = 4,
        candidate_limit_per_ticker: int = 15,
        mode: str = "hybrid_rerank",
    ) -> RetrievalResponse:
        """Execute multi-entity retrieval ensuring balanced evidence per company.

        Eliminates single-entity context starvation by querying each ticker independently
        with a balanced quota (e.g. 4 chunks AAPL + 4 chunks MSFT).

        Args:
            query: The user query string.
            tickers: List of target ticker symbols (e.g. ['AAPL', 'MSFT']).
            filter_params: Optional shared filter (form, fiscal_year, item).
            per_ticker_limit: Number of top chunks to retain per ticker (default 4).
            candidate_limit_per_ticker: Candidate pool size per ticker for reranking.
            mode: Retrieval mode ('hybrid_rerank', 'hybrid', 'dense', 'bm25').

        Returns:
            Consolidated RetrievalResponse containing balanced chunks across all entities.
        """
        t0 = time.perf_counter()
        clean_tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
        if not clean_tickers:
            return self.retrieve(
                query=query,
                mode=mode,
                filter_params=filter_params,
                limit=per_ticker_limit * 2,
                candidate_limit=candidate_limit_per_ticker * 2,
            )

        combined_chunks: List[RetrievedChunk] = []
        chunks_by_ticker: Dict[str, List[RetrievedChunk]] = {}

        for ticker in clean_tickers:
            ticker_filter = RetrievalFilter(
                ticker=ticker,
                form=filter_params.form if filter_params else None,
                fiscal_year=filter_params.fiscal_year if filter_params else None,
                item=filter_params.item if filter_params else None,
                accession_number=filter_params.accession_number if filter_params else None,
            )
            ticker_query = query
            if ticker.lower() not in query.lower():
                ticker_query = f"{ticker} {query}"

            resp = self.retrieve(
                query=ticker_query,
                mode=mode,
                filter_params=ticker_filter,
                limit=per_ticker_limit,
                candidate_limit=candidate_limit_per_ticker,
            )
            chunks_by_ticker[ticker] = resp.chunks

        # Interleave chunks across entities to ensure fair presentation order in LLM context
        max_chunks = max((len(c) for c in chunks_by_ticker.values()), default=0)
        for i in range(max_chunks):
            for ticker in clean_tickers:
                ticker_chunks = chunks_by_ticker.get(ticker, [])
                if i < len(ticker_chunks):
                    combined_chunks.append(ticker_chunks[i])

        # Reassign sequential ranks
        final_chunks: List[RetrievedChunk] = []
        for idx, chunk in enumerate(combined_chunks, start=1):
            final_chunks.append(chunk.model_copy(update={"rank": idx}))

        latency = (time.perf_counter() - t0) * 1000
        return RetrievalResponse(
            query=query,
            mode=f"multi_entity_{mode}",
            total_found=len(final_chunks),
            latency_ms=round(latency, 2),
            chunks=final_chunks,
        )

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        filter_params: Optional[RetrievalFilter] = None,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> RetrievalResponse:
        """Unified retrieval entrypoint supporting 'dense', 'bm25', 'hybrid', and 'hybrid_rerank' modes."""
        selected_mode = mode.lower().strip()
        if selected_mode == "dense":
            return self.retrieve_dense(query=query, filter_params=filter_params, limit=limit)
        elif selected_mode == "bm25":
            return self.retrieve_bm25(query=query, filter_params=filter_params, limit=limit)
        elif selected_mode in ("hybrid", "hybrid_rrf"):
            return self.retrieve_hybrid(query=query, filter_params=filter_params, limit=limit)
        elif selected_mode in ("hybrid_rerank", "rerank"):
            return self.retrieve_hybrid_rerank(
                query=query,
                filter_params=filter_params,
                limit=limit,
                candidate_limit=candidate_limit,
            )
        else:
            raise ValueError(
                f"Unsupported retrieval mode: '{mode}'. "
                f"Supported: ['dense', 'bm25', 'hybrid', 'hybrid_rerank']"
            )
