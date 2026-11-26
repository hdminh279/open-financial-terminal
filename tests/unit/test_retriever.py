"""Unit tests for HybridRetriever, FlashRankReranker, and data contracts."""

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.ai.retrieval.reranker import FlashRankReranker
from backend.ai.retrieval.retriever import HybridRetriever, RetrievalFilter, RetrievedChunk


@pytest.fixture
def mock_retriever_setup():
    """Setup isolated retriever with in-memory Qdrant and mocked embeddings."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_retrieval",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )

    # Insert sample points
    client.upsert(
        collection_name="test_retrieval",
        points=[
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector={
                    "dense": [1.0, 0.0, 0.0, 0.0],
                    "bm25": models.SparseVector(indices=[10, 20], values=[2.0, 3.0]),
                },
                payload={
                    "chunk_id": "c1",
                    "ticker": "AAPL",
                    "form": "10-K",
                    "fiscal_year": 2024,
                    "item": "ITEM 1",
                    "text": "Apple designs iPhone devices.",
                    "source_url": "https://sec.gov/aapl",
                    "accession_number": "0000320193-24-000106",
                },
            ),
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000002",
                vector={
                    "dense": [0.0, 1.0, 0.0, 0.0],
                    "bm25": models.SparseVector(indices=[30, 40], values=[2.0, 3.0]),
                },
                payload={
                    "chunk_id": "c2",
                    "ticker": "MSFT",
                    "form": "10-K",
                    "fiscal_year": 2024,
                    "item": "ITEM 1",
                    "text": "Microsoft Azure cloud computing.",
                    "source_url": "https://sec.gov/msft",
                    "accession_number": "0000789019-24-000050",
                },
            ),
        ],
    )

    mock_qdrant_storage = MagicMock()
    mock_qdrant_storage.client = client
    mock_qdrant_storage.collection_name = "test_retrieval"

    # Mock Dense
    mock_dense = MagicMock()
    mock_dense.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]

    # Mock Sparse
    mock_sparse = MagicMock()
    dummy_sparse = MagicMock()
    dummy_sparse.indices.tolist.return_value = [10, 20]
    dummy_sparse.values.tolist.return_value = [2.0, 3.0]
    mock_sparse.embed.return_value = [dummy_sparse]

    # Mock Reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, chunks, top_k: [
        c.model_copy(update={"rank": i + 1, "score": 0.95 - (i * 0.1), "retrieval_mode": "hybrid_rrf+rerank"})
        for i, c in enumerate(chunks[:top_k])
    ]

    retriever = HybridRetriever(
        qdrant_storage=mock_qdrant_storage,
        dense_engine=mock_dense,
        sparse_engine=mock_sparse,
        reranker=mock_reranker,
    )
    return retriever


class TestHybridRetriever:
    """Test retrieval modes and filter behaviors."""

    def test_filter_building(self, mock_retriever_setup):
        """Test conversion of RetrievalFilter to Qdrant Filter."""
        retriever = mock_retriever_setup
        filter_params = RetrievalFilter(ticker="AAPL", fiscal_year=2024, min_fiscal_year=2022)
        qdrant_filter = retriever._build_qdrant_filter(filter_params)

        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 3

    def test_retrieve_dense_mode(self, mock_retriever_setup):
        """Test dense vector retrieval returns top matching chunk."""
        retriever = mock_retriever_setup
        res = retriever.retrieve(query="iPhone smartphone", mode="dense", limit=1)

        assert res.mode == "dense"
        assert res.total_found == 1
        assert res.chunks[0].ticker == "AAPL"
        assert res.chunks[0].chunk_id == "c1"

    def test_retrieve_bm25_mode(self, mock_retriever_setup):
        """Test sparse BM25 retrieval returns matching chunk."""
        retriever = mock_retriever_setup
        res = retriever.retrieve(query="iPhone", mode="bm25", limit=1)

        assert res.mode == "bm25"
        assert res.total_found == 1
        assert res.chunks[0].ticker == "AAPL"

    def test_retrieve_hybrid_mode(self, mock_retriever_setup):
        """Test hybrid RRF returns combined results."""
        retriever = mock_retriever_setup
        res = retriever.retrieve(query="iPhone Azure", mode="hybrid", limit=2)

        assert res.mode == "hybrid_rrf"
        assert res.total_found == 2
        assert len(res.chunks) == 2

    def test_retrieve_hybrid_rerank_mode(self, mock_retriever_setup):
        """Test hybrid + reranker returns reranked chunks with updated scores."""
        retriever = mock_retriever_setup
        res = retriever.retrieve(query="iPhone products", mode="hybrid_rerank", limit=2)

        assert res.mode == "hybrid_rerank"
        assert res.total_found == 2
        assert res.chunks[0].rank == 1
        assert res.chunks[0].score == 0.95
        assert res.chunks[0].retrieval_mode == "hybrid_rrf+rerank"

    def test_retrieve_with_ticker_filter(self, mock_retriever_setup):
        """Test metadata filtering restricts retrieval strictly to specified company."""
        retriever = mock_retriever_setup
        aapl_filter = RetrievalFilter(ticker="AAPL")
        res = retriever.retrieve(query="cloud computing", mode="hybrid", filter_params=aapl_filter, limit=5)

        for chunk in res.chunks:
            assert chunk.ticker == "AAPL"

    def test_flashrank_reranker_real(self):
        """Test FlashRankReranker with sample chunks and ONNX model."""
        reranker = FlashRankReranker(model_name="ms-marco-TinyBERT-L-2-v2")
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                score=0.1,
                rank=1,
                retrieval_mode="hybrid",
                ticker="AAPL",
                form="10-K",
                fiscal_year=2024,
                filed_date="2024-11-01",
                period_end="2024-09-28",
                item="ITEM 1",
                content_type="text",
                text="Bananas and tropical fruits growing in trees.",
                source_url="http://test",
                accession_number="acc-1",
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                score=0.1,
                rank=2,
                retrieval_mode="hybrid",
                ticker="AAPL",
                form="10-K",
                fiscal_year=2024,
                filed_date="2024-11-01",
                period_end="2024-09-28",
                item="ITEM 1",
                content_type="text",
                text="Apple designs, manufactures, and markets smartphones including iPhone.",
                source_url="http://test",
                accession_number="acc-2",
            ),
        ]
        reranked = reranker.rerank(query="Apple iPhone smartphones", chunks=chunks, top_k=2)
        assert len(reranked) == 2
        # chunk-2 should be ranked 1 because it directly discusses Apple iPhone
        assert reranked[0].chunk_id == "chunk-2"
        assert reranked[0].rank == 1
        assert reranked[0].score > reranked[1].score

    def test_retrieve_multi_entity_balanced_chunks(self, mock_retriever_setup):
        """Test multi-entity fan-out retrieves balanced chunks across companies."""
        retriever = mock_retriever_setup
        res = retriever.retrieve_multi_entity(
            query="cloud and devices",
            tickers=["AAPL", "MSFT"],
            per_ticker_limit=1,
            mode="hybrid",
        )
        assert res.mode == "multi_entity_hybrid"
        assert res.total_found == 2
        tickers_returned = [c.ticker for c in res.chunks]
        assert "AAPL" in tickers_returned
        assert "MSFT" in tickers_returned
        assert res.chunks[0].rank == 1
        assert res.chunks[1].rank == 2

    def test_rag_node_multi_entity_fan_out(self):
        """Test that rag_node fans out retrieval when state has multiple tickers."""
        from backend.ai.agent.graph import rag_node

        mock_retriever = MagicMock()
        mock_resp = MagicMock()
        mock_resp.total_found = 2
        mock_resp.chunks = [
            RetrievedChunk(
                chunk_id="c1",
                score=0.9,
                rank=1,
                retrieval_mode="multi_entity_hybrid_rerank",
                ticker="AAPL",
                form="10-K",
                fiscal_year=2024,
                filed_date="2024-11-01",
                period_end="2024-09-28",
                item="ITEM 1",
                content_type="text",
                text="Apple chunk",
                source_url="http://aapl",
                accession_number="acc-1",
            ),
            RetrievedChunk(
                chunk_id="c2",
                score=0.9,
                rank=2,
                retrieval_mode="multi_entity_hybrid_rerank",
                ticker="MSFT",
                form="10-K",
                fiscal_year=2024,
                filed_date="2024-11-01",
                period_end="2024-09-28",
                item="ITEM 1",
                content_type="text",
                text="Microsoft chunk",
                source_url="http://msft",
                accession_number="acc-2",
            ),
        ]
        mock_resp.model_dump.return_value = {"total_found": 2, "chunks": []}
        mock_retriever.retrieve_multi_entity.return_value = mock_resp

        state = {
            "query": "Compare AAPL and MSFT",
            "routing_decision": {"tickers": ["AAPL", "MSFT"], "fiscal_years": [2024]},
            "rag_params": {"search_query": "Compare AAPL and MSFT business models"},
        }
        res = rag_node(state, retriever=mock_retriever)
        assert mock_retriever.retrieve_multi_entity.called
        assert len(res["evidence_items"]) == 2
        assert res["evidence_items"][0]["ticker"] == "AAPL"
        assert res["evidence_items"][1]["ticker"] == "MSFT"
