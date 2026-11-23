"""Unit tests for CorpusSyncEngine."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ai.vectordb.qdrant_client import QdrantStorage
from src.ai.vectordb.sync_engine import CorpusSyncEngine


@pytest.fixture
def sync_test_setup():
    """Setup isolated in-memory Qdrant and mock embedding engines."""
    mem_client = QdrantClient(":memory:")
    storage = QdrantStorage(client=mem_client, collection_name="test_sync_coll")

    # Mock Dense Engine (4 dimensions for fast testing)
    mock_dense = MagicMock()
    mock_dense.dimension = 4
    mock_dense.embed_documents.side_effect = lambda texts, batch_size=32: [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    # Mock Sparse Engine
    mock_sparse = MagicMock()
    sparse_dummy = MagicMock()
    sparse_dummy.indices.tolist.return_value = [10, 20]
    sparse_dummy.values.tolist.return_value = [1.5, 2.5]
    mock_sparse.embed.side_effect = lambda texts, batch_size=32: [sparse_dummy for _ in texts]

    mock_s3 = MagicMock()

    engine = CorpusSyncEngine(
        qdrant_storage=storage,
        dense_engine=mock_dense,
        sparse_engine=mock_sparse,
        s3_storage=mock_s3,
    )
    return engine, storage


class TestCorpusSyncEngine:
    """Test CorpusSyncEngine core behaviors."""

    def test_chunk_id_to_uuid_deterministic(self):
        """Verify identical chunk_id produces exact same UUIDv5 string."""
        chunk_id = "AAPL/0000320193-24-000106/PART I/ITEM 1/main/0"
        uuid1 = CorpusSyncEngine.chunk_id_to_uuid(chunk_id)
        uuid2 = CorpusSyncEngine.chunk_id_to_uuid(chunk_id)
        assert uuid1 == uuid2
        assert uuid.UUID(uuid1).version == 5

    def test_sync_inserts_and_purges_stale_points(self, sync_test_setup):
        """Verify new chunks are inserted and existing orphan points are purged."""
        engine, storage = sync_test_setup

        # Seed storage with 1 orphan point
        storage.create_collection(dense_dim=4)
        orphan_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "orphan_chunk"))
        storage.upsert_points(
            [
                models.PointStruct(
                    id=orphan_uuid,
                    vector={"dense": [0.0] * 4, "bm25": models.SparseVector(indices=[1], values=[1.0])},
                    payload={"chunk_id": "orphan_chunk"},
                )
            ]
        )
        assert len(storage.scroll_all_ids()) == 1

        # Mock Parquet chunk records
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "accession_number": "0000320193-24-000106",
                "form": "10-K",
                "fiscal_year": 2024,
                "filed_date": "2024-11-01",
                "period_end": "2024-09-28",
                "part": "PART I",
                "item": "ITEM 1",
                "subsection": None,
                "content_type": "text",
                "chunk_index": 0,
                "token_count": 120,
                "text": "Apple designs smartphones.",
                "source_url": "https://sec.gov/sample",
                "source_content_hash": "hash_001",
            },
            {
                "chunk_id": "chunk_002",
                "cik": "0000320193",
                "ticker": "AAPL",
                "accession_number": "0000320193-24-000106",
                "form": "10-K",
                "fiscal_year": 2024,
                "filed_date": "2024-11-01",
                "period_end": "2024-09-28",
                "part": "PART I",
                "item": "ITEM 1",
                "subsection": None,
                "content_type": "text",
                "chunk_index": 1,
                "token_count": 150,
                "text": "Apple services include App Store.",
                "source_url": "https://sec.gov/sample",
                "source_content_hash": "hash_002",
            },
        ]

        with patch.object(engine, "load_chunks_from_parquet", return_value=mock_chunks):
            result = engine.sync(batch_size=2)

        assert result["status"] == "SUCCESS"
        assert result["upserted_count"] == 2
        assert result["stale_purged_count"] == 1

        current_ids = storage.scroll_all_ids()
        assert len(current_ids) == 2
        assert orphan_uuid not in current_ids
        assert CorpusSyncEngine.chunk_id_to_uuid("chunk_001") in current_ids
        assert CorpusSyncEngine.chunk_id_to_uuid("chunk_002") in current_ids

    def test_sync_idempotency(self, sync_test_setup):
        """Running sync multiple times should not create duplicate points."""
        engine, storage = sync_test_setup
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "accession_number": "0000320193-24-000106",
                "form": "10-K",
                "fiscal_year": 2024,
                "filed_date": "2024-11-01",
                "period_end": "2024-09-28",
                "part": "PART I",
                "item": "ITEM 1",
                "subsection": None,
                "content_type": "text",
                "chunk_index": 0,
                "token_count": 120,
                "text": "Apple text",
                "source_url": "https://sec.gov",
                "source_content_hash": "hash_001",
            }
        ]

        with patch.object(engine, "load_chunks_from_parquet", return_value=mock_chunks):
            engine.sync()
            assert len(storage.scroll_all_ids()) == 1

            # Run again
            engine.sync()
            assert len(storage.scroll_all_ids()) == 1

    def test_load_chunks_backward_compatible_legacy_and_hive(self, sync_test_setup) -> None:
        """Verify CorpusSyncEngine reads legacy text_chunks.parquet, Hive partitions, and merges without duplicates."""
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq

        from src.data.pipelines.text_pipeline import CHUNK_PARQUET_SCHEMA

        engine, _ = sync_test_setup

        def _make_parquet(rows: list[dict]) -> bytes:
            tbl = pa.Table.from_pylist(rows, schema=CHUNK_PARQUET_SCHEMA)
            buf = io.BytesIO()
            pq.write_table(tbl, buf)
            return buf.getvalue()

        def _row(cid: str, ticker: str) -> dict:
            return {
                "chunk_id": cid,
                "cik": "0000320193",
                "ticker": ticker,
                "accession_number": "acc-1",
                "form": "10-K",
                "fiscal_year": 2024,
                "filed_date": "2024-11-01",
                "period_end": "2024-09-28",
                "part": "PART I",
                "item": "ITEM 1",
                "subsection": None,
                "content_type": "text",
                "chunk_index": 0,
                "token_count": 100,
                "text": f"text for {ticker}",
                "source_url": "https://sec.gov/sample",
                "source_content_hash": f"hash_{cid}",
            }

        legacy_bytes = _make_parquet([_row("aapl_old", "AAPL"), _row("msft_legacy", "MSFT")])
        hive_aapl_bytes = _make_parquet([_row("aapl_hive", "AAPL")])

        engine.s3.list_objects.return_value = [
            "chunks/text_chunks.parquet",
            "chunks/ticker=AAPL/chunks.parquet",
        ]
        engine.s3.download_bytes.side_effect = lambda bucket, key: (
            hive_aapl_bytes if key == "chunks/ticker=AAPL/chunks.parquet" else legacy_bytes
        )

        loaded = engine.load_chunks_from_parquet()
        loaded_ids = {c["chunk_id"] for c in loaded}
        assert loaded_ids == {"aapl_hive", "msft_legacy"}
