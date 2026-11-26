"""Unit tests for QdrantStorage adapter."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.ai.vectordb.qdrant_client import QdrantStorage


@pytest.fixture
def in_memory_qdrant():
    """Provide an isolated in-memory Qdrant instance for each test."""
    client = QdrantClient(":memory:")
    storage = QdrantStorage(client=client, collection_name="test_sec_chunks")
    return storage


class TestQdrantStorageLifecycle:
    """Test collection creation, schema verification, and index management."""

    def test_create_collection_with_named_vectors(self, in_memory_qdrant):
        """Verify collection is created with both dense and sparse BM25 vectors."""
        storage = in_memory_qdrant
        assert storage.collection_exists() is False

        storage.create_collection(dense_dim=384)
        assert storage.collection_exists() is True

        info = storage.get_collection_info()
        assert info["name"] == "test_sec_chunks"
        assert info["points_count"] == 0

    def test_create_collection_idempotent_no_recreate(self, in_memory_qdrant):
        """Calling create_collection without recreate should safely no-op."""
        storage = in_memory_qdrant
        storage.create_collection()
        # Call again without recreate
        storage.create_collection(recreate=False)
        assert storage.collection_exists() is True

    def test_recreate_collection_clears_data(self, in_memory_qdrant):
        """Calling create_collection with recreate=True should drop and recreate."""
        storage = in_memory_qdrant
        storage.create_collection()

        # Insert dummy point
        point = models.PointStruct(
            id="11111111-1111-1111-1111-111111111111",
            vector={
                "dense": [0.0] * 384,
                "bm25": models.SparseVector(indices=[1, 2], values=[1.0, 2.0]),
            },
            payload={"ticker": "AAPL", "text": "Apple text"},
        )
        storage.upsert_points([point])
        assert storage.get_collection_info()["points_count"] == 1

        # Recreate
        storage.create_collection(recreate=True)
        assert storage.get_collection_info()["points_count"] == 0


class TestQdrantStorageCRUD:
    """Test point upsert, scroll pagination, and point deletion."""

    def test_upsert_and_scroll_all_ids(self, in_memory_qdrant):
        """Verify batch upsert and retrieving all IDs via scroll."""
        storage = in_memory_qdrant
        storage.create_collection()

        points = [
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector={
                    "dense": [0.1] * 384,
                    "bm25": models.SparseVector(indices=[10], values=[1.5]),
                },
                payload={"ticker": "AAPL", "fiscal_year": 2024, "text": "iPhone revenue"},
            ),
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000002",
                vector={
                    "dense": [0.2] * 384,
                    "bm25": models.SparseVector(indices=[20], values=[2.5]),
                },
                payload={"ticker": "MSFT", "fiscal_year": 2024, "text": "Azure cloud"},
            ),
        ]
        storage.upsert_points(points)

        ids = storage.scroll_all_ids()
        assert len(ids) == 2
        assert "00000000-0000-0000-0000-000000000001" in ids
        assert "00000000-0000-0000-0000-000000000002" in ids

    def test_delete_points(self, in_memory_qdrant):
        """Verify deleting specific points removes them from collection."""
        storage = in_memory_qdrant
        storage.create_collection()

        p1_id = "00000000-0000-0000-0000-000000000001"
        p2_id = "00000000-0000-0000-0000-000000000002"

        points = [
            models.PointStruct(
                id=p1_id,
                vector={"dense": [0.1] * 384, "bm25": models.SparseVector(indices=[1], values=[1.0])},
                payload={"ticker": "AAPL"},
            ),
            models.PointStruct(
                id=p2_id,
                vector={"dense": [0.2] * 384, "bm25": models.SparseVector(indices=[2], values=[2.0])},
                payload={"ticker": "MSFT"},
            ),
        ]
        storage.upsert_points(points)
        assert len(storage.scroll_all_ids()) == 2

        # Delete p1
        storage.delete_points([p1_id])
        remaining_ids = storage.scroll_all_ids()
        assert len(remaining_ids) == 1
        assert p1_id not in remaining_ids
        assert p2_id in remaining_ids
