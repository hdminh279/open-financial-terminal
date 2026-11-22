"""Integration test for Qdrant Vector Database connectivity."""

import os

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

pytestmark = [pytest.mark.integration]


def get_qdrant_client() -> QdrantClient:
    """Instantiate QdrantClient using environment-aware host resolution."""
    app_env = os.getenv("APP_ENV", "").lower()
    if app_env == "docker":
        host = os.getenv("QDRANT_HOST", "qdrant")
    else:
        host = os.getenv("QDRANT_HOST_LOCAL", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, timeout=5.0, check_compatibility=False)


def test_qdrant_service_health():
    """Verify Qdrant REST service is responsive and healthy."""
    client = get_qdrant_client()
    # Test server connectivity by listing collections
    collections_response = client.get_collections()
    assert collections_response is not None
    assert hasattr(collections_response, "collections")


def test_qdrant_temporary_collection_lifecycle():
    """Verify create, check existence, and delete collection operations."""
    client = get_qdrant_client()
    test_collection = "test_healthcheck_collection"

    # Clean up if left over
    if client.collection_exists(test_collection):
        client.delete_collection(test_collection)

    # Create temporary collection
    from qdrant_client.http import models

    client.create_collection(
        collection_name=test_collection,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    assert client.collection_exists(test_collection) is True

    # Clean up
    client.delete_collection(test_collection)
    assert client.collection_exists(test_collection) is False
