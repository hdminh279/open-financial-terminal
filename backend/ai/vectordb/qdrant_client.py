"""Qdrant Vector Database Adapter for Hybrid Medallion Architecture.

Manages Qdrant collections with Named Vectors (1024-dim dense + sparse BM25),
payload indexing, point reconciliation, and snapshot lifecycle.
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantStorage:
    """Gateway adapter for managing Qdrant Vector DB collections and indexes."""

    DEFAULT_COLLECTION_NAME = "sec_filings_chunks"
    DEFAULT_DENSE_DIM = 384

    def __init__(
        self,
        client: QdrantClient | None = None,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize QdrantStorage with environment-aware connection fallback.

        Args:
            client: Optional pre-configured QdrantClient (e.g. for testing with :memory:).
            host: Optional host override.
            port: Optional port override.
            collection_name: Optional collection name override.
            timeout: Network timeout in seconds.
        """
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME") or self.DEFAULT_COLLECTION_NAME

        if client is not None:
            self.client = client
        else:
            app_env = os.getenv("APP_ENV", "").lower()
            qdrant_url = os.getenv("QDRANT_URL", "").strip()
            api_key = os.getenv("QDRANT_API_KEY") or None

            if qdrant_url and qdrant_url.startswith("https://"):
                logger.info("Connecting to Qdrant Cloud via URL: %s...", qdrant_url)
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=api_key,
                    timeout=timeout,
                    check_compatibility=False,
                )
            else:
                url_host: str | None = None
                url_port: int | None = None
                if qdrant_url:
                    parsed = urlparse(qdrant_url if "://" in qdrant_url else f"http://{qdrant_url}")
                    url_host = parsed.hostname
                    url_port = parsed.port

                if app_env in ("docker", "prod", "production"):
                    resolved_host = host or url_host or os.getenv("QDRANT_HOST", "qdrant")
                else:
                    resolved_host = (
                        host or os.getenv("QDRANT_HOST_LOCAL") or url_host or os.getenv("QDRANT_HOST") or "localhost"
                    )
                resolved_port = port or url_port or int(os.getenv("QDRANT_PORT", "6333"))

                logger.info("Connecting to Qdrant at %s:%d...", resolved_host, resolved_port)
                self.client = QdrantClient(
                    host=resolved_host,
                    port=resolved_port,
                    api_key=api_key,
                    timeout=timeout,
                    check_compatibility=False,
                )

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check whether a collection exists in Qdrant."""
        target = collection_name or self.collection_name
        return self.client.collection_exists(collection_name=target)

    def create_collection(
        self,
        recreate: bool = False,
        dense_dim: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Create Qdrant collection with Named Vectors (dense + sparse BM25) and payload indexes.

        Args:
            recreate: If True, deletes existing collection before creation.
            dense_dim: Dimensionality of dense vectors (defaults to 384).
            collection_name: Target collection name override.
        """
        target = collection_name or self.collection_name
        dim = dense_dim or self.DEFAULT_DENSE_DIM

        if self.collection_exists(target):
            if recreate:
                logger.warning("Deleting existing collection '%s' for recreation...", target)
                self.client.delete_collection(collection_name=target)
            else:
                logger.info("Collection '%s' already exists. Skipping creation.", target)
                return

        logger.info(
            "Creating collection '%s' with %d-dim dense vector and sparse BM25 vector...",
            target,
            dim,
        )
        self.client.create_collection(
            collection_name=target,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

        self._setup_payload_indexes(target)
        logger.info("Collection '%s' and payload indexes successfully created.", target)

    def _setup_payload_indexes(self, target_collection: str) -> None:
        """Create payload indexes on filtering and text search fields."""
        keyword_fields = ["ticker", "form", "item", "accession_number", "content_type"]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=target_collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Integer index for fiscal year range/equality filtering
        self.client.create_payload_index(
            collection_name=target_collection,
            field_name="fiscal_year",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        # Full-text payload index on text field
        self.client.create_payload_index(
            collection_name=target_collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
        )

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Retrieve collection metrics including points count and vector status."""
        target = collection_name or self.collection_name
        info = self.client.get_collection(collection_name=target)
        return {
            "name": target,
            "status": str(info.status),
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
        }

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def upsert_points(
        self,
        points: list[models.PointStruct],
        collection_name: str | None = None,
        wait: bool = True,
    ) -> None:
        """Upsert a batch of points (dense + sparse vectors + payload) into Qdrant."""
        if not points:
            return
        target = collection_name or self.collection_name
        self.client.upsert(collection_name=target, points=points, wait=wait)

    def delete_points(
        self,
        point_ids: list[str],
        collection_name: str | None = None,
        wait: bool = True,
    ) -> None:
        """Delete specific points by their UUIDs (used for stale-point cleanup)."""
        if not point_ids:
            return
        target = collection_name or self.collection_name
        self.client.delete(
            collection_name=target,
            points_selector=models.PointIdsList(points=point_ids),
            wait=wait,
        )

    def scroll_all_ids(
        self,
        collection_name: str | None = None,
        batch_size: int = 1000,
    ) -> set[str]:
        """Retrieve all point IDs present in the collection using cursor pagination."""
        target = collection_name or self.collection_name
        all_ids: set[str] = set()
        next_page_offset: Any | None = None

        while True:
            records, next_page_offset = self.client.scroll(
                collection_name=target,
                limit=batch_size,
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                all_ids.add(str(record.id))

            if next_page_offset is None:
                break

        return all_ids

    def create_snapshot(self, collection_name: str | None = None) -> str:
        """Create a point-in-time snapshot of the collection."""
        target = collection_name or self.collection_name
        snapshot_desc = self.client.create_snapshot(collection_name=target)
        logger.info("Snapshot created successfully for collection '%s': %s", target, snapshot_desc.name)
        return snapshot_desc.name

    def list_snapshots(self, collection_name: str | None = None) -> list[dict[str, Any]]:
        """List all available snapshots for the collection."""
        target = collection_name or self.collection_name
        snapshots = self.client.list_snapshots(collection_name=target)
        return [
            {
                "name": snap.name,
                "size": snap.size,
                "created_at": snap.creation_time,
            }
            for snap in snapshots
        ]
