"""Corpus Synchronization Engine between Parquet Marts and Qdrant.

Ensures Parquet remains the authoritative Single Source of Truth.
Handles deterministic UUIDv5 mapping, stale-point cleanup, dual-vector generation
(dense + sparse BM25), batch upserting, and count/hash integrity audits.
"""

import io
import logging
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client.http import models

from src.ai.embeddings.base import EmbeddingStrategy
from src.ai.embeddings.factory import EmbeddingFactory
from src.ai.vectordb.qdrant_client import QdrantStorage
from src.data.storage.s3_storage import S3Storage

load_dotenv()
logger = logging.getLogger(__name__)


class CorpusSyncEngine:
    """Synchronizes text chunks from Gold Parquet lakehouse to Qdrant Vector DB."""

    def __init__(
        self,
        qdrant_storage: Optional[QdrantStorage] = None,
        dense_engine: Optional[EmbeddingStrategy] = None,
        sparse_engine: Optional[SparseTextEmbedding] = None,
        s3_storage: Optional[S3Storage] = None,
    ) -> None:
        """Initialize sync engine with storage and embedding adapters."""
        self.qdrant = qdrant_storage or QdrantStorage()
        self.dense_engine = dense_engine or EmbeddingFactory.create()
        self.sparse_engine = sparse_engine or SparseTextEmbedding(model_name="Qdrant/bm25")
        self.s3 = s3_storage or S3Storage()

    @staticmethod
    def chunk_id_to_uuid(chunk_id: str) -> str:
        """Convert a deterministic SHA-256 chunk_id into a stable UUIDv5 string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def load_chunks_from_parquet(
        self,
        bucket: str = "sec-gold",
        key: str = "chunks/text_chunks.parquet",
        local_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load text chunks from local Parquet file or MinIO S3 object."""
        resolved_local = local_path or os.getenv("CHUNKS_PARQUET_LOCAL_PATH")

        if resolved_local and os.path.isfile(resolved_local):
            logger.info("Loading chunks directly from local Parquet: %s", resolved_local)
            table = pq.read_table(resolved_local)
        else:
            logger.info("Downloading chunks from MinIO s3://%s/%s...", bucket, key)
            parquet_bytes = self.s3.download_bytes(bucket=bucket, key=key)
            table = pq.read_table(io.BytesIO(parquet_bytes))

        if limit is not None and limit > 0:
            table = table.slice(0, limit)

        pydict = table.to_pydict()
        num_rows = table.num_rows
        chunks: List[Dict[str, Any]] = []

        keys = list(pydict.keys())
        for i in range(num_rows):
            chunk = {k: pydict[k][i] for k in keys}
            chunks.append(chunk)

        logger.info("Loaded %d chunks from Parquet source of truth.", len(chunks))
        return chunks

    def sync(
        self,
        bucket: str = "sec-gold",
        key: str = "chunks/text_chunks.parquet",
        local_path: Optional[str] = None,
        recreate: bool = False,
        limit: Optional[int] = None,
        batch_size: int = 64,
        create_snapshot_after: bool = False,
    ) -> Dict[str, Any]:
        """Perform end-to-end synchronization: stale cleanup, embedding, and upsert."""
        chunks = self.load_chunks_from_parquet(
            bucket=bucket, key=key, local_path=local_path, limit=limit
        )

        if not chunks:
            logger.warning("No chunks found to synchronize.")
            return {"status": "SKIPPED", "total_parquet_chunks": 0, "upserted_count": 0}

        # 1. Ensure collection and payload indexes exist
        dense_dim = self.dense_engine.dimension
        self.qdrant.create_collection(recreate=recreate, dense_dim=dense_dim)

        # 2. Build mapping of valid Point IDs
        parquet_point_ids: Dict[str, Dict[str, Any]] = {
            self.chunk_id_to_uuid(c["chunk_id"]): c for c in chunks
        }
        valid_id_set = set(parquet_point_ids.keys())

        # 3. Detect and purge stale points from Qdrant
        existing_ids = self.qdrant.scroll_all_ids()
        stale_ids = existing_ids - valid_id_set

        if stale_ids:
            logger.warning("Detected %d stale points in Qdrant not in Parquet. Purging...", len(stale_ids))
            self.qdrant.delete_points(point_ids=list(stale_ids), wait=True)
            existing_ids = existing_ids - stale_ids
            logger.info("Successfully purged %d stale points.", len(stale_ids))
        else:
            logger.info("Zero stale points detected in Qdrant.")

        # Incremental Resume: Filter out chunks that are already indexed in Qdrant
        chunks_to_embed = [
            c for c in chunks
            if self.chunk_id_to_uuid(c["chunk_id"]) not in existing_ids
        ]
        already_indexed_count = len(chunks) - len(chunks_to_embed)
        logger.info(
            "%d chunks already indexed in Qdrant (skipped). %d remaining chunks to embed.",
            already_indexed_count,
            len(chunks_to_embed),
        )

        total_chunks = len(chunks_to_embed)
        if total_chunks == 0:
            logger.info("All %d chunks already synchronized in Qdrant.", len(chunks))
            return {
                "status": "SUCCESS",
                "total_parquet_chunks": len(chunks),
                "upserted_count": 0,
                "already_indexed_count": already_indexed_count,
                "stale_purged_count": len(stale_ids),
                "qdrant_points_count": len(existing_ids),
                "snapshot_name": None,
            }

        # 4. Batch embed and upsert remaining chunks
        logger.info(
            "Embedding and upserting %d remaining chunks in batches of %d...",
            total_chunks,
            batch_size,
        )

        upserted_count = 0
        for start_idx in range(0, total_chunks, batch_size):
            end_idx = min(start_idx + batch_size, total_chunks)
            batch_chunks = chunks_to_embed[start_idx:end_idx]
            batch_texts = [c["text"] for c in batch_chunks]

            # Generate dense embeddings (BGE-M3 / FastEmbed)
            dense_vectors = self.dense_engine.embed_documents(batch_texts, batch_size=batch_size)

            # Generate sparse BM25 term vectors
            sparse_vectors = list(self.sparse_engine.embed(batch_texts, batch_size=batch_size))

            # Assemble Qdrant points with full 17 metadata fields in payload
            points: List[models.PointStruct] = []
            for chunk_data, d_vec, s_vec in zip(batch_chunks, dense_vectors, sparse_vectors):
                p_id = self.chunk_id_to_uuid(chunk_data["chunk_id"])

                payload = {
                    "chunk_id": str(chunk_data["chunk_id"]),
                    "cik": str(chunk_data["cik"]),
                    "ticker": str(chunk_data["ticker"]),
                    "accession_number": str(chunk_data["accession_number"]),
                    "form": str(chunk_data["form"]),
                    "fiscal_year": int(chunk_data["fiscal_year"]),
                    "filed_date": str(chunk_data["filed_date"]),
                    "period_end": str(chunk_data["period_end"]),
                    "part": chunk_data.get("part"),
                    "item": str(chunk_data["item"]),
                    "subsection": chunk_data.get("subsection"),
                    "content_type": str(chunk_data["content_type"]),
                    "chunk_index": int(chunk_data["chunk_index"]),
                    "token_count": int(chunk_data["token_count"]),
                    "text": str(chunk_data["text"]),
                    "source_url": str(chunk_data["source_url"]),
                    "source_content_hash": str(chunk_data["source_content_hash"]),
                }

                points.append(
                    models.PointStruct(
                        id=p_id,
                        vector={
                            "dense": d_vec,
                            "bm25": models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )
                )

            is_last_batch = (end_idx >= total_chunks)
            self.qdrant.upsert_points(points=points, wait=is_last_batch)
            upserted_count += len(points)
            logger.info(
                "Sync progress: %d/%d new chunks indexed (overall in Qdrant: %d/%d)...",
                upserted_count,
                total_chunks,
                already_indexed_count + upserted_count,
                len(chunks),
            )

        # 5. Optional snapshot creation
        snapshot_name: Optional[str] = None
        if create_snapshot_after:
            snapshot_name = self.qdrant.create_snapshot()

        final_info = self.qdrant.get_collection_info()
        return {
            "status": "SUCCESS",
            "total_parquet_chunks": len(chunks),
            "upserted_count": upserted_count,
            "already_indexed_count": already_indexed_count,
            "stale_purged_count": len(stale_ids),
            "qdrant_points_count": final_info["points_count"],
            "snapshot_name": snapshot_name,
        }

    def verify_integrity(
        self,
        bucket: str = "sec-gold",
        key: str = "chunks/text_chunks.parquet",
        local_path: Optional[str] = None,
        sample_size: int = 50,
    ) -> Dict[str, Any]:
        """Perform comprehensive integrity checks between Parquet and Qdrant."""
        chunks = self.load_chunks_from_parquet(bucket=bucket, key=key, local_path=local_path)
        parquet_count = len(chunks)

        qdrant_ids = self.qdrant.scroll_all_ids()
        qdrant_count = len(qdrant_ids)

        parquet_point_ids = {self.chunk_id_to_uuid(c["chunk_id"]): c for c in chunks}
        valid_id_set = set(parquet_point_ids.keys())

        # Check 1: Count match
        counts_match = (parquet_count == qdrant_count)

        # Check 2: Stale check
        stale_count = len(qdrant_ids - valid_id_set)
        missing_count = len(valid_id_set - qdrant_ids)

        # Check 3: Sample hash verification
        sampled_count = min(sample_size, parquet_count)
        sampled_uuids = random.sample(list(valid_id_set), sampled_count)

        # Fetch records from Qdrant by Point IDs
        fetched_points = self.qdrant.client.retrieve(
            collection_name=self.qdrant.collection_name,
            ids=sampled_uuids,
            with_payload=True,
            with_vectors=False,
        )

        hashes_matched = 0
        discrepancies: List[str] = []

        for pt in fetched_points:
            pt_id = str(pt.id)
            parquet_chunk = parquet_point_ids.get(pt_id)
            if not parquet_chunk:
                discrepancies.append(f"Point {pt_id} not found in Parquet mapping.")
                continue

            qdrant_hash = pt.payload.get("source_content_hash")
            expected_hash = parquet_chunk["source_content_hash"]

            if qdrant_hash == expected_hash:
                hashes_matched += 1
            else:
                discrepancies.append(
                    f"Hash mismatch for {pt_id}: Qdrant='{qdrant_hash}' vs Parquet='{expected_hash}'"
                )

        is_passed = counts_match and (stale_count == 0) and (missing_count == 0) and (len(discrepancies) == 0)

        report = {
            "status": "PASSED" if is_passed else "FAILED",
            "counts_match": counts_match,
            "parquet_count": parquet_count,
            "qdrant_count": qdrant_count,
            "stale_count": stale_count,
            "missing_count": missing_count,
            "sampled_audited": sampled_count,
            "hashes_matched": hashes_matched,
            "discrepancies": discrepancies,
        }
        return report
