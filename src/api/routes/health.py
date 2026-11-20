"""Health and readiness probe router.

Probes connectivity across Medallion MinIO S3 object storage, Qdrant vector database,
and DuckDB Gold Parquet lakehouse engine.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, status

from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.vectordb.qdrant_client import QdrantStorage
from src.api.contracts import HealthResponse
from src.data.storage.s3_storage import S3Storage

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Probe system health and storage connectivity",
    description="Validates MinIO S3, Qdrant vector database, and DuckDB Gold analytical marts.",
)
async def check_health() -> HealthResponse:
    """Execute live probes across core infrastructure components."""
    minio_ok = False
    qdrant_ok = False
    duckdb_ok = False

    # 1. Probe MinIO
    try:
        s3 = S3Storage()
        s3.client.list_buckets()
        minio_ok = True
    except Exception:
        minio_ok = False

    # 2. Probe Qdrant
    try:
        qdrant = QdrantStorage()
        qdrant.client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    # 3. Probe DuckDB
    try:
        repo = DuckDBFinancialRepository()
        res = repo.conn.execute("SELECT 1").fetchall()
        duckdb_ok = len(res) > 0
    except Exception:
        duckdb_ok = False

    # Determine overall status
    if minio_ok and qdrant_ok and duckdb_ok:
        overall = "healthy"
    elif duckdb_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        minio_connected=minio_ok,
        qdrant_connected=qdrant_ok,
        duckdb_ready=duckdb_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
