"""Health and readiness probe router.

Probes connectivity across Medallion MinIO S3 object storage, Qdrant vector database,
and DuckDB Gold Parquet lakehouse engine.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, status

from src.api.contracts import HealthResponse
from src.api.deps import get_duckdb_repository, get_qdrant_storage, get_s3_storage
from src.core.observability.telemetry import summarize_traffic_telemetry

router = APIRouter(prefix="/health", tags=["Health"])
telemetry_router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Probe system health and storage connectivity",
    description="Validates MinIO S3, Qdrant vector database, and DuckDB Gold analytical marts.",
)
def check_health() -> HealthResponse:
    """Execute live probes across core infrastructure components using shared adapters."""
    minio_ok = False
    qdrant_ok = False
    duckdb_ok = False

    # 1. Probe MinIO
    try:
        s3 = get_s3_storage()
        if s3 is not None:
            s3.client.list_buckets()
            minio_ok = True
    except Exception:
        minio_ok = False

    # 2. Probe Qdrant
    try:
        qdrant = get_qdrant_storage()
        if qdrant is not None:
            qdrant.client.get_collections()
            qdrant_ok = True
    except Exception:
        qdrant_ok = False

    # 3. Probe DuckDB
    try:
        repo = get_duckdb_repository()
        res = repo.execute_fetchall("SELECT 1")
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
        timestamp=datetime.now(UTC).isoformat(),
    )


@telemetry_router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Retrieve anonymized traffic and latency telemetry summary",
    description="Returns aggregate privacy-first usage statistics (no PII or raw query text).",
)
def get_traffic_telemetry() -> dict:
    """Return anonymized traffic telemetry computed from logs/run_telemetry.jsonl."""
    return summarize_traffic_telemetry(persist=False)
