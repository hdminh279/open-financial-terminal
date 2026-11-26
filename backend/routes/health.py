"""Health and readiness probe router.

Probes connectivity across Medallion MinIO S3 object storage, Qdrant vector database,
and DuckDB Gold Parquet lakehouse engine, plus protected telemetry reporting.
"""

import hmac
import os
from datetime import UTC, datetime

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse

from backend.contracts import HealthResponse
from backend.core.observability.telemetry import summarize_traffic_telemetry
from backend.deps import get_duckdb_repository, get_qdrant_storage, get_s3_storage
from backend.security import is_production_env

router = APIRouter(prefix="/health", tags=["Health"])
telemetry_router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Probe system health and storage connectivity",
    description="Validates MinIO S3, Qdrant vector database, and DuckDB Gold analytical marts.",
)
def check_health() -> HealthResponse | JSONResponse:
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

    if minio_ok and qdrant_ok and duckdb_ok:
        overall = "healthy"
    elif duckdb_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    payload = HealthResponse(
        status=overall,
        minio_connected=minio_ok,
        qdrant_connected=qdrant_ok,
        duckdb_ready=duckdb_ok,
        timestamp=datetime.now(UTC).isoformat(),
    )
    if overall == "unhealthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
        )
    return payload


@telemetry_router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Retrieve anonymized traffic and latency telemetry summary",
    description="Returns aggregate privacy-first usage statistics (token-protected in production).",
)
def get_traffic_telemetry(
    x_telemetry_token: str | None = Header(default=None, alias="X-Telemetry-Token"),
) -> dict:
    """Return anonymized traffic telemetry computed from logs/run_telemetry.jsonl."""
    if is_production_env():
        expected = os.getenv("TELEMETRY_API_TOKEN", "").strip()
        if not expected or not hmac.compare_digest(x_telemetry_token or "", expected):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Telemetry endpoint requires a valid X-Telemetry-Token in production.",
            )
    return summarize_traffic_telemetry(persist=False)
