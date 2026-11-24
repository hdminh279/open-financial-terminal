"""Corporate Filing Intelligence - FastAPI Entrypoint Application.

Provides production REST API with CORS support, structured exception handling,
Medallion Lakehouse endpoints, and static UI file serving.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Ensure local environment variables are loaded before service initialization
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.deps import get_duckdb_repository, get_s3_storage
from src.api.routes.companies import router as companies_router
from src.api.routes.health import router as health_router
from src.api.routes.health import telemetry_router
from src.api.routes.history import router as history_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.query import router as query_router
from src.api.routes.watchlist import close_market_services
from src.api.routes.watchlist import router as watchlist_router
from src.api.security import SecurityHardeningMiddleware, get_allowed_origins, validate_security_config
from src.core.observability.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("corporate_filing_intelligence.api")

IS_PROD = os.getenv("APP_ENV", "local").lower() in ("prod", "production")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate security config, warm up DuckDB/S3 on startup, and release HTTP pools on shutdown."""
    validate_security_config()

    try:
        repo = get_duckdb_repository()
        repo.execute_fetchall("SELECT 1")
    except Exception as exc:
        logger.warning("DuckDB warmup skipped: %s", exc)

    try:
        s3 = get_s3_storage()
        if s3 is not None:
            s3.ensure_buckets_exist()
    except Exception as exc:
        logger.warning("MinIO bucket warmup skipped: %s", exc)

    try:
        yield
    finally:
        close_market_services()


app = FastAPI(
    title="Corporate Filing Intelligence API",
    description="Local-first verifiable financial research assistant over SEC EDGAR filings (10-K & 10-Q).",
    version="1.0.0",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    lifespan=lifespan,
)

app.add_middleware(SecurityHardeningMiddleware)

# Configure strict Cross-Origin Resource Sharing (CORS) whitelist
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Disable caching for static UI assets in development to prevent stale CSS/JS."""
    response = await call_next(request)
    if not IS_PROD and (request.url.path == "/" or request.url.path.startswith("/static")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Register API Routers under /api namespace
app.include_router(query_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(companies_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(telemetry_router, prefix="/api")
app.include_router(watchlist_router, prefix="/api")


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Standardize unexpected internal server errors without leaking internals in production."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=exc)
    is_prod = os.getenv("APP_ENV", "local").lower() in ("prod", "production")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected internal error occurred." if is_prod else str(exc),
            "path": request.url.path,
        },
    )


# Serve Static Frontend UI
UI_DIR = Path("src/ui")
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Serve SPA index.html on root endpoint."""
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "Corporate Filing Intelligence API is running. UI not found."})
else:

    @app.get("/", include_in_schema=False)
    async def root_fallback() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Corporate Filing Intelligence API is running.",
                "docs": "/docs",
                "health": "/api/health",
            }
        )
