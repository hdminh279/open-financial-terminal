"""Corporate Filing Intelligence - FastAPI Entrypoint Application.

Provides production REST API with CORS support, structured exception handling,
Medallion Lakehouse endpoints, and static UI file serving.
"""

import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure local environment variables are loaded before service initialization
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.companies import router as companies_router
from src.api.routes.health import router as health_router
from src.api.routes.history import router as history_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.query import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("corporate_filing_intelligence.api")

app = FastAPI(
    title="Corporate Filing Intelligence API",
    description="Local-first verifiable financial research assistant over SEC EDGAR filings (10-K & 10-Q).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for local development and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Disable caching for static UI assets in development to prevent stale CSS/JS."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
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


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Standardize unexpected internal server errors."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
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
            return FileResponse(
                index_file,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return JSONResponse({"message": "Corporate Filing Intelligence API is running. UI not found."})
else:
    @app.get("/", include_in_schema=False)
    async def root_fallback() -> JSONResponse:
        return JSONResponse({
            "message": "Corporate Filing Intelligence API is running.",
            "docs": "/docs",
            "health": "/api/health",
        })
