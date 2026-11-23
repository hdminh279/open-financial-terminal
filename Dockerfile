FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ARG APP_UID=1000
ARG APP_GID=1000

WORKDIR /app

# PYTHONUNBUFFERED=1 -> unbuffered stdout/stderr logs
# PYTHONDONTWRITEBYECODE=1 -> avoid writing .pyc files into read-only /app/src
# UV_SYSTEM_PYTHON=0 -> isolate dependencies inside /app/.venv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    HOME=/home/appuser \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models/fastembed \
    HF_HOME=/app/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${APP_GID} appuser \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /usr/sbin/nologin appuser

COPY --chown=root:root pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

# Application code and configs stay root-owned (read-only for appuser)
COPY --chown=root:root . .

# Grant write access strictly to runtime, data, log, model, and cache directories
RUN mkdir -p /app/data /app/logs /app/models/fastembed /app/models/huggingface /home/appuser/.cache /home/appuser/.duckdb \
    && chown -R appuser:appuser /app/data /app/logs /app/models /home/appuser \
    && chmod -R u=rwX,g=rX,o= /app/data /app/logs /app/models /home/appuser

USER appuser

# Pre-install DuckDB httpfs and community gcs extensions at build time to prevent cold-start network delays
RUN python -c "import duckdb; c = duckdb.connect(':memory:'); c.execute('INSTALL httpfs; LOAD httpfs;'); c.execute('INSTALL gcs FROM community; LOAD gcs;')"

# Pre-download FastEmbed embedding model weights at build time for instant cold starts
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/models/fastembed')"

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3)"]

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
