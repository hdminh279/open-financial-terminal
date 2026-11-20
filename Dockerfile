FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# PYTHONUNBUFFERED=1 -> python export out stdout/stderr, disable buffer
# PYTHONDONTWRITEBYECODE=1 -> prevent python from automatically generating .pyc binary files
# UV_SYSTEM_PYTHON=0 -> uv packages must not be installed into the operating system's native Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYECODE=1 \
    UV_SYSTEM_PYTHON=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

# Copy all code
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
