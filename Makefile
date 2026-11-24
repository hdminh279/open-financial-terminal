.PHONY: help install up down prod-up prod-down dev lint format check test dbt \
        pc-ingest pc-process pc-index pc-pipeline verify-s3 eval-50 telemetry

UV ?= uv

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install frozen Python dependencies with uv
	$(UV) sync --frozen

# ---------------------------------------------------------------------------
# Laptop Stack (Docker Compose App + MinIO + Qdrant + dbt Gold Marts)
# ---------------------------------------------------------------------------
up: ## Start local Docker stack (FastAPI UI + MinIO + Qdrant)
	docker compose up -d

down: ## Stop local Docker stack
	docker compose down

prod-up: ## Start hardened production stack (App + MinIO + Qdrant + Cloudflared)
	docker compose -f docker-compose.prod.yml up -d --build

prod-down: ## Stop production stack
	docker compose -f docker-compose.prod.yml down

dev: ## Run FastAPI backend locally with hot reload (requires MinIO & Qdrant running)
	$(UV) run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src --reload-dir configs

dbt: ## Build DuckDB Gold financial marts & run dbt data quality tests (Laptop)
	cd transform && $(UV) run dbt build --target local

# ---------------------------------------------------------------------------
# PC Heavy Batch Pipeline (Standalone Python CLI — NO Docker Compose, NO dbt)
# Expects MinIO (localhost:9000) and Qdrant (localhost:6333) reachable locally.
# ---------------------------------------------------------------------------
pc-ingest: ## [PC] Ingest 100 SEC EDGAR + 50 Vietnam HOSE filings into MinIO Bronze
	$(UV) run python -m src.cli ingest --all --limit 4
	$(UV) run python -m src.cli ingest --vn

pc-process: ## [PC] Parse Silver facts/VN facts and re-extract clean HTML chunks (--force)
	$(UV) run python -m src.cli process --facts
	$(UV) run python -m src.cli process --vn
	$(UV) run python -m src.cli process --text --force

pc-index: ## [PC] Recreate Qdrant hybrid index (dense + BM25) and verify integrity
	$(UV) run python -m src.cli index --recreate --verify

pc-pipeline: pc-ingest pc-process pc-index ## [PC] Run full ingestion -> processing -> Qdrant indexing

verify-s3: ## Verify SHA-256 integrity of objects in MinIO Bronze
	$(UV) run python -m src.cli verify --bucket sec-bronze

# ---------------------------------------------------------------------------
# Quality, Tests & Benchmarks
# ---------------------------------------------------------------------------
lint: ## Run Ruff lint & format checks
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
	@for f in src/ui/js/*.js; do node --input-type=module --check < "$$f" || exit 1; done

format: ## Auto-format Python code with Ruff
	$(UV) run ruff format src tests

test: ## Run offline unit test suite
	APP_ENV=local USE_TEST_FIXTURES=true $(UV) run pytest tests/unit -q

check: lint test ## Run all offline lint, syntax, and unit test gates

eval-50: ## Run live 50-question real-data benchmark (Retrieval + E2E Agent)
	$(UV) run python -m tests.benchmark.benchmark_production_50

telemetry: ## Print anonymized traffic & latency summary from logs/run_telemetry.jsonl
	$(UV) run python -c "import json; from src.core.observability import summarize_traffic_telemetry; print(json.dumps(summarize_traffic_telemetry(persist=True), indent=2))"
