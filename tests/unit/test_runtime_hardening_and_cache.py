"""Regression and feature unit tests for API hardening, semantic validation, and TTL query caching."""

import json
import time
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.ai.agent.contracts import (
    AgentResponse,
    IntentType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.research_agent import ResearchAgent, TTLQueryCache
from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.agent.tools.financial_sql_tool import FinancialMetricRecord, FinancialQueryInput, FinancialQueryResponse
from src.ai.vectordb.qdrant_client import QdrantStorage
from src.api.chat_history import ChatHistoryManager, validate_session_id
from src.api.deps import get_chat_history, get_duckdb_repository
from src.api.main import app
from src.api.security import reset_rate_limiters
from src.core.financial.semantic_guardrails import FinancialSemanticValidator
from src.data.market.service import WatchlistRepository
from src.data.storage.s3_storage import S3Storage


@pytest.fixture(autouse=True)
def _reset_limiters() -> None:
    reset_rate_limiters()


def test_prod_env_resolves_container_service_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify APP_ENV=prod targets Docker internal service hosts instead of localhost."""
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("S3_ENDPOINT_URL_LOCAL", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "prod_access")
    monkeypatch.setenv("S3_SECRET_KEY", "prod'secret")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_HOST_LOCAL", "localhost")

    with patch("src.data.storage.s3_storage.boto3.client") as mock_boto:
        storage = S3Storage()
        assert storage.endpoint_url == "http://minio:9000"
        assert mock_boto.call_args.kwargs["endpoint_url"] == "http://minio:9000"

    mock_conn = MagicMock()
    DuckDBFinancialRepository(
        statements_path="s3://sec-gold/marts/fct_financial_statements.parquet",
        ratios_path="s3://sec-gold/marts/fct_financial_ratios.parquet",
        connection=mock_conn,
    )
    executed_sql = " ".join(str(call.args[0]) for call in mock_conn.execute.call_args_list)
    assert "CREATE OR REPLACE SECRET s3_lakehouse" in executed_sql
    assert "ENDPOINT 'minio:9000'" in executed_sql
    assert "SECRET 'prod''secret'" in executed_sql
    assert "SET GLOBAL s3_" not in executed_sql

    with patch("src.ai.vectordb.qdrant_client.QdrantClient") as mock_qdrant:
        QdrantStorage()
        assert mock_qdrant.call_args.kwargs["host"] == "qdrant"
        assert mock_qdrant.call_args.kwargs["port"] == 6333


def test_duckdb_prod_s3_requires_credentials_and_rejects_unsafe_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DuckDB fails fast in prod without S3 credentials and rejects control characters in SQL literals."""
    from src.ai.agent.tools.duckdb_repository import _sql_literal

    assert _sql_literal("a'b") == "'a''b'"
    for bad in ("abc\x00def", "abc\ndef", "abc\rdef"):
        with pytest.raises(ValueError, match="NUL or newline"):
            _sql_literal(bad)

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="S3_ACCESS_KEY and S3_SECRET_KEY are required"):
        DuckDBFinancialRepository(
            statements_path="s3://sec-gold/marts/fct_financial_statements.parquet",
            ratios_path="s3://sec-gold/marts/fct_financial_ratios.parquet",
            connection=MagicMock(),
        )


def test_metrics_route_reflects_dynamic_ticker_catalog_changes() -> None:
    """Verify /api/metrics validates tickers dynamically via get_supported_tickers() instead of import-time state."""
    mock_repo = MagicMock(spec=DuckDBFinancialRepository)
    app.dependency_overrides[get_duckdb_repository] = lambda: mock_repo
    try:
        client = TestClient(app)
        with (
            patch("src.api.routes.metrics.get_supported_tickers", return_value={"AAPL", "DYNCO"}),
            patch("src.ai.agent.tools.financial_sql_tool.get_supported_tickers", return_value={"AAPL", "DYNCO"}),
        ):
            sample_resp = FinancialQueryResponse(
                query=FinancialQueryInput(tickers=["DYNCO"], metrics=["revenues"]),
                record_count=1,
                data=[
                    FinancialMetricRecord(
                        record_key="k-dyn",
                        ticker="DYNCO",
                        cik="0009999999",
                        currency="USD",
                        fiscal_year=2025,
                        fiscal_period="FY",
                        period_type="annual",
                        metric_name="revenues",
                        metric_value=Decimal("1000.00"),
                    )
                ],
                execution_time_ms=1.5,
            )
            with patch("src.api.routes.metrics.FinancialSQLTool.execute", return_value=sample_resp):
                resp = client.get("/api/metrics?ticker=DYNCO&metric=revenues")
        assert resp.status_code == 200
        assert resp.json()["tickers"] == ["DYNCO"]

        # Once removed from the catalog, the same ticker must be rejected immediately
        with patch("src.api.routes.metrics.get_supported_tickers", return_value={"AAPL"}):
            rejected = client.get("/api/metrics?ticker=DYNCO&metric=revenues")
        assert rejected.status_code == 400
        assert "Unsupported ticker" in rejected.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_duckdb_repository, None)


def test_watchlist_loads_persisted_s3_config_using_object_exists() -> None:
    """Verify WatchlistRepository calls S3Storage.object_exists and restores saved tickers."""
    mock_s3 = MagicMock(spec=S3Storage)
    mock_s3.object_exists.return_value = True
    saved_payload = {
        "version": "1.0",
        "user_id": "default_user",
        "items": [{"ticker": "FPT", "market": "VN", "added_at": "2026-11-22T00:00:00Z"}],
        "updated_at": "2026-11-22T00:00:00Z",
    }
    mock_s3.download_bytes.return_value = json.dumps(saved_payload).encode("utf-8")

    repo = WatchlistRepository(s3_storage=mock_s3, quote_service=MagicMock())
    cfg = repo.get_watchlist()

    mock_s3.object_exists.assert_called_once_with("cfi-app-state", "watchlist/default_user.json")
    assert [item.ticker for item in cfg.items] == ["FPT"]


def test_session_id_path_traversal_rejected_and_cache_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invalid session IDs are rejected at API and manager boundaries and cache evicts LRU."""
    monkeypatch.setenv("ENABLE_S3_CHAT_HISTORY", "false")
    with pytest.raises(ValueError, match="Invalid session_id"):
        validate_session_id("../user_settings/watchlist")

    mgr = ChatHistoryManager(s3_storage=None, max_sessions=2)
    mgr.get_or_create_session("sess-1")
    mgr.get_or_create_session("sess-2")
    mgr.get_or_create_session("sess-3")
    assert list(mgr._sessions.keys()) == ["default_user:sess-2", "default_user:sess-3"]

    app.dependency_overrides[get_chat_history] = lambda: mgr
    try:
        client = TestClient(app)
        resp_query = client.post(
            "/api/query",
            json={"query": "Apple revenue", "session_id": "../user_settings/watchlist"},
        )
        assert resp_query.status_code == 422

        resp_hist = client.get("/api/history/..%2Fwatchlist")
        assert resp_hist.status_code in (400, 404)

        resp_invalid = client.delete("/api/history/bad$id")
        assert resp_invalid.status_code == 400
    finally:
        app.dependency_overrides.pop(get_chat_history, None)


def test_metrics_endpoint_uses_real_currency_form_and_sanitizes_prod_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /api/metrics returns actual currency/form and hides raw DB errors in prod."""
    mock_repo = MagicMock(spec=DuckDBFinancialRepository)
    app.dependency_overrides[get_duckdb_repository] = lambda: mock_repo
    try:
        client = TestClient(app)
        sample_resp = FinancialQueryResponse(
            query=FinancialQueryInput(tickers=["FPT"], metrics=["revenues", "revenue_growth_yoy"]),
            record_count=2,
            data=[
                FinancialMetricRecord(
                    record_key="k1",
                    ticker="FPT",
                    cik="VN:FPT",
                    currency="VND",
                    fiscal_year=2024,
                    fiscal_period="FY",
                    period_type="annual",
                    metric_name="revenues",
                    metric_value=Decimal("62849000000000.00"),
                    source_filed_date=date(2025, 3, 15),
                ),
                FinancialMetricRecord(
                    record_key="k2",
                    ticker="FPT",
                    cik="VN:FPT",
                    currency="VND",
                    fiscal_year=2024,
                    fiscal_period="FY",
                    period_type="annual",
                    metric_name="revenue_growth_yoy",
                    metric_value=Decimal("0.1940"),
                ),
            ],
            execution_time_ms=4.2,
        )
        with patch("src.api.routes.metrics.FinancialSQLTool.execute", return_value=sample_resp):
            resp = client.get("/api/metrics?ticker=FPT&metrics=revenues&metrics=revenue_growth_yoy")
        assert resp.status_code == 200
        records = {r["metric_name"]: r for r in resp.json()["records"]}
        assert records["revenues"]["unit"] == "VND"
        assert records["revenues"]["form"] == "BCTC"
        assert records["revenue_growth_yoy"]["unit"] == "Ratio"

        monkeypatch.setenv("APP_ENV", "prod")
        with patch("src.api.routes.metrics.FinancialSQLTool.execute", side_effect=RuntimeError("secret_bucket_path")):
            err_resp = client.get("/api/metrics?ticker=FPT&metric=revenues")
        assert err_resp.status_code == 500
        assert "secret_bucket_path" not in err_resp.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_duckdb_repository, None)


def test_negative_margin_validation_preserves_percentage_scale() -> None:
    """Verify negative margins expressed in percent (-5.0) are not multiplied by 100."""
    is_valid_pct, calc_pct, _ = FinancialSemanticValidator.validate_margin(
        revenue=100.0,
        income=-5.0,
        reported_margin=-5.0,
    )
    assert is_valid_pct is True
    assert calc_pct == -5.0

    is_valid_ratio, _, _ = FinancialSemanticValidator.validate_margin(
        revenue=100.0,
        income=-5.0,
        reported_margin=-0.05,
    )
    assert is_valid_ratio is True


def test_ttl_query_cache_and_research_agent_caching_rules() -> None:
    """Verify TTLQueryCache hit (<50ms), expiry, eviction, and ResearchAgent bypass rules."""
    cache = TTLQueryCache(ttl_seconds=0.05, max_size=2)
    dummy = AgentResponse(query="q1", intent=IntentType.FINANCIAL_METRIC, answer="A1")
    cache.set("k1", dummy)
    cache.set("k2", dummy)
    cache.set("k3", dummy)
    assert cache.size == 2
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    time.sleep(0.06)
    assert cache.get("k2") is None

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "intent": IntentType.FINANCIAL_METRIC.value,
        "final_answer": "Apple FY2024 revenue was $391.04B [1].",
        "tool_results": [
            ToolExecutionResult(
                tool_name="financial_sql_tool",
                status=ToolExecutionStatus.SUCCESS,
                execution_time_ms=10.0,
            ).model_dump()
        ],
        "citations": [],
        "charts": [],
        "report_sections": [],
        "activity_timeline": [],
    }
    agent = ResearchAgent(compiled_graph=mock_graph, cache_ttl_seconds=60.0)

    r1 = agent.query("What is Apple revenue FY2024?")
    r2 = agent.query("  what is   apple revenue fy2024? ")
    assert mock_graph.invoke.call_count == 1
    assert r1.answer == r2.answer
    assert r2.execution_time_ms < 50.0

    # Follow-up queries with conversation history bypass the standalone cache
    agent.query("What is Apple revenue FY2024?", conversation_history=[{"role": "user", "content": "prev"}])
    assert mock_graph.invoke.call_count == 2

    # Live web search responses must never be cached
    mock_graph.invoke.return_value["tool_results"] = [
        ToolExecutionResult(
            tool_name="web_search_tool",
            status=ToolExecutionStatus.SUCCESS,
            execution_time_ms=120.0,
        ).model_dump()
    ]
    agent.query("Latest Apple news today")
    agent.query("Latest Apple news today")
    assert mock_graph.invoke.call_count == 4
