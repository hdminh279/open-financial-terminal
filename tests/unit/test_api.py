"""Unit tests for FastAPI endpoints, contracts, and chat history persistence.

Validates request validation, exception handling, dependency injection overrides,
and MinIO S3 chat history synchronization.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.ai.agent.contracts import (
    AgentResponse,
    ChartData,
    ChartType,
    Citation,
    IntentType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from backend.deps import (
    get_chat_history,
    get_research_agent,
)
from backend.main import app
from backend.security import get_current_user_id
from backend.services.chat_history import ChatHistoryManager


@pytest.fixture
def memory_chat_history(monkeypatch: pytest.MonkeyPatch) -> ChatHistoryManager:
    """Create memory-only ChatHistoryManager for tests with S3 disabled."""
    monkeypatch.setenv("ENABLE_S3_CHAT_HISTORY", "false")
    mgr = ChatHistoryManager(s3_storage=None)
    mgr._s3_enabled = False
    mgr._s3 = None
    mgr._sessions.clear()
    app.dependency_overrides[get_chat_history] = lambda: mgr
    app.dependency_overrides[get_current_user_id] = lambda: "default_user"
    yield mgr
    app.dependency_overrides.pop(get_chat_history, None)
    app.dependency_overrides.pop(get_current_user_id, None)


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create mock ResearchAgent."""
    agent = MagicMock()
    sample_response = AgentResponse(
        query="What is Apple's revenue in 2024?",
        intent=IntentType.FINANCIAL_METRIC,
        answer="Apple's revenue for FY2024 was $391.04B [1].",
        citations=[
            Citation(
                citation_id=1,
                formatted_reference="[1] AAPL FY2024 10-K (Item 8)",
                source_url="https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019324000106/aapl-20240928.htm",
            )
        ],
        charts=[
            ChartData(
                title="Apple Revenue Trend",
                chart_type=ChartType.BAR,
                x_labels=["FY2022", "FY2023", "FY2024"],
                datasets=[{"label": "Revenues ($B)", "values": [394.33, 383.29, 391.04]}],
            )
        ],
        tool_results=[
            ToolExecutionResult(
                tool_name="financial_sql_tool",
                status=ToolExecutionStatus.SUCCESS,
                execution_time_ms=12.5,
            )
        ],
        execution_time_ms=120.5,
    )
    agent.query.return_value = sample_response
    app.dependency_overrides[get_research_agent] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_research_agent, None)


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_serve_static_ui(client: TestClient) -> None:
    """Validate root endpoint serves UI index.html and modular CSS/JS assets are mounted."""
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "<title>Corporate Filing Intelligence" in root_resp.text
    assert '<script type="module" src="/static/js/terminal.js"></script>' in root_resp.text

    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert '@import url("./styles/theme.css")' in css_resp.text

    for css_mod, expected_token in (
        ("/static/styles/theme.css", "--bg-app"),
        ("/static/styles/layout.css", ".workstation-container"),
        ("/static/styles/terminal.css", ".chat-pane"),
        ("/static/styles/watchlist.css", ".stock-terminal-fullscreen"),
    ):
        mod_resp = client.get(css_mod)
        assert mod_resp.status_code == 200
        assert expected_token in mod_resp.text

    for js_mod in (
        "/static/js/api.js",
        "/static/js/chart.js",
        "/static/js/watchlist.js",
        "/static/js/terminal.js",
    ):
        js_resp = client.get(js_mod)
        assert js_resp.status_code == 200
        assert "Corporate Filing Intelligence" in js_resp.text


def test_get_companies(client: TestClient) -> None:
    """Validate GET /api/companies returns AAPL, MSFT, GOOGL."""
    resp = client.get("/api/companies")
    assert resp.status_code == 200
    data = resp.json()
    assert "companies" in data
    tickers = [c["ticker"] for c in data["companies"]]
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers


def test_get_health(client: TestClient) -> None:
    """Validate GET /api/health probes core infrastructure."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "minio_connected" in data
    assert "qdrant_connected" in data
    assert "duckdb_ready" in data


def test_get_health_returns_503_when_unhealthy(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate GET /api/health returns HTTP 503 when core DuckDB lakehouse probe fails."""
    from backend.routes import health as health_mod

    broken_repo = MagicMock()
    broken_repo.execute_fetchall.side_effect = RuntimeError("duckdb unavailable")
    monkeypatch.setattr(health_mod, "get_duckdb_repository", lambda: broken_repo)
    monkeypatch.setattr(health_mod, "get_s3_storage", lambda: None)
    monkeypatch.setattr(health_mod, "get_qdrant_storage", lambda: None)

    resp = client.get("/api/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"
    assert resp.json()["duckdb_ready"] is False


def test_get_metrics_valid(client: TestClient) -> None:
    """Validate GET /api/metrics successfully queries DuckDB repository."""
    resp = client.get("/api/metrics?ticker=AAPL&metric=revenues&fiscal_year=2024&fiscal_period=FY")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["metric"] == "revenues"
    assert data["tickers"] == ["AAPL"]
    assert data["metrics"] == ["revenues"]
    assert "records" in data
    assert isinstance(data["execution_time_ms"], (int, float))


def test_get_metrics_batch_multi_company(client: TestClient) -> None:
    """Validate GET /api/metrics supports multi-ticker batch queries."""
    resp = client.get("/api/metrics?tickers=AAPL&tickers=MSFT&metrics=revenues&fiscal_year=2023&fiscal_period=FY")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tickers"] == ["AAPL", "MSFT"]
    assert data["metrics"] == ["revenues"]
    assert "records" in data


def test_get_metrics_invalid_ticker(client: TestClient) -> None:
    """Validate GET /api/metrics rejects unsupported tickers with 400."""
    resp = client.get("/api/metrics?ticker=INVALID&metric=revenues")
    assert resp.status_code == 400
    assert "Unsupported ticker" in resp.json()["detail"]


def test_get_metrics_invalid_metric(client: TestClient) -> None:
    """Validate GET /api/metrics rejects unsupported metrics with 400."""
    resp = client.get("/api/metrics?ticker=AAPL&metric=unknown_metric")
    assert resp.status_code == 400
    assert "Unsupported metric" in resp.json()["detail"]


def test_post_query_success(
    client: TestClient,
    mock_agent: MagicMock,
    memory_chat_history: ChatHistoryManager,
) -> None:
    """Validate POST /api/query executes research workflow and creates chat history."""
    payload = {
        "query": "What is Apple's revenue in 2024?",
        "session_id": "test-session-123",
    }
    resp = client.post("/api/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == payload["query"]
    assert data["session_id"] == "test-session-123"
    assert data["intent"] == "FINANCIAL_METRIC"
    assert "Apple's revenue for FY2024" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_id"] == 1
    assert len(data["charts"]) == 1

    # Verify session in ChatHistoryManager
    session = memory_chat_history.get_session("test-session-123")
    assert session is not None
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == payload["query"]
    assert session.messages[1].role == "assistant"
    assert "Apple's revenue for FY2024" in session.messages[1].content


def test_post_query_empty_rejected(client: TestClient) -> None:
    """Validate POST /api/query rejects empty query with 422."""
    resp = client.post("/api/query", json={"query": "   "})
    assert resp.status_code == 422


def test_chat_history_lifecycle(
    client: TestClient,
    memory_chat_history: ChatHistoryManager,
) -> None:
    """Validate GET /api/history, GET /api/history/{id}, and DELETE /api/history/{id}."""
    memory_chat_history.get_or_create_session("sess-lifecycle-1")
    memory_chat_history.add_user_message("sess-lifecycle-1", "Hello AI")
    memory_chat_history.add_assistant_message("sess-lifecycle-1", "Hello Analyst")

    # 1. List sessions
    list_resp = client.get("/api/history")
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["session_id"] == "sess-lifecycle-1" for s in sessions)

    # 2. Get specific session
    detail_resp = client.get("/api/history/sess-lifecycle-1")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["session_id"] == "sess-lifecycle-1"
    assert len(detail_data["messages"]) == 2

    # 3. Delete session
    del_resp = client.delete("/api/history/sess-lifecycle-1")
    assert del_resp.status_code == 200

    # 4. Confirm deleted
    assert client.get("/api/history/sess-lifecycle-1").status_code == 404


def test_post_query_agent_failure(
    client: TestClient,
    mock_agent: MagicMock,
    memory_chat_history: ChatHistoryManager,
) -> None:
    """Validate POST /api/query returns 500 and records error turn when agent fails."""
    mock_agent.query.side_effect = RuntimeError("DeepSeek connection timed out")
    payload = {"query": "What are Alphabet risks?", "session_id": "test-err-sess"}
    resp = client.post("/api/query", json=payload)
    assert resp.status_code == 500
    assert "Query execution failed" in resp.json()["detail"]

    # Ensure error is audited in chat history
    session = memory_chat_history.get_session("test-err-sess")
    assert session is not None
    assert len(session.messages) == 2
    assert "error occurred" in session.messages[1].content.lower()


def test_post_query_sanitizes_persisted_error_in_prod(
    client: TestClient,
    mock_agent: MagicMock,
    memory_chat_history: ChatHistoryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify production mode omits raw exception details from persisted chat history metadata."""
    monkeypatch.setenv("APP_ENV", "prod")
    mock_agent.query.side_effect = RuntimeError("internal_s3_secret_host:9000")
    resp = client.post("/api/query", json={"query": "Apple revenue", "session_id": "test-prod-err"})
    assert resp.status_code == 500
    session = memory_chat_history.get_session("test-prod-err")
    assert session is not None
    meta = session.messages[1].metadata or {}
    assert meta.get("error_code") == "INTERNAL_SERVER_ERROR"
    assert "error" not in meta
    assert "internal_s3_secret_host" not in str(meta)


def test_chat_history_s3_persistence() -> None:
    """Validate ChatHistoryManager syncs serialized session bytes to S3Storage."""
    mock_s3 = MagicMock()
    mock_s3.object_exists.return_value = False

    mgr = ChatHistoryManager(s3_storage=mock_s3)
    mgr.get_or_create_session("s3-test-session")
    mgr.add_user_message("s3-test-session", "What is Microsoft gross profit?")
    mgr.add_assistant_message("s3-test-session", "Microsoft gross profit was $146B.")

    # Verify put_object / upload_bytes was invoked
    assert mock_s3.upload_bytes.called
    call_args = mock_s3.upload_bytes.call_args[1]
    assert call_args["bucket"] == "cfi-app-state"
    assert call_args["key"] == "sessions/default_user/s3-test-session.json"
    assert b"Microsoft gross profit was $146B" in call_args["data"]


def test_chat_history_s3_non_ascii_unicode_title() -> None:
    """Validate ChatHistoryManager sanitizes non-ASCII Vietnamese titles in S3 metadata while preserving UTF-8 payload."""
    mock_s3 = MagicMock()
    mock_s3.object_exists.return_value = False

    mgr = ChatHistoryManager(s3_storage=mock_s3)
    mgr.get_or_create_session("s3-unicode-session")
    # Add Vietnamese user message
    mgr.add_user_message("s3-unicode-session", "so sánh 2 công ty microsoft và google")
    mgr.add_assistant_message("s3-unicode-session", "Báo cáo so sánh Microsoft và Google.")

    assert mock_s3.upload_bytes.called
    call_args = mock_s3.upload_bytes.call_args[1]
    # S3 Metadata should only contain ASCII characters
    meta = call_args["metadata"]
    for k, v in meta.items():
        assert all(ord(c) < 128 for c in str(v)), f"Metadata key {k} has non-ASCII value: {v}"
    # Payload bytes must retain full UTF-8 Vietnamese content
    assert "so sánh 2 công ty microsoft và google".encode() in call_args["data"]


def test_chat_history_lazy_persistence_ignores_empty_sessions() -> None:
    """Validate empty sessions without messages are not eagerly written to S3."""
    mock_s3 = MagicMock()
    mgr = ChatHistoryManager(s3_storage=mock_s3)
    # Creating a session without messages must not upload
    mgr.get_or_create_session("empty-session-1")
    assert not mock_s3.upload_bytes.called


def test_chat_history_s3_delete_and_reconcile() -> None:
    """Validate session deletion issues delete_object to S3Storage adapter."""
    mock_s3 = MagicMock()
    mock_s3.object_exists.return_value = True

    mgr = ChatHistoryManager(s3_storage=mock_s3)
    mgr.get_or_create_session("sess-to-delete")
    mgr.add_user_message("sess-to-delete", "Hello")

    deleted = mgr.delete_session("sess-to-delete")
    assert deleted is True
    assert mock_s3.delete_object.called
    assert "default_user:sess-to-delete" not in mgr._sessions
