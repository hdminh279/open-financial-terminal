"""Unit tests verifying LangGraph concurrency, database error safety, and telemetry."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from src.ai.agent.contracts import AgentRequest, IntentType, ToolExecutionStatus
from src.ai.agent.graph import sql_node
from src.ai.agent.research_agent import ResearchAgent
from src.ai.agent.state import timeline_reducer
from src.ai.agent.tools.duckdb_repository import (
    DatabaseExecutionError,
    DuckDBFinancialRepository,
)
from src.ai.agent.tools.financial_sql_tool import (
    FinancialQueryInput,
    FinancialSQLTool,
)
from src.core.observability.telemetry import (
    TelemetryLogger,
    sanitize_query,
)


def test_timeline_reducer_parallel_accumulation() -> None:
    """Verify timeline_reducer safely accumulates parallel branches without overwriting."""
    existing = [
        {"step": "Understanding question", "status": "completed", "duration_ms": 45.2},
        {"step": "Resolving metrics", "status": "completed", "duration_ms": 12.0},
    ]
    sql_branch = [{"step": "Querying DuckDB", "status": "completed", "duration_ms": 18.5}]
    rag_branch = [{"step": "Searching SEC filings", "status": "completed", "duration_ms": 32.1}]

    # Simulate fan-out resolution where both branches return concurrently
    step1 = timeline_reducer(existing, sql_branch)
    step2 = timeline_reducer(step1, rag_branch)

    assert len(step2) == 4
    step_names = [e["step"] for e in step2]
    assert "Querying DuckDB" in step_names
    assert "Searching SEC filings" in step_names


def test_mixed_research_parallel_fanout_timeline() -> None:
    """Validate that Mixed Research parallel execution records both tools without event loss."""
    agent = ResearchAgent()
    request = AgentRequest(query="What was Apple's revenue in 2023 and what risks did they disclose regarding China?")

    response = agent.query(request)

    assert response.intent == IntentType.MIXED_RESEARCH
    assert len(response.activity_timeline) == 6

    timeline_map = {step.step: step for step in response.activity_timeline}

    # Verify both concurrent branches finished and were recorded
    assert "Querying DuckDB" in timeline_map
    assert "Searching SEC filings" in timeline_map
    assert timeline_map["Querying DuckDB"].status in ("completed", "failed")
    assert timeline_map["Searching SEC filings"].status in ("completed", "failed")

    # Verify strict canonical ordering
    steps_order = [s.step for s in response.activity_timeline]
    sql_idx = steps_order.index("Querying DuckDB")
    rag_idx = steps_order.index("Searching SEC filings")
    validation_idx = steps_order.index("Validating evidence")

    assert sql_idx < validation_idx
    assert rag_idx < validation_idx


def test_database_execution_error_chaining() -> None:
    """Verify DuckDB repository wraps storage/duckdb errors and preserves exception chain."""
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = duckdb.IOException("Simulated S3 Parquet read failure")

    repo = DuckDBFinancialRepository(connection=mock_conn)
    query = FinancialQueryInput(tickers=["AAPL"], metrics=["revenues"], fiscal_years=[2023], fiscal_periods=["FY"])

    with pytest.raises(DatabaseExecutionError) as exc_info:
        repo.query_metric(query)

    assert "Database query execution failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, duckdb.IOException)


def test_programming_error_not_wrapped() -> None:
    """Verify programming errors (KeyError, TypeError) are NOT wrapped as DatabaseExecutionError."""
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = KeyError("developer_bug_column_missing")

    repo = DuckDBFinancialRepository(connection=mock_conn)
    query = FinancialQueryInput(tickers=["AAPL"], metrics=["revenues"], fiscal_years=[2023], fiscal_periods=["FY"])

    # Must raise KeyError directly, NOT DatabaseExecutionError
    with pytest.raises(KeyError):
        repo.query_metric(query)


def test_database_error_graceful_fallback_and_timeline() -> None:
    """Verify sql_node catches DatabaseExecutionError, sanitizes output, and sets timeline status to failed."""
    mock_tool = MagicMock(spec=FinancialSQLTool)
    mock_tool.execute.side_effect = DatabaseExecutionError("DuckDB connection broken")

    state = {
        "query": "What is Apple's revenue in 2023?",
        "sql_params": {
            "tickers": ["AAPL"],
            "metrics": ["revenues"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
        },
    }

    result = sql_node(state, sql_tool=mock_tool)

    assert result["sql_data"] is None
    assert len(result["tool_results"]) == 1
    tool_res = result["tool_results"][0]
    assert tool_res["status"] == ToolExecutionStatus.TOOL_FAILED.value
    assert tool_res["error_message"] == "Database service temporarily unavailable"

    assert len(result["activity_timeline"]) == 1
    timeline_step = result["activity_timeline"][0]
    assert timeline_step["step"] == "Querying DuckDB"
    assert timeline_step["status"] == "failed"
    assert "Database query interrupted" in timeline_step["description"]


def test_telemetry_logger_jsonl_output(tmp_path: Path) -> None:
    """Verify TelemetryLogger writes valid JSON Lines format with all expected fields."""
    log_file = tmp_path / "test_run_telemetry.jsonl"
    logger = TelemetryLogger(log_path=str(log_file))

    success = logger.log_run(
        request_id="req-12345",
        session_id="sess-abc",
        query="Compare Apple and Microsoft revenues",
        intent="FINANCIAL_METRIC",
        latency_ms=145.2,
        tickers=["AAPL", "MSFT"],
        metrics=["revenues"],
        periods=["FY2023"],
        tools_called=["financial_sql_tool"],
        tool_status={"financial_sql_tool": "success"},
        error_code=None,
    )

    assert success is True
    assert log_file.exists()

    with open(log_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1
    data = json.loads(lines[0])

    assert data["request_id"] == "req-12345"
    assert data["session_id"] == "sess-abc"
    assert data["intent"] == "FINANCIAL_METRIC"
    assert data["tickers"] == ["AAPL", "MSFT"]
    assert data["metrics"] == ["revenues"]
    assert data["latency_ms"] == 145.2
    assert data["error_code"] is None
    assert "timestamp" in data


def test_telemetry_query_sanitization() -> None:
    """Verify sensitive tokens, API keys, emails, and passwords are redacted."""
    raw_query = "Check Apple numbers email john.doe@example.com with key sk-1234567890abcdef1234567890 and password=mysecretpassword"
    sanitized = sanitize_query(raw_query)

    assert "john.doe@example.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "sk-1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_KEY]" in sanitized
    assert "mysecretpassword" not in sanitized
    assert "password=[REDACTED_SECRET]" in sanitized


def test_telemetry_concurrent_multithread_writes(tmp_path: Path) -> None:
    """Verify concurrent writes across multiple threads produce uncorrupted JSONL lines."""
    log_file = tmp_path / "concurrent_telemetry.jsonl"
    logger = TelemetryLogger(log_path=str(log_file))

    def write_worker(idx: int) -> bool:
        return logger.log_run(
            request_id=f"thread-{idx}",
            query=f"Query from worker {idx}",
            intent="FINANCIAL_METRIC",
            latency_ms=10.0 + idx,
        )

    thread_count = 25
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_worker, range(thread_count)))

    assert all(results)

    with open(log_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == thread_count
    # Verify every line parses cleanly as JSON
    for line in lines:
        parsed = json.loads(line)
        assert parsed["request_id"].startswith("thread-")


def test_telemetry_fail_safe_on_write_error() -> None:
    """Verify TelemetryLogger handles inaccessible destinations gracefully without throwing."""
    # Invalid root path that cannot be created or written
    logger = TelemetryLogger(log_path="/sys/kernel/debug/forbidden/telemetry.jsonl")

    # Must return False and not raise any exception
    success = logger.log_run(
        request_id="fail-safe-test",
        query="Does not crash",
        intent="TEST",
        latency_ms=1.0,
    )

    assert success is False
