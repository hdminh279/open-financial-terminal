"""Unit tests for Day 20 production hardening, Hive text pipeline, modular logging, and cost telemetry."""

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from src.ai.agent.contracts import AgentRequest, IntentType
from src.ai.agent.research_agent import ResearchAgent
from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.llm.contracts import LLMResponse, TokenUsage
from src.ai.llm.deepseek_adapter import DeepSeekAdapter
from src.core.observability.logging_config import configure_logging
from src.core.observability.telemetry import (
    TelemetryLogger,
    estimate_deepseek_cost_usd,
    summarize_traffic_telemetry,
)
from src.data.chunking.chunking_engine import TextChunk
from src.data.pipelines.text_pipeline import FilingsTextPipeline


def test_duckdb_concurrent_cursor_queries() -> None:
    """Verify DuckDBFinancialRepository executes parallel read-only queries safely using cursors."""
    repo = DuckDBFinancialRepository(
        statements_path="tests/fixtures/gold/fct_financial_statements.parquet",
        ratios_path="tests/fixtures/gold/fct_financial_ratios.parquet",
    )

    def _run(i: int) -> int:
        rows = repo.execute_fetchall("SELECT ? AS val", [i])
        return int(rows[0][0])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_run, range(16)))
    assert results == list(range(16))


def test_text_pipeline_hive_partitioning_and_dead_letter(tmp_path: Path) -> None:
    """Verify FilingsTextPipeline writes per-ticker Hive Parquet and isolates failed filings in dead-letter."""
    stored_objects: dict[tuple[str, str], bytes] = {}

    mock_s3 = MagicMock()
    mock_s3.list_objects.return_value = [
        "raw/filings/AAPL/10-K_2024-09-28_0001.htm",
        "raw/filings/MSFT/10-K_2024-06-30_bad.htm",
    ]
    mock_s3.object_exists.side_effect = lambda b, k: (b, k) in stored_objects
    mock_s3.download_bytes.side_effect = lambda b, k: stored_objects[(b, k)]

    def _upload(bucket: str, key: str, data: bytes, **_kwargs: object) -> bool:
        stored_objects[(bucket, key)] = data
        return True

    mock_s3.upload_bytes.side_effect = _upload

    pipeline = FilingsTextPipeline(
        s3_storage=mock_s3,
        dead_letter_dir=tmp_path / "dead_letter",
    )

    sample_chunk = TextChunk(
        chunk_id="aapl-c1",
        cik="0000320193",
        ticker="AAPL",
        accession_number="0001",
        form="10-K",
        fiscal_year=2024,
        filed_date="2024-11-01",
        period_end="2024-09-28",
        part="PART I",
        item="ITEM 1",
        subsection=None,
        content_type="text",
        chunk_index=0,
        token_count=42,
        text="Apple Inc. designs smartphones.",
        source_url="https://www.sec.gov/sample",
        source_content_hash="abc123",
    )

    def _fake_process_filing(key: str, force: bool = False):
        if "bad.htm" in key:
            raise ValueError("Simulated corrupt HTML filing")
        return [sample_chunk], {
            "ticker": "AAPL",
            "accession_number": "0001",
            "form": "10-K",
            "fiscal_year": 2024,
            "toc_found": True,
            "sections_extracted": 1,
            "tables_extracted": 0,
            "total_chunks": 1,
            "unidentified_sections": [],
        }

    pipeline.process_filing = _fake_process_filing  # type: ignore[method-assign]
    summary = pipeline.process_all()

    assert summary["hive_partitioned"] is True
    assert summary["partitions_written"] == ["chunks/ticker=AAPL/chunks.parquet"]
    assert summary["total_chunks_in_gold"] == 1
    assert summary["dead_letter_count"] == 1
    assert summary["dead_letter_entries"][0]["ticker"] == "MSFT"

    part_bytes = stored_objects[("sec-gold", "chunks/ticker=AAPL/chunks.parquet")]
    table = pq.read_table(io.BytesIO(part_bytes))
    assert table.num_rows == 1
    assert table["chunk_id"].to_pylist() == ["aapl-c1"]

    dl_local = tmp_path / "dead_letter" / "text_pipeline_dead_letter.json"
    assert dl_local.exists()
    dl_payload = json.loads(dl_local.read_text(encoding="utf-8"))
    assert dl_payload["failed_count"] == 1


def test_modular_rotating_file_logging(tmp_path: Path) -> None:
    """Verify configure_logging routes API, Agent, and Pipeline logs into separate files."""
    paths = configure_logging(log_dir=tmp_path, level=logging.INFO, max_bytes=1024 * 1024, backup_count=2, force=True)
    assert set(paths.keys()) == {"api.log", "agent.log", "pipeline.log"}

    logging.getLogger("src.api.test").info("api log entry")
    logging.getLogger("src.ai.test").info("agent log entry")
    logging.getLogger("src.data.test").info("pipeline log entry")

    for h in (
        logging.getLogger("src.api").handlers
        + logging.getLogger("src.ai").handlers
        + logging.getLogger("src.data").handlers
    ):
        h.flush()

    assert "api log entry" in (tmp_path / "api.log").read_text(encoding="utf-8")
    assert "agent log entry" in (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "pipeline log entry" in (tmp_path / "pipeline.log").read_text(encoding="utf-8")


def test_token_usage_and_cost_telemetry_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DeepSeek TokenUsage propagates through ResearchAgent into telemetry and /api/telemetry summary."""
    cost = estimate_deepseek_cost_usd(
        model="deepseek-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cache_hit_tokens=500_000,
        cache_miss_tokens=500_000,
    )
    # 0.5M * 0.003 + 0.5M * 0.15 + 1.0M * 0.60 = 0.0015 + 0.075 + 0.60 = 0.6765
    assert cost == pytest.approx(0.6765, rel=1e-6)

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(DeepSeekAdapter, "__init__", lambda self, *args, **kwargs: None)

    def _fake_generate(self: DeepSeekAdapter, messages: list, json_mode: bool = False, **kwargs: object) -> LLMResponse:
        if json_mode:
            payload = {
                "intent": "FINANCIAL_METRIC",
                "tickers": ["AAPL"],
                "metrics": ["revenues"],
                "fiscal_years": [2024],
                "fiscal_periods": ["FY"],
                "requires_sql": True,
                "requires_rag": False,
                "comparison_type": "none",
                "chart_preference": "none",
            }
            return LLMResponse(
                content=json.dumps(payload),
                model="deepseek-flash",
                finish_reason="stop",
                latency_ms=5.0,
                usage=TokenUsage(
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_tokens=1100,
                    cache_hit_tokens=800,
                    cache_miss_tokens=200,
                ),
            )
        return LLMResponse(
            content="Apple FY2024 revenue was $391.04B [1].",
            model="deepseek-flash",
            finish_reason="stop",
            latency_ms=10.0,
            usage=TokenUsage(
                prompt_tokens=2000,
                completion_tokens=200,
                total_tokens=2200,
                cache_hit_tokens=1200,
                cache_miss_tokens=800,
            ),
        )

    monkeypatch.setattr(DeepSeekAdapter, "generate", _fake_generate)

    agent = ResearchAgent()
    resp = agent.query(AgentRequest(query="What was Apple's revenue in FY2024?"))
    assert resp.intent == IntentType.FINANCIAL_METRIC
    assert resp.token_usage is not None
    assert resp.token_usage.prompt_tokens == 3000
    assert resp.token_usage.completion_tokens == 300
    assert resp.token_usage.cache_hit_tokens == 2000
    assert resp.token_usage.total_tokens == 3300
    assert resp.token_usage.estimated_cost_usd > 0.0

    run_log = tmp_path / "run_telemetry.jsonl"
    tlogger = TelemetryLogger(log_path=str(run_log))
    tlogger.log_run(
        request_id="req-cost-1",
        session_id="sess-1",
        query="Apple revenue FY2024",
        intent="FINANCIAL_METRIC",
        latency_ms=15.0,
        prompt_tokens=resp.token_usage.prompt_tokens,
        completion_tokens=resp.token_usage.completion_tokens,
        cache_hit_tokens=resp.token_usage.cache_hit_tokens,
        estimated_cost_usd=resp.token_usage.estimated_cost_usd,
    )

    summary = summarize_traffic_telemetry(run_log_path=run_log, persist=False)
    assert summary["total_prompt_tokens"] == 3000
    assert summary["total_completion_tokens"] == 300
    assert summary["total_cache_hit_tokens"] == 2000
    assert summary["cache_hit_rate_pct"] == pytest.approx(66.67, rel=1e-2)
    assert summary["total_estimated_cost_usd"] == pytest.approx(resp.token_usage.estimated_cost_usd, abs=1e-6)
    assert summary["avg_cost_per_request_usd"] == pytest.approx(resp.token_usage.estimated_cost_usd, abs=1e-6)
