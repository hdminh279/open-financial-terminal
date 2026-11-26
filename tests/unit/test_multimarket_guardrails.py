"""Unit tests for multi-currency support, contextual isolation, strict abstention, and fiscal footnotes."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from backend.ai.agent.abstention import AbstentionGate
from backend.ai.agent.chart_engine import ChartEngine
from backend.ai.agent.citation import build_fiscal_period_notes, citation_node
from backend.ai.agent.contracts import (
    AbstentionReason,
    EvidenceSourceType,
    IntentType,
    ToolExecutionStatus,
)
from backend.ai.agent.graph import sql_node
from backend.ai.agent.router import (
    IntentRouter,
    has_anaphora_reference,
    is_realtime_news_query,
    should_attach_history,
)
from backend.ai.agent.state import AgentState
from backend.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from backend.ai.agent.tools.financial_sql_tool import FinancialQueryInput, FinancialSQLTool
from backend.ai.llm.contracts import LLMResponse, TokenUsage
from backend.core.financial.response_templates import detect_query_language, render_template
from backend.core.financial.year_utils import is_shifted_fiscal_period

# ---------------------------------------------------------------------------
# Slice 1: Multi-Currency Repository, Tool & Chart Guardrail Tests
# ---------------------------------------------------------------------------


def test_repository_reads_currency_column_across_markets(tmp_path: Path) -> None:
    """Verify DuckDBFinancialRepository preserves VND, TWD, and USD currency metadata."""
    stmt_path = tmp_path / "fct_financial_statements.parquet"
    ratio_path = tmp_path / "fct_financial_ratios.parquet"

    conn = duckdb.connect(":memory:")
    conn.execute(f"""
        COPY (
            SELECT
                'k1' AS statement_key, '0001046179' AS cik, 'TSM' AS ticker,
                'US' AS market, 'TWD' AS currency,
                2024 AS fiscal_year, 'FY' AS fiscal_period, 'annual' AS period_type,
                DATE '2024-01-01' AS start_date, DATE '2024-12-31' AS end_date,
                CAST(2894307699000.00 AS DECIMAL(28,2)) AS revenues,
                CAST(1173267800000.00 AS DECIMAL(28,2)) AS net_income,
                CAST(NULL AS DECIMAL(28,2)) AS operating_income,
                CAST(NULL AS DECIMAL(28,2)) AS gross_profit,
                CAST(NULL AS DECIMAL(28,2)) AS eps_diluted,
                CAST(NULL AS DECIMAL(28,2)) AS cash_and_cash_equivalents,
                CAST(NULL AS DECIMAL(28,2)) AS current_assets,
                CAST(NULL AS DECIMAL(28,2)) AS current_liabilities,
                CAST(NULL AS DECIMAL(28,2)) AS total_assets,
                CAST(NULL AS DECIMAL(28,2)) AS total_liabilities,
                CAST(NULL AS DECIMAL(28,2)) AS stockholders_equity_parent,
                CAST(NULL AS DECIMAL(28,2)) AS stockholders_equity_total,
                FALSE AS is_derived, 'as_reported' AS derivation_method,
                '0001046179-25-000010' AS source_accession_number, DATE '2025-04-17' AS source_filed_date
            UNION ALL
            SELECT
                'k2', 'VN_FPT', 'FPT', 'VN', 'VND',
                2024, 'FY', 'annual',
                DATE '2024-01-01', DATE '2024-12-31',
                CAST(62849000000000.00 AS DECIMAL(28,2)),
                CAST(9427000000000.00 AS DECIMAL(28,2)),
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                FALSE, 'as_reported', 'VN-FPT-2024-FY', DATE '2025-03-15'
        ) TO '{stmt_path}' (FORMAT PARQUET)
    """)
    conn.execute(f"""
        COPY (
            SELECT
                'r1' AS ratio_key, 'VN_FPT' AS cik, 'FPT' AS ticker,
                'VN' AS market, 'VND' AS currency,
                2024 AS fiscal_year, 'FY' AS fiscal_period, 'annual' AS period_type,
                DATE '2024-01-01' AS start_date, DATE '2024-12-31' AS end_date,
                CAST(0.1500 AS DECIMAL(28,2)) AS net_margin,
                CAST(NULL AS DECIMAL(28,2)) AS operating_margin,
                CAST(NULL AS DECIMAL(28,2)) AS gross_margin,
                CAST(NULL AS DECIMAL(28,2)) AS current_ratio,
                CAST(NULL AS DECIMAL(28,2)) AS debt_to_equity,
                CAST(NULL AS DECIMAL(28,2)) AS revenue_growth_yoy
        ) TO '{ratio_path}' (FORMAT PARQUET)
    """)

    repo = DuckDBFinancialRepository(
        statements_path=str(stmt_path),
        ratios_path=str(ratio_path),
        connection=conn,
    )
    tool = FinancialSQLTool(repository=repo)
    resp = tool.execute(
        FinancialQueryInput(tickers=["TSM", "FPT"], metrics=["revenues"], fiscal_years=[2024], fiscal_periods=["FY"])
    )

    by_ticker = {r.ticker: r for r in resp.data}
    assert by_ticker["TSM"].currency == "TWD"
    assert by_ticker["FPT"].currency == "VND"
    assert by_ticker["TSM"].metric_value == Decimal("2894307699000.00")


def test_chart_engine_uses_record_currency_and_blocks_mixed_currency_axis() -> None:
    """Verify ChartEngine labels VND charts accurately and skips mixed-currency combined charts."""
    engine = ChartEngine()

    # Single-currency VND chart -> unit must be VND, not USD
    vnd_sql_data = {
        "data": [
            {
                "ticker": "FPT",
                "metric_name": "revenues",
                "currency": "VND",
                "fiscal_year": 2023,
                "fiscal_period": "FY",
                "period_type": "annual",
                "metric_value": 52618000000000,
            },
            {
                "ticker": "FPT",
                "metric_name": "revenues",
                "currency": "VND",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_type": "annual",
                "metric_value": 62849000000000,
            },
        ]
    }
    vnd_charts = engine.extract_financial_charts(sql_data=vnd_sql_data, chart_preference="line")
    assert len(vnd_charts) == 1
    assert "VND" in vnd_charts[0].unit
    assert "USD" not in vnd_charts[0].unit

    # Mixed-currency monetary chart (FPT in VND vs INFY in USD) -> skipped to prevent misleading axis
    mixed_sql_data = {
        "data": [
            {
                "ticker": "FPT",
                "metric_name": "revenues",
                "currency": "VND",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_type": "annual",
                "metric_value": 62849000000000,
            },
            {
                "ticker": "INFY",
                "metric_name": "revenues",
                "currency": "USD",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_type": "annual",
                "metric_value": 18562000000,
            },
        ]
    }
    mixed_charts = engine.extract_financial_charts(sql_data=mixed_sql_data, chart_preference="bar")
    assert mixed_charts == []


# ---------------------------------------------------------------------------
# Slice 2: Router Contextual Isolation, Semantic News & sql_node Guard Tests
# ---------------------------------------------------------------------------


def test_realtime_news_and_history_attachment_helpers() -> None:
    """Verify high-precision live news detection and history attachment isolation."""
    assert is_realtime_news_query("Cho tôi tin tức mới nhất ngày hôm nay") is True
    assert is_realtime_news_query("Breaking news about NVIDIA today") is True
    assert is_realtime_news_query("Doanh thu FPT năm 2024 là bao nhiêu?") is False

    assert should_attach_history("Cho tôi tin tức mới nhất ngày hôm nay") is False
    assert should_attach_history("Tin tức mới nhất về FPT hôm nay") is False
    assert should_attach_history("Còn Microsoft thì sao?") is True
    assert has_anaphora_reference("What about Microsoft?") is True


def test_router_strips_history_and_leaked_tickers_for_general_live_news() -> None:
    """Verify router isolates standalone live news from prior AAPL conversation history."""
    captured_messages: list[Any] = []

    fake_llm = MagicMock()

    def _fake_generate(messages: list[Any], **kwargs: Any) -> LLMResponse:
        captured_messages.extend(messages)
        # Simulate a misbehaving LLM attempting to leak AAPL + SQL + RAG
        payload = {
            "intent": "MIXED_RESEARCH",
            "is_followup": False,
            "is_realtime_news": True,
            "chart_preference": "none",
            "target_tools": ["financial_sql_tool", "hybrid_retriever", "web_search_tool"],
            "execution_strategy": "parallel",
            "tickers": ["AAPL"],
            "fiscal_years": [2024],
            "fiscal_periods": ["FY"],
            "sql_params": {
                "tickers": ["AAPL"],
                "metrics": ["revenues"],
                "fiscal_years": [2024],
                "fiscal_periods": ["FY"],
            },
            "rag_params": {"search_query": "Apple 2015 news"},
            "web_params": {"search_query": "tin tức tài chính hôm nay", "max_results": 5},
            "reasoning": "Fetching today's news.",
        }
        return LLMResponse(
            content=json.dumps(payload),
            model="deepseek-chat",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
            latency_ms=10.0,
        )

    fake_llm.generate.side_effect = _fake_generate
    router = IntentRouter(llm_adapter=fake_llm)

    history = [
        {"role": "user", "content": "Doanh thu Apple năm 2024?"},
        {"role": "assistant", "content": "Apple FY2024 revenue was 391B USD."},
    ]
    decision = router.route("Cho tôi tin tức mới nhất ngày hôm nay", conversation_history=history)

    # 1. History must NOT be attached to the LLM prompt
    user_msg = captured_messages[-1].content
    assert "Prior Conversation Context" not in user_msg
    assert "Apple" not in user_msg

    # 2. Supervisor must strip leaked ticker and restrict tools strictly to web_search_tool
    assert decision.target_tools == ["web_search_tool"]
    assert decision.tickers == []
    assert decision.sql_params is None
    assert decision.rag_params is None
    assert decision.web_params is not None
    assert decision.is_realtime_news is True


def test_router_preserves_hybrid_sql_and_web_fanout_without_rag() -> None:
    """Verify hybrid revenue + today's news routes to SQL + Web in parallel without spurious RAG."""
    router = IntentRouter()
    raw_llm_json = {
        "intent": "MIXED_RESEARCH",
        "is_followup": False,
        "is_realtime_news": True,
        "chart_preference": "none",
        "target_tools": ["financial_sql_tool", "web_search_tool"],
        "execution_strategy": "parallel",
        "tickers": ["FPT"],
        "fiscal_years": [2024],
        "fiscal_periods": ["FY"],
        "metrics": ["revenues"],
        "sql_params": {"tickers": ["FPT"], "metrics": ["revenues"], "fiscal_years": [2024], "fiscal_periods": ["FY"]},
        "rag_params": None,
        "web_params": {"search_query": "FPT latest news today", "max_results": 5},
        "reasoning": "Retrieving FPT 2024 revenue and today's latest news.",
    }
    decision = router._supervise_and_validate(
        raw_llm_json,
        original_query="Doanh thu FPT năm 2024 và tin tức mới nhất về FPT hôm nay",
    )
    assert decision.intent == IntentType.MIXED_RESEARCH
    assert decision.target_tools == ["financial_sql_tool", "web_search_tool"]
    assert "hybrid_retriever" not in decision.target_tools
    assert decision.sql_params is not None
    assert decision.web_params is not None


def test_sql_node_does_not_leak_history_ticker_when_not_followup() -> None:
    """Verify sql_node refuses to mine conversation_history when is_followup is False."""
    mock_tool = MagicMock(spec=FinancialSQLTool)
    state: AgentState = {
        "query": "Cho tôi tin tức mới nhất ngày hôm nay",
        "sql_params": None,
        "routing_decision": {"tickers": [], "is_followup": False},
        "conversation_history": [{"role": "user", "content": "Doanh thu Apple (AAPL) năm 2024 là bao nhiêu?"}],
    }

    result = sql_node(state, sql_tool=mock_tool)
    mock_tool.execute.assert_not_called()
    assert result["sql_data"] is None
    assert result["tool_results"][0]["status"] == ToolExecutionStatus.NO_DATA_FOUND.value


# ---------------------------------------------------------------------------
# Slice 3: Bilingual Templates, Strict Abstention & Fiscal Auto-Footnote Tests
# ---------------------------------------------------------------------------


def test_bilingual_language_detection_and_templates() -> None:
    """Verify language detection and externalized YAML response template rendering."""
    assert detect_query_language("Cho tôi tin tức mới nhất ngày hôm nay") == "vi"
    assert detect_query_language("Give me today's breaking news") == "en"

    vi_msg = render_template("web_news_abstention", lang="vi")
    en_msg = render_template("web_news_abstention", lang="en")
    assert "không có tin tức thời sự" in vi_msg
    assert "breaking news" in en_msg.lower()


@pytest.mark.parametrize(
    "fy,start_d,end_d,expected_shifted,expected_cal_year",
    [
        # NVDA Q1 FY2027 (100% inside CY2026) -> shifted
        (2027, date(2026, 1, 26), date(2026, 4, 26), True, 2026),
        # NVDA FY2025 (~92.8% inside CY2024) -> shifted
        (2025, date(2024, 1, 29), date(2025, 1, 26), True, 2024),
        # AAPL FY2024 (~75% inside CY2024 == fy) -> NOT shifted
        (2024, date(2023, 10, 1), date(2024, 9, 28), False, 2024),
        # MSFT FY2024 (~50.3% inside CY2023 < 75%) -> NOT shifted
        (2024, date(2023, 7, 1), date(2024, 6, 30), False, 2023),
        # FPT FY2024 (100% inside CY2024) -> NOT shifted
        (2024, date(2024, 1, 1), date(2024, 12, 31), False, 2024),
        # Instant Balance Sheet (start_date=None) -> NOT shifted
        (2024, None, date(2024, 12, 31), False, None),
    ],
)
def test_shifted_fiscal_period_detection(
    fy: int,
    start_d: Any,
    end_d: Any,
    expected_shifted: bool,
    expected_cal_year: Any,
) -> None:
    """Verify data-driven shifted fiscal calendar detection across major issuers."""
    shifted, cal_year = is_shifted_fiscal_period(fy, start_d, end_d)
    assert shifted is expected_shifted
    assert cal_year == expected_cal_year


def test_abstention_gate_strict_web_only_and_hybrid_sql_web() -> None:
    """Verify decisive web-only abstention and clean hybrid SQL+Web progression."""
    gate = AbstentionGate()

    # 1. Pure web search yields empty results -> decisive VI abstention
    empty_web_state: AgentState = {
        "query": "Cho tôi tin tức mới nhất ngày hôm nay",
        "intent": IntentType.TEXTUAL_RESEARCH.value,
        "routing_decision": {"tickers": [], "target_tools": ["web_search_tool"]},
        "tool_results": [{"tool_name": "web_search_tool", "status": ToolExecutionStatus.NO_DATA_FOUND.value}],
        "evidence_items": [],
    }
    info_empty = gate.evaluate(empty_web_state)
    assert info_empty.is_abstaining is True
    assert info_empty.is_partial is False
    assert info_empty.reason == AbstentionReason.INSUFFICIENT_EVIDENCE
    assert "không có tin tức thời sự" in info_empty.explanation

    # 2. Hybrid SQL + Web both succeed (RAG not invoked) -> NO false partial abstention
    hybrid_ok_state: AgentState = {
        "query": "Doanh thu FPT năm 2024 và tin tức mới nhất về FPT hôm nay",
        "intent": IntentType.MIXED_RESEARCH.value,
        "routing_decision": {"tickers": ["FPT"], "target_tools": ["financial_sql_tool", "web_search_tool"]},
        "tool_results": [
            {"tool_name": "financial_sql_tool", "status": ToolExecutionStatus.SUCCESS.value},
            {"tool_name": "web_search_tool", "status": ToolExecutionStatus.SUCCESS.value},
        ],
        "evidence_items": [
            {"citation_id": 1, "source_type": EvidenceSourceType.SQL.value},
            {"citation_id": 2, "source_type": EvidenceSourceType.WEB.value},
        ],
    }
    info_hybrid = gate.evaluate(hybrid_ok_state)
    assert info_hybrid.is_abstaining is False
    assert info_hybrid.is_partial is False


def test_citation_node_appends_deduped_fiscal_period_footnote() -> None:
    """Verify citation_node deterministically attaches 1 localized footnote per shifted ticker."""
    sql_data = {
        "data": [
            {
                "ticker": "NVDA",
                "fiscal_year": 2027,
                "fiscal_period": "Q1",
                "start_date": "2026-01-26",
                "end_date": "2026-04-26",
                "metric_name": "revenues",
                "metric_value": "44062000000.00",
                "currency": "USD",
            },
            {
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "fiscal_period": "FY",
                "start_date": "2025-01-27",
                "end_date": "2026-01-25",
                "metric_name": "revenues",
                "metric_value": "190000000000.00",
                "currency": "USD",
            },
            {
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "start_date": "2023-10-01",
                "end_date": "2024-09-28",
                "metric_name": "revenues",
                "metric_value": "391035000000.00",
                "currency": "USD",
            },
        ]
    }

    notes = build_fiscal_period_notes(sql_data=sql_data, query="Doanh thu NVIDIA FY2027 là bao nhiêu?")
    assert len(notes) == 1
    assert "NVDA" in notes[0]
    assert "FY2027" in notes[0]
    assert "2026" in notes[0]
    assert "AAPL" not in notes[0]

    state: AgentState = {
        "query": "Doanh thu NVIDIA FY2027 là bao nhiêu?",
        "intent": IntentType.FINANCIAL_METRIC.value,
        "draft_answer": "Doanh thu quý Q1 FY2027 của NVIDIA đạt 44.06 tỷ USD [1].",
        "sql_data": sql_data,
        "evidence_items": [],
        "routing_decision": {"tickers": ["NVDA"], "chart_preference": "none"},
        "is_abstaining": False,
        "is_partial": False,
    }
    out = citation_node(state)
    assert "Năm tài chính của Nvidia Corporation (NVDA) lệch so với năm dương lịch" in out["final_answer"]


def test_multimarket_offline_all_20_scenarios_pass() -> None:
    """Verify full offline multi-market benchmark achieves 20/20 pass rate."""
    from tests.benchmark.benchmark_qg17 import run_qg17_benchmark

    results = run_qg17_benchmark(mode="offline", write_report=False)
    assert len(results) == 20
    failures = [(r.question_id, r.failure_causes) for r in results if not r.passed]
    assert failures == [], f"Multi-market offline benchmark failures: {failures}"


def test_q1_to_q20_audit_fixes(tmp_path: Path) -> None:
    """Verify VN30 provenance URLs, 20-F/BCTC forms, ratio 4-decimal precision, macro news & FY footnote."""
    from backend.ai.agent.graph import _resolve_filing_form
    from backend.ai.agent.tools.financial_sql_tool import generate_edgar_url

    # 1. VN30 URL must point to VNDirect, never SEC.gov
    vn_url = generate_edgar_url(cik="VN:VCB", accession_number="7fd826ee8132b81c517412746ed8997a")
    assert vn_url == "https://api-finfo.vndirect.com.vn/v4/financial_statements?q=code:VCB"
    assert _resolve_filing_form("VCB", "VN:VCB", "FY") == "BCTC"
    assert _resolve_filing_form("TM", "0001094517", "FY") == "20-F"
    assert _resolve_filing_form("TM", "0001094517", "Q4") == "20-F"
    assert _resolve_filing_form("AAPL", "0000320193", "FY") == "10-K"
    assert _resolve_filing_form("AAPL", "0000320193", "Q4") == "10-K"
    assert _resolve_filing_form("AAPL", "0000320193", "Q1") == "10-Q"

    # 2. Macro economic news triggers realtime news detection
    assert is_realtime_news_query("Tin tức kinh tế nổi bật hôm nay là gì?") is True

    # 3. Shifted FY footnote avoids duplicate 'FY FY2024'
    sql_data = {
        "data": [
            {
                "ticker": "TM",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "start_date": "2023-04-01",
                "end_date": "2024-03-28",
                "metric_name": "revenues",
                "metric_value": "45095325000000.00",
                "currency": "JPY",
            }
        ]
    }
    notes = build_fiscal_period_notes(sql_data=sql_data, query="Doanh thu Toyota FY2024?")
    assert len(notes) == 1
    assert "FY2024" in notes[0]
    assert "FY FY2024" not in notes[0]

    # 4. Ratio query preserves 4 decimal places and reads source_accession_number
    stmt_p = str(tmp_path / "stmt.parquet")
    ratio_p = str(tmp_path / "ratio.parquet")
    conn = duckdb.connect(":memory:")
    conn.execute(
        f"COPY (SELECT 'k1' AS statement_key, 'TM' AS ticker, '0001094517' AS cik, 'JPY' AS currency, "
        f"2024 AS fiscal_year, 'FY' AS fiscal_period, 'annual' AS period_type, "
        f"DATE '2023-04-01' AS start_date, DATE '2024-03-31' AS end_date, "
        f"CAST(45095325000000.00 AS DECIMAL(28,2)) AS revenues, FALSE AS is_derived, "
        f"'as_reported' AS derivation_method, '0001193125-24-167462' AS source_accession_number, "
        f"DATE '2024-06-25' AS source_filed_date) TO '{stmt_p}' (FORMAT PARQUET)"
    )
    conn.execute(
        f"COPY (SELECT 'k1' AS ratio_key, 'TM' AS ticker, '0001094517' AS cik, 'JPY' AS currency, "
        f"2024 AS fiscal_year, 'FY' AS fiscal_period, 'annual' AS period_type, "
        f"DATE '2023-04-01' AS start_date, DATE '2024-03-31' AS end_date, "
        f"0.1125::DOUBLE AS net_margin, '0001193125-24-167462' AS source_accession_number, "
        f"DATE '2024-06-25' AS source_filed_date) TO '{ratio_p}' (FORMAT PARQUET)"
    )
    repo = DuckDBFinancialRepository(statements_path=stmt_p, ratios_path=ratio_p)
    rows = repo.query_metric(
        FinancialQueryInput(tickers=["TM"], metrics=["net_margin"], fiscal_years=[2024], fiscal_periods=["FY"])
    )
    assert len(rows) == 1
    assert rows[0]["metric_value"] == Decimal("0.1125")
    assert rows[0]["source_accession_number"] == "0001193125-24-167462"


def test_chat2_rag_table_compaction_form_filter_and_web_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RAG table compaction preserves numbers, 10-K inference, and web_node error propagation."""
    from backend.ai.agent.graph import _compact_rag_chunk_text, _infer_rag_form, web_node
    from backend.ai.agent.tools.web_search_tool import WebSearchTool
    from pipelines.processing.table_processor import TableProcessor

    raw_table = (
        "[Financial Table]\n"
        "|  |  |  |  |  |  |  |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "|  | 2024 |  | 2023 |  |  |  |\n"
        "| iPhone | $ | 201,183 | $ | 200,583 |  |  |\n"
        "| Services |  | 96,169 |  | 85,200 |  |  |"
    )
    compacted = _compact_rag_chunk_text(raw_table, max_chars=3000)
    assert "201,183" in compacted
    assert "96,169" in compacted
    assert "|  |  |  |" not in compacted

    # TableProcessor drops blank spacer columns on ingestion too
    tp = TableProcessor()
    md = tp.format_markdown_table(
        headers=["", "2024", "", "2023"],
        rows=[["iPhone", "201,183", "", "200,583"]],
    )
    assert "201,183" in md
    assert "| 2024 | 2023 |" in md
    assert "| iPhone | 201,183 | 200,583 |" in md

    assert _infer_rag_form("Trong báo cáo 10-K năm 2024 của Microsoft", None, ["MSFT"]) == "10-K"
    assert _infer_rag_form("Rủi ro chính của Apple là gì?", None, ["AAPL"]) == "10-K"
    assert _infer_rag_form("Doanh thu quý 2 10-Q của Apple", None, ["AAPL"]) is None

    # WebSearchTool error propagates to web_node as TOOL_FAILED
    import backend.ai.agent.tools.web_search_tool as wst

    monkeypatch.setattr(wst, "DDGS", None)
    tool = WebSearchTool()
    res = tool.search("NVIDIA news")
    assert res.error is not None
    node_out = web_node({"query": "NVIDIA news"}, web_tool=tool)
    assert node_out["tool_results"][0]["status"] == ToolExecutionStatus.TOOL_FAILED.value


def test_points_2_3_4_web_rss_cache_multiyear_and_vn30_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RSS fallback + TTL cache, LLM fiscal_years=[] preservation, and VN30 30-ticker expansion."""
    import backend.ai.agent.tools.web_search_tool as wst
    from backend.ai.agent.graph import sql_node
    from backend.ai.agent.tools.web_search_tool import WebSearchTool

    # 1. Point 2: Google News RSS fallback when DDGS rate-limits + TTL cache hit
    mock_rss_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss><channel><item>"
        "<title>Apple News Today</title>"
        "<link>https://news.google.com/articles/apple-today</link>"
        "<description>Apple launches new chip.</description>"
        "<pubDate>Sun, 22 Nov 2026 08:00:00 GMT</pubDate>"
        "</item></channel></rss>"
    )
    mock_resp = MagicMock()
    mock_resp.text = mock_rss_xml
    mock_resp.raise_for_status = MagicMock()
    mock_get = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(wst, "DDGS", None)
    monkeypatch.setattr(wst.httpx, "get", mock_get)

    web_tool = WebSearchTool(timeout=2, max_retries=1, cache_ttl_seconds=300, enable_rss_fallback=True)
    r1 = web_tool.search("Apple news today", max_results=3)
    assert r1.error is None
    assert r1.total_found == 1
    assert r1.results[0].url == "https://news.google.com/articles/apple-today"
    # Second identical call hits TTL cache (no extra httpx.get call)
    r2 = web_tool.search("Apple news today", max_results=3)
    assert r2.total_found == 1
    assert mock_get.call_count == 1

    def _make_fake_llm(payload_json: str) -> MagicMock:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            content=payload_json,
            model="deepseek-chat",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
            latency_ms=10.0,
        )
        return mock_llm

    # 2. Point 3: Router & sql_node preserve LLM's fiscal_years=[] instead of clamping to [2024]
    ba_payload = json.dumps(
        {
            "intent": "FINANCIAL_METRIC",
            "is_followup": False,
            "is_realtime_news": False,
            "chart_preference": "none",
            "target_tools": ["financial_sql_tool"],
            "execution_strategy": "parallel",
            "tickers": ["BA"],
            "fiscal_years": [],
            "fiscal_periods": ["FY"],
            "sql_params": {
                "tickers": ["BA"],
                "metrics": ["revenues"],
                "fiscal_years": [],
                "fiscal_periods": ["FY"],
            },
            "reasoning": "Fetch all available years to compare 2024 with prior 3 years.",
        }
    )
    ba_router = IntentRouter(llm_adapter=_make_fake_llm(ba_payload))
    ba_decision = ba_router.route("Doanh thu của BA năm 2024 là bao nhiêu? Hãy so sánh với kết quả 3 năm trước đó.")
    assert ba_decision.fiscal_years == []
    assert ba_decision.sql_params is not None
    assert ba_decision.sql_params.fiscal_years == []

    # sql_node also keeps fiscal_years=[] instead of overwriting with [2024]
    captured_queries = []

    class _CaptureSQLTool:
        def execute(self, q):
            captured_queries.append(q)
            from backend.ai.agent.tools.financial_sql_tool import FinancialQueryResponse

            return FinancialQueryResponse(query=q, record_count=0, data=[])

    sql_node(
        {
            "query": "Doanh thu của BA năm 2024 là bao nhiêu? Hãy so sánh với kết quả 3 năm trước đó.",
            "sql_params": ba_decision.sql_params.model_dump(),
            "query_plan": ba_decision.query_plan.model_dump(),
            "routing_decision": ba_decision.model_dump(),
        },
        sql_tool=_CaptureSQLTool(),
    )
    assert len(captured_queries) == 1
    assert captured_queries[0].fiscal_years == []

    # 3. Point 4: VN30 basket auto-expands to all 30 tickers + includes total_operating_income
    vn30_payload = json.dumps(
        {
            "intent": "FINANCIAL_METRIC",
            "is_followup": False,
            "is_realtime_news": False,
            "chart_preference": "none",
            "target_tools": ["financial_sql_tool"],
            "execution_strategy": "parallel",
            "tickers": ["FPT", "VNM"],
            "fiscal_years": [2024],
            "fiscal_periods": ["FY"],
            "sql_params": {
                "tickers": ["FPT", "VNM"],
                "metrics": ["revenues"],
                "fiscal_years": [2024],
                "fiscal_periods": ["FY"],
            },
            "reasoning": "Query 2024 revenues for VN30 companies.",
        }
    )
    vn30_router = IntentRouter(llm_adapter=_make_fake_llm(vn30_payload))
    vn30_decision = vn30_router.route("Liệt kê doanh thu năm 2024 của toàn bộ 30 công ty VN30 trong hệ thống.")
    assert len(vn30_decision.tickers) == 30
    assert "BCM" in vn30_decision.tickers
    assert "VJC" in vn30_decision.tickers
    assert "VCB" in vn30_decision.tickers
    assert vn30_decision.sql_params is not None
    assert len(vn30_decision.sql_params.tickers) == 30
    assert "revenues" in vn30_decision.sql_params.metrics
    assert "total_operating_income" in vn30_decision.sql_params.metrics
