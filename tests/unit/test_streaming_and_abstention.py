"""Unit tests for Airflow orchestration config/DAGs, SSE streaming, and negative-RAG abstention."""

import json
import py_compile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from src.ai.agent.citation import citation_node, detect_insufficient_evidence
from src.ai.agent.contracts import IntentType
from src.ai.agent.research_agent import ResearchAgent
from src.ai.agent.router import IntentRouter
from src.ai.llm.contracts import LLMStreamChunk, TokenUsage
from src.ai.llm.deepseek_adapter import DeepSeekAdapter
from src.api.chat_history import ChatHistoryManager
from src.api.deps import get_chat_history, get_research_agent
from src.api.main import app


def test_airflow_compose_and_dags_syntax() -> None:
    """Verify docker-compose.airflow.yml uses LocalExecutor + Postgres 16 and DAGs compile cleanly."""
    compose = yaml.safe_load(Path("docker-compose.airflow.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "airflow-postgres" in services
    assert "postgres:16" in services["airflow-postgres"]["image"]
    assert "airflow-webserver" in services
    assert "airflow-scheduler" in services
    assert services["airflow-scheduler"]["environment"]["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
    assert compose["networks"]["cfi_internal"]["external"] is True

    for dag_file in ("dags/dag_sec_monthly_ingestion.py", "dags/dag_vn30_metrics_sync.py"):
        py_compile.compile(dag_file, doraise=True)
        src_text = Path(dag_file).read_text(encoding="utf-8")
        assert "DAG(" in src_text


def test_rag_negative_router_and_citation_abstention_fixes() -> None:
    """Verify P50-22 style filing questions stay TEXTUAL_RESEARCH and P50-21/23/25 trigger abstention."""
    router = IntentRouter(llm_adapter=MagicMock())

    # Simulate LLM wrongly classifying P50-22 (Microsoft classified contracts) as OUT_OF_SCOPE
    decision = router._supervise_and_validate(
        {"intent": "OUT_OF_SCOPE", "tickers": ["MSFT"], "reasoning": "Unlikely topic"},
        "What were Microsoft's classified contracts for commercial nuclear submarine construction in 1998?",
    )
    assert decision.intent == IntentType.TEXTUAL_RESEARCH
    assert "hybrid_retriever" in decision.target_tools

    # Genuine investment advice for a supported ticker remains OUT_OF_SCOPE
    advice_decision = router._supervise_and_validate(
        {"intent": "OUT_OF_SCOPE", "tickers": ["MSFT"], "reasoning": "Investment advice"},
        "Should I buy MSFT stock tomorrow?",
    )
    assert advice_decision.intent == IntentType.OUT_OF_SCOPE

    # Citation node promotes [INSUFFICIENT_EVIDENCE] and strips tag + clears citations
    res_tagged = citation_node(
        {
            "query": "What did Apple disclose about secret Mars colony real estate?",
            "intent": IntentType.TEXTUAL_RESEARCH.value,
            "is_abstaining": False,
            "is_partial": False,
            "draft_answer": "[INSUFFICIENT_EVIDENCE] Apple's 10-K contains no disclosure of secret Mars colony investments [1].",
            "evidence_items": [
                {
                    "citation_id": 1,
                    "source_type": "rag",
                    "ticker": "AAPL",
                    "form": "10-K",
                    "fiscal_year": 2024,
                    "item_or_metric": "ITEM 1A",
                    "content_summary": "Supply chain risks...",
                    "source_url": "https://www.sec.gov/sample",
                }
            ],
            "tool_results": [{"tool_name": "hybrid_retriever", "status": "success"}],
            "routing_decision": {"tickers": ["AAPL"]},
        }
    )
    assert res_tagged["is_abstaining"] is True
    assert res_tagged["abstention_info"]["reason"] == "insufficient_evidence"
    assert res_tagged["citations"] == []
    assert "[INSUFFICIENT_EVIDENCE]" not in res_tagged["final_answer"]

    # Positive RAG answer with factual disclosures must NOT be flagged as insufficient evidence
    flagged_pos, _ = detect_insufficient_evidence(
        "### Official Disclosures\nApple discloses supply chain concentration risks in Item 1A [1].",
        is_already_abstaining=False,
        is_partial=False,
        has_sql_success=False,
    )
    assert flagged_pos is False

    # Partial/substantive answer with Markdown table or SQL success keeps citations even if tagged
    res_partial_table = citation_node(
        {
            "query": "Cho tôi tin tức mới nhất về FPT hôm nay",
            "intent": IntentType.TEXTUAL_RESEARCH.value,
            "is_abstaining": False,
            "is_partial": False,
            "draft_answer": (
                "[INSUFFICIENT_EVIDENCE] Chưa có bài viết chi tiết trong ngày, nhưng nguồn [1] ghi nhận:\n\n"
                "| Nội dung | Nguồn |\n| :--- | :---: |\n| FPT chia cổ tức tháng 12 | [1] |"
            ),
            "evidence_items": [
                {
                    "citation_id": 1,
                    "source_type": "web",
                    "ticker": "FPT",
                    "form": "WEB_NEWS",
                    "item_or_metric": "FPT News",
                    "content_summary": "FPT dividend plan...",
                    "source_url": "https://cafef.vn/fpt-news.htm",
                }
            ],
            "tool_results": [{"tool_name": "web_search_tool", "status": "success"}],
            "routing_decision": {"tickers": ["FPT"]},
        }
    )
    assert not res_partial_table.get("is_abstaining", False)
    assert len(res_partial_table["citations"]) == 1
    assert res_partial_table["citations"][0]["source_type"] == "web"
    assert "[INSUFFICIENT_EVIDENCE]" not in res_partial_table["final_answer"]


def test_sse_stream_endpoint_and_agent_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ResearchAgent.stream_query and POST /api/query/stream emit status, token, and done SSE frames."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(DeepSeekAdapter, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setenv("ENABLE_S3_CHAT_HISTORY", "false")

    def _fake_generate(self: DeepSeekAdapter, messages: list, json_mode: bool = False, **kwargs: object):
        from src.ai.llm.contracts import LLMResponse

        return LLMResponse(
            content=json.dumps(
                {
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
            ),
            model="deepseek-flash",
            finish_reason="stop",
            latency_ms=3.0,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250, cache_hit_tokens=100),
        )

    def _fake_stream(self: DeepSeekAdapter, messages: list, **kwargs: object):
        yield LLMStreamChunk(content_delta="Apple FY2024 ")
        yield LLMStreamChunk(
            content_delta="revenue was $391.04B [1].",
            is_final=True,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=500, completion_tokens=40, total_tokens=540, cache_hit_tokens=300),
        )

    monkeypatch.setattr(DeepSeekAdapter, "generate", _fake_generate)
    monkeypatch.setattr(DeepSeekAdapter, "stream", _fake_stream)

    agent = ResearchAgent()
    mgr = ChatHistoryManager(s3_storage=None)
    mgr._s3_enabled = False
    mgr._s3 = None

    app.dependency_overrides[get_research_agent] = lambda: agent
    app.dependency_overrides[get_chat_history] = lambda: mgr
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/query/stream",
            json={"query": "What was Apple's revenue in FY2024?", "session_id": "sse-sess-1"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        assert "event: status" in body
        assert "event: token" in body
        assert "event: done" in body
        assert "Apple FY2024 " in body
        assert "revenue was $391.04B [1]." in body
    finally:
        app.dependency_overrides.pop(get_research_agent, None)
        app.dependency_overrides.pop(get_chat_history, None)
