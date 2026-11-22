"""Unit tests for AI Financial Terminal UI/UX & Clean Formatting."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.ai.agent.citation import CitationEngine, citation_node, sanitize_answer_formatting
from src.ai.agent.contracts import (
    ActivityTimelineStep,
    AgentResponse,
    Citation,
    EvidenceSourceType,
    IntentType,
)
from src.api.chat_history import ChatHistoryManager
from src.api.deps import get_chat_history, get_research_agent
from src.api.main import app


def test_sanitize_answer_formatting_removes_dividers_and_header_brackets() -> None:
    """Verify horizontal rules and bracketed section headings are stripped while keeping citations."""
    raw = (
        "### [Công bố Chính thức & Bằng chứng]\n"
        "Doanh thu NVIDIA đạt 215,9 tỷ USD năm FY2026 [107].\n\n"
        "---\n\n"
        "### [Phân tích & Bình luận]\n"
        "Microsoft duy trì tăng trưởng 15-18%/năm [75].\n"
        "***\n"
        "### [Official Disclosures]\n"
        "Apple revenue was $478.8B in FY2026 [45]."
    )

    cleaned = sanitize_answer_formatting(raw)

    assert "---" not in cleaned
    assert "***" not in cleaned
    assert "### Công bố Chính thức & Bằng chứng" in cleaned
    assert "### Phân tích & Bình luận" in cleaned
    assert "### Official Disclosures" in cleaned
    assert "[Công bố Chính thức & Bằng chứng]" not in cleaned
    assert "[Phân tích & Bình luận]" not in cleaned
    assert "[107]" in cleaned
    assert "[75]" in cleaned
    assert "[45]" in cleaned


def test_build_citations_populates_drawer_metadata() -> None:
    """Verify CitationEngine preserves excerpt, ticker, form, period, and accession for UI drawer."""
    evidence = [
        {
            "citation_id": 1,
            "source_type": EvidenceSourceType.SQL.value,
            "ticker": "NVDA",
            "form": "10-K",
            "fiscal_year": 2026,
            "fiscal_period": "FY",
            "item_or_metric": "revenues",
            "content_summary": "NVDA FY2026 revenues = 215,900,000,000 USD",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000010/nvda.htm",
            "accession_number": "0001045810-26-000010",
        }
    ]

    citations = CitationEngine.build_citations(evidence)
    assert len(citations) == 1
    cit = citations[0]
    assert cit.ticker == "NVDA"
    assert cit.form == "10-K"
    assert cit.fiscal_year == 2026
    assert cit.fiscal_period == "FY"
    assert cit.item_or_metric == "revenues"
    assert cit.excerpt == "NVDA FY2026 revenues = 215,900,000,000 USD"
    assert cit.accession_number == "0001045810-26-000010"
    assert cit.source_type == "sql"


def test_citation_node_sanitizes_draft_answer() -> None:
    """Verify citation_node strips horizontal dividers and bracketed headings from final_answer."""
    state = {
        "query": "So sánh doanh thu NVIDIA và Microsoft",
        "intent": IntentType.FINANCIAL_METRIC.value,
        "draft_answer": (
            "### [Công bố Chính thức & Bằng chứng]\n"
            "Doanh thu NVDA đạt 215,9 tỷ USD [1].\n\n---\n\n"
            "### [Phân tích & Bình luận]\n"
            "Tăng trưởng vượt trội [1]."
        ),
        "evidence_items": [
            {
                "citation_id": 1,
                "source_type": EvidenceSourceType.SQL.value,
                "ticker": "NVDA",
                "form": "10-K",
                "fiscal_year": 2026,
                "fiscal_period": "FY",
                "item_or_metric": "revenues",
                "content_summary": "NVDA FY2026 revenues = 215.9B USD",
                "source_url": "https://www.sec.gov/edgar",
            }
        ],
        "routing_decision": {"tickers": ["NVDA"]},
    }

    result = citation_node(state)
    final_answer = result["final_answer"]
    assert "---" not in final_answer
    assert "### Công bố Chính thức & Bằng chứng" in final_answer
    assert "### Phân tích & Bình luận" in final_answer
    assert result["citations"][0]["excerpt"] == "NVDA FY2026 revenues = 215.9B USD"


def test_query_persists_activity_timeline_and_ui_serves_task_18_1_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify query metadata persists activity_timeline and UI exposes drawer, chips, and exports."""
    monkeypatch.setenv("ENABLE_S3_CHAT_HISTORY", "false")
    mgr = ChatHistoryManager(s3_storage=None)
    mgr._s3_enabled = False
    mgr._s3 = None

    mock_agent = MagicMock()
    mock_agent.query.return_value = AgentResponse(
        query="Compare AAPL and MSFT",
        intent=IntentType.FINANCIAL_METRIC,
        answer="### Official Disclosures\nAAPL revenue was $391B [1].",
        citations=[
            Citation(
                citation_id=1,
                formatted_reference="[1] AAPL Form 10-K (FY2024)",
                source_url="https://www.sec.gov/edgar",
                ticker="AAPL",
                form="10-K",
                fiscal_year=2024,
                fiscal_period="FY",
                item_or_metric="revenues",
                excerpt="Apple FY2024 revenues = $391.04B",
            )
        ],
        activity_timeline=[
            ActivityTimelineStep(
                step="Understanding question",
                status="completed",
                description="Planned SQL query",
                duration_ms=42.0,
            )
        ],
        execution_time_ms=155.4,
    )

    app.dependency_overrides[get_chat_history] = lambda: mgr
    app.dependency_overrides[get_research_agent] = lambda: mock_agent
    try:
        client = TestClient(app)
        resp = client.post("/api/query", json={"query": "Compare AAPL and MSFT", "session_id": "sess-18-1"})
        assert resp.status_code == 200

        session = mgr.get_session("sess-18-1")
        assert session is not None
        meta = session.messages[1].metadata or {}
        assert meta["intent"] == "FINANCIAL_METRIC"
        assert meta["execution_time_ms"] == 155.4
        assert len(meta["activity_timeline"]) == 1
        assert meta["activity_timeline"][0]["step"] == "Understanding question"

        html = client.get("/").text
        assert "citation-drawer" in html
        assert "quick-suggestions-row" in html
        assert "export-canvas-csv-btn" in html
        assert "export-canvas-json-btn" in html

        js = client.get("/static/app.js").text
        assert "openCitationDrawer" in js
        assert "exportTableData" in js

        css = client.get("/static/styles.css").text
        assert ".assistant-response-text hr" in css
    finally:
        app.dependency_overrides.pop(get_chat_history, None)
        app.dependency_overrides.pop(get_research_agent, None)
