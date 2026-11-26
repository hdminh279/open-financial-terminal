"""Unit tests for LangGraph Web Search Integration.

Tests routing to web_search_tool, route_next_step conditional branching,
web_node execution, AbstentionGate evaluation, and CitationEngine reference formatting.
"""

from unittest.mock import MagicMock

from backend.ai.agent.abstention import AbstentionGate
from backend.ai.agent.citation import CitationEngine
from backend.ai.agent.contracts import (
    EvidenceSourceType,
    IntentType,
    ToolExecutionStatus,
)
from backend.ai.agent.graph import web_node
from backend.ai.agent.router import IntentRouter, route_next_step
from backend.ai.agent.state import AgentState
from backend.ai.agent.tools.web_search_tool import (
    WebSearchResponse,
    WebSearchResultItem,
    WebSearchTool,
)


def test_router_selects_web_search_for_news() -> None:
    """Verify IntentRouter autonomously plans web_search_tool when news keywords are present."""
    router = IntentRouter()
    raw_llm_json = {
        "intent": "TEXTUAL_RESEARCH",
        "chart_preference": "none",
        "target_tools": ["web_search_tool"],
        "execution_strategy": "parallel",
        "tickers": ["FPT"],
        "fiscal_years": [],
        "fiscal_periods": [],
        "sql_params": None,
        "rag_params": None,
        "web_params": {
            "search_query": "Tin tức mới nhất về FPT hợp tác NVIDIA",
            "max_results": 5,
        },
        "reasoning": "Searching latest breaking news about FPT and NVIDIA partnership.",
    }

    decision = router._supervise_and_validate(raw_llm_json, original_query="Tin tức mới nhất về FPT hợp tác NVIDIA")
    assert "web_search_tool" in decision.target_tools
    assert decision.web_params is not None
    assert decision.web_params.search_query == "Tin tức mới nhất về FPT hợp tác NVIDIA"
    assert decision.query_plan is not None
    assert decision.query_plan.requires_web is True


def test_route_next_step_single_web() -> None:
    """Verify route_next_step directs cleanly to 'web_node' when only web search is required."""
    state: AgentState = {
        "intent": IntentType.TEXTUAL_RESEARCH.value,
        "query_plan": {
            "requires_sql": False,
            "requires_rag": False,
            "requires_web": True,
        },
        "routing_decision": {
            "target_tools": ["web_search_tool"],
        },
    }
    next_node = route_next_step(state)
    assert next_node == "web_node"


def test_route_next_step_fanout_sql_and_web() -> None:
    """Verify route_next_step performs parallel fan-out for hybrid SQL and Web query."""
    state: AgentState = {
        "intent": IntentType.MIXED_RESEARCH.value,
        "query_plan": {
            "requires_sql": True,
            "requires_rag": False,
            "requires_web": True,
        },
        "routing_decision": {
            "target_tools": ["financial_sql_tool", "web_search_tool"],
        },
    }
    next_nodes = route_next_step(state)
    assert isinstance(next_nodes, list)
    assert "sql_node" in next_nodes
    assert "web_node" in next_nodes


def test_web_node_execution() -> None:
    """Verify web_node calls WebSearchTool, creates EvidenceItems and registers timeline step."""
    mock_tool = MagicMock(spec=WebSearchTool)
    mock_tool.search.return_value = WebSearchResponse(
        query="Apple antitrust DOJ update",
        total_found=1,
        results=[
            WebSearchResultItem(
                title="DOJ Sues Apple Over Smartphone Monopoly",
                url="https://www.justice.gov/opa/pr/apple-suit",
                snippet="The Justice Department along with 16 states filed an antitrust lawsuit against Apple.",
                date="2024-03-21",
            )
        ],
    )

    state: AgentState = {
        "query": "What is the latest DOJ news about Apple?",
        "web_params": {
            "search_query": "Apple antitrust DOJ update",
            "max_results": 3,
        },
        "routing_decision": {
            "tickers": ["AAPL"],
        },
    }

    result = web_node(state, web_tool=mock_tool)

    assert result["web_data"]["total_found"] == 1
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["status"] == ToolExecutionStatus.SUCCESS.value
    assert len(result["evidence_items"]) == 1

    ev = result["evidence_items"][0]
    assert ev["source_type"] == EvidenceSourceType.WEB.value
    assert ev["ticker"] == "AAPL"
    assert ev["source_url"] == "https://www.justice.gov/opa/pr/apple-suit"
    assert "Web: DOJ Sues Apple Over Smartphone Monopoly" in ev["item_or_metric"]

    assert len(result["activity_timeline"]) == 1
    assert result["activity_timeline"][0]["step"] == "Searching live web"
    assert result["activity_timeline"][0]["status"] == "completed"


def test_abstention_gate_with_web_results() -> None:
    """Verify AbstentionGate allows progression to synthesis when web search yields evidence."""
    gate = AbstentionGate()
    state: AgentState = {
        "intent": IntentType.TEXTUAL_RESEARCH.value,
        "routing_decision": {
            "tickers": ["FPT"],
            "target_tools": ["web_search_tool"],
        },
        "tool_results": [
            {
                "tool_name": "web_search_tool",
                "status": ToolExecutionStatus.SUCCESS.value,
            }
        ],
        "evidence_items": [
            {
                "citation_id": 1,
                "source_type": EvidenceSourceType.WEB.value,
                "ticker": "FPT",
                "item_or_metric": "Web: FPT and NVIDIA",
                "content_summary": "Partnership announced.",
                "source_url": "https://example.com/fpt",
            }
        ],
    }

    info = gate.evaluate(state)
    assert info.is_abstaining is False
    assert info.is_partial is False


def test_citation_engine_formats_web_citation() -> None:
    """Verify CitationEngine formats web sources with [Live Web Source] markdown link."""
    evidence_dicts = [
        {
            "citation_id": 1,
            "source_type": EvidenceSourceType.WEB.value,
            "ticker": "FPT",
            "form": "WEB_NEWS",
            "item_or_metric": "Web: FPT AI Factory Announced",
            "source_url": "https://fpt.com/en/news/ai-factory",
            "content_summary": "Details on investment and GPU infrastructure.",
        }
    ]

    citations = CitationEngine.build_citations(evidence_dicts)
    assert len(citations) == 1
    ref = citations[0].formatted_reference
    assert "[1] Web: FPT AI Factory Announced — [Live Web Source](https://fpt.com/en/news/ai-factory)" in ref


def test_citation_engine_formats_vn_bctc_citation() -> None:
    """Verify CitationEngine formats VN30 financial statements with [Financial Statement Source]."""
    evidence_dicts = [
        {
            "citation_id": 1,
            "source_type": EvidenceSourceType.SQL.value,
            "ticker": "VNM",
            "form": "BCTC",
            "fiscal_year": 2023,
            "fiscal_period": "FY",
            "item_or_metric": "revenues",
            "source_url": "https://api-finfo.vndirect.com.vn/v4/financial_statements",
            "content_summary": "60374000000000 VND",
        }
    ]

    citations = CitationEngine.build_citations(evidence_dicts)
    assert len(citations) == 1
    ref = citations[0].formatted_reference
    assert "[1] VNM BCTC (FY 2023) revenues — [Financial Statement Source]" in ref


def test_canonicalize_timeline_preserves_web_search() -> None:
    """Verify canonicalize_timeline preserves 'Searching live web' without dropping it."""
    from backend.ai.agent.research_agent import canonicalize_timeline

    raw_events = [
        {"step": "Understanding question", "status": "completed", "duration_ms": 10.0},
        {"step": "Resolving metrics", "status": "completed", "duration_ms": 5.0},
        {"step": "Searching live web", "status": "completed", "duration_ms": 250.0},
        {"step": "Validating evidence", "status": "completed", "duration_ms": 2.0},
    ]

    ordered = canonicalize_timeline(raw_events, intent=IntentType.TEXTUAL_RESEARCH, has_charts=False)
    step_names = [s.step for s in ordered]

    assert "Searching live web" in step_names
    web_step = next(s for s in ordered if s.step == "Searching live web")
    assert web_step.status == "completed"
    assert web_step.duration_ms == 250.0
