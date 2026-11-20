"""Unit tests for FinancialQueryPlan and Router Orchestration.

Tests cover:
1. ComparisonType classification across diverse analytical query structures.
2. MetricRegistry resolution, SQL vs RAG partitioning, and auto-promotion to MIXED_RESEARCH.
3. LangGraph route_next_step conditional edge routing and parallel fan-out.
"""

from src.ai.agent.contracts import (
    ComparisonType,
    ExecutionStrategy,
    FinancialQueryPlan,
    IntentType,
)
from src.ai.agent.router import IntentRouter, route_next_step
from src.ai.agent.state import AgentState
from src.core.financial.metric_registry import MetricRegistry


def test_comparison_type_classification() -> None:
    """Verify semantic ComparisonType detection across query patterns."""
    router = IntentRouter()

    # 1. Single entity, single metric, single year -> NONE
    d1 = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="What was Apple revenue in 2023?",
    )
    assert d1.query_plan is not None
    assert d1.query_plan.comparison_type == ComparisonType.NONE

    # 2. Multi-company -> CROSS_COMPANY
    d2 = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL", "MSFT"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Compare Apple and Microsoft revenue in 2023",
    )
    assert d2.query_plan is not None
    assert d2.query_plan.comparison_type == ComparisonType.CROSS_COMPANY

    # 3. Multi-year / Trend -> TIME_SERIES
    d3 = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2022, 2023, 2024],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Show Apple revenue trend from 2022 to 2024",
    )
    assert d3.query_plan is not None
    assert d3.query_plan.comparison_type == ComparisonType.TIME_SERIES

    # 4. Multi-metric -> MULTI_METRIC
    d4 = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues", "net_income", "operating_margin"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Apple revenues, net income, and operating margin for 2023",
    )
    assert d4.query_plan is not None
    assert d4.query_plan.comparison_type == ComparisonType.MULTI_METRIC

    # 5. Multi-company + Multi-year / Multi-metric -> COMPREHENSIVE
    d5 = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL", "MSFT"],
            "fiscal_years": [2022, 2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues", "net_income"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Compare revenues and net income of AAPL and MSFT for 2022 and 2023",
    )
    assert d5.query_plan is not None
    assert d5.query_plan.comparison_type == ComparisonType.COMPREHENSIVE


def test_metric_registry_partitioning_and_promotion() -> None:
    """Verify metric resolution partitions SQL vs RAG and promotes to MIXED_RESEARCH."""
    registry = MetricRegistry()
    router = IntentRouter(metric_registry=registry)

    # Mixed query: 'revenues' (sql_gold) and 'doanh thu iphone' (rag_textual)
    decision = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues", "doanh thu iphone"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="So sánh tổng doanh thu và doanh thu iPhone của Apple năm 2023",
    )

    assert decision.intent == IntentType.MIXED_RESEARCH
    assert "financial_sql_tool" in decision.target_tools
    assert "hybrid_retriever" in decision.target_tools
    assert decision.execution_strategy == ExecutionStrategy.PARALLEL

    plan = decision.query_plan
    assert plan is not None
    assert plan.requires_sql is True
    assert plan.requires_rag is True
    assert "revenues" in plan.sql_metrics
    assert "product_revenue_iphone" in plan.rag_metrics


def test_route_next_step_conditional_edges() -> None:
    """Verify LangGraph route_next_step respects query_plan flags."""
    # 1. Mixed Fan-Out
    mixed_state: AgentState = {
        "intent": IntentType.MIXED_RESEARCH.value,
        "query_plan": {
            "requires_sql": True,
            "requires_rag": True,
        },
    }
    assert route_next_step(mixed_state) == ["sql_node", "rag_node"]

    # 2. SQL only
    sql_state: AgentState = {
        "intent": IntentType.FINANCIAL_METRIC.value,
        "query_plan": {
            "requires_sql": True,
            "requires_rag": False,
        },
    }
    assert route_next_step(sql_state) == "sql_node"

    # 3. RAG only
    rag_state: AgentState = {
        "intent": IntentType.TEXTUAL_RESEARCH.value,
        "query_plan": {
            "requires_sql": False,
            "requires_rag": True,
        },
    }
    assert route_next_step(rag_state) == "rag_node"

    # 4. Out of scope -> abstention
    oos_state: AgentState = {
        "intent": IntentType.OUT_OF_SCOPE.value,
        "query_plan": {
            "requires_sql": False,
            "requires_rag": False,
        },
    }
    assert route_next_step(oos_state) == "abstention_node"


def test_historical_year_not_blocked() -> None:
    """Verify historical year (e.g. 2015) is accepted into query plan without being blocked."""
    router = IntentRouter()
    decision = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2015],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "target_tools": ["financial_sql_tool"],
        },
        original_query="What was Apple revenue in 2015?",
    )
    assert decision.intent == IntentType.FINANCIAL_METRIC
    assert decision.query_plan is not None
    assert decision.query_plan.fiscal_years == [2015]
    assert decision.query_plan.requires_sql is True
    assert decision.sql_params is not None
    assert decision.sql_params.fiscal_years == [2015]

