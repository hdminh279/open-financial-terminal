"""Unit tests for Refactored Financial Chart Engine.

Covers:
1. Router chart preference parsing & fallback.
2. Annual and quarterly planning.
3. Annual FY chart & chronological sorting.
4. Quarterly chart Q1-Q4 chronological sorting.
5. Missing values preserved as None (NULL != 0).
6. Multi-ticker unit consistency and Plotly figure traces.
7. Insufficient data (< 2 points) and chart_preference='none' guardrails.
8. Duration metrics YTD conversion (Q1 discrete, Q2/Q3 subtraction, Q4 derivation).
9. Point-in-time metrics (Assets/Cash) no subtraction.
10. Missing / incompatible source data handling.
11. Backward compatibility with ChartData and CitationNode.
"""

from src.ai.agent.chart_engine import (
    ChartEngine,
    TimeSeriesNormalizer,
)
from src.ai.agent.citation import citation_node
from src.ai.agent.contracts import ChartType
from src.ai.agent.router import IntentRouter
from src.ai.agent.state import AgentState


# ---------------------------------------------------------------------------
# 1. Router Chart Preference & Fallback Tests
# ---------------------------------------------------------------------------

def test_router_chart_preference_parsing() -> None:
    """Verify router correctly parses valid chart preferences and falls back on invalid values."""
    router = IntentRouter()

    # Valid line preference
    d_line = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2022, 2023, 2024],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "chart_preference": "line",
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Show AAPL revenue trend from 2022 to 2024 as a line chart",
    )
    assert d_line.chart_preference == "line"
    assert d_line.query_plan is not None
    assert d_line.query_plan.chart_preference == "line"

    # Valid bar preference
    d_bar = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL", "MSFT"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "chart_preference": "bar",
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Compare AAPL and MSFT revenue in 2023 as a bar chart",
    )
    assert d_bar.chart_preference == "bar"

    # Unsupported preference fallback to 'none'
    d_invalid = router._supervise_and_validate(
        {
            "intent": "FINANCIAL_METRIC",
            "tickers": ["AAPL"],
            "fiscal_years": [2023],
            "fiscal_periods": ["FY"],
            "metrics": ["revenues"],
            "chart_preference": "pie_chart_3d",
            "target_tools": ["financial_sql_tool"],
        },
        original_query="Show revenue in 3D pie chart",
    )
    assert d_invalid.chart_preference == "none"

    # Omitted preference default to 'none'
    d_omitted = router._supervise_and_validate(
        {
            "intent": "TEXTUAL_RESEARCH",
            "tickers": ["AAPL"],
            "fiscal_years": [2023],
            "target_tools": ["hybrid_retriever"],
        },
        original_query="What are Apple's main risk factors?",
    )
    assert d_omitted.chart_preference == "none"


# ---------------------------------------------------------------------------
# 2. Annual & Quarterly Planning
# ---------------------------------------------------------------------------

def test_annual_and_quarterly_planning() -> None:
    """Verify router plans annual vs quarterly periods without ambiguity."""
    normalizer = TimeSeriesNormalizer()

    # Annual records
    records_annual = [
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383285000000},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 391035000000},
    ]
    ts_annual = normalizer.normalize(records_annual, "revenues", "AAPL", target_periods=["FY"])
    assert ts_annual is not None
    assert ts_annual.x_labels == ["FY2023", "FY2024"]

    # Quarterly records
    records_quarterly = [
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "Q1", "metric_value": 119575000000},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "Q2", "metric_value": 90753000000},
    ]
    ts_quarterly = normalizer.normalize(records_quarterly, "revenues", "AAPL", target_periods=["Q1", "Q2"])
    assert ts_quarterly is not None
    assert ts_quarterly.x_labels == ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4"]


# ---------------------------------------------------------------------------
# 3. Annual FY Chart & Chronological Ordering
# ---------------------------------------------------------------------------

def test_annual_fy_chart_and_chronological_order() -> None:
    """Ensure annual records are strictly sorted by fiscal year regardless of input order."""
    engine = ChartEngine()

    shuffled_data = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 391.04, "unit": "USD"},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2021, "fiscal_period": "FY", "metric_value": 365.82, "unit": "USD"},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383.29, "unit": "USD"},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2022, "fiscal_period": "FY", "metric_value": 394.33, "unit": "USD"},
        ]
    }

    charts = engine.extract_financial_charts(
        sql_data=shuffled_data,
        chart_preference="line",
        ticker="AAPL",
    )

    assert len(charts) == 1
    chart = charts[0]
    assert chart.chart_type == ChartType.LINE
    assert chart.x_labels == ["FY2021", "FY2022", "FY2023", "FY2024"]

    # Check Plotly JSON
    plotly_json = chart.plotly_figure_json
    assert plotly_json is not None
    assert plotly_json["data"][0]["x"] == ["FY2021", "FY2022", "FY2023", "FY2024"]
    assert plotly_json["data"][0]["y"] == [365.82, 394.33, 383.29, 391.04]
    assert plotly_json["data"][0]["connectgaps"] is False


# ---------------------------------------------------------------------------
# 4. Quarterly Chronological Order
# ---------------------------------------------------------------------------

def test_quarterly_chart_chronological_order() -> None:
    """Ensure quarterly records are strictly ordered Q1->Q4 chronologically across years."""
    engine = ChartEngine()

    shuffled_quarters = {
        "data": [
            {"ticker": "MSFT", "metric_name": "net_income", "fiscal_year": 2023, "fiscal_period": "Q3", "metric_value": 18.3},
            {"ticker": "MSFT", "metric_name": "net_income", "fiscal_year": 2023, "fiscal_period": "Q1", "metric_value": 17.56},
            {"ticker": "MSFT", "metric_name": "net_income", "fiscal_year": 2023, "fiscal_period": "Q4", "metric_value": 20.08},
            {"ticker": "MSFT", "metric_name": "net_income", "fiscal_year": 2023, "fiscal_period": "Q2", "metric_value": 16.43},
        ]
    }

    charts = engine.extract_financial_charts(
        sql_data=shuffled_quarters,
        chart_preference="bar",
        ticker="MSFT",
        query_plan={"fiscal_periods": ["Q1", "Q2", "Q3", "Q4"]},
    )

    assert len(charts) == 1
    chart = charts[0]
    assert chart.chart_type == ChartType.BAR
    assert chart.x_labels == ["2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4"]
    assert chart.datasets[0]["values"] == [17.56, 16.43, 18.3, 20.08]


# ---------------------------------------------------------------------------
# 5. Missing Values Not Converted to Zero
# ---------------------------------------------------------------------------

def test_missing_values_preserved_not_zero() -> None:
    """Strictly enforce NULL != ZERO. Missing values must remain None/null in datasets and Plotly."""
    engine = ChartEngine()

    data_with_missing = {
        "data": [
            {"ticker": "GOOGL", "metric_name": "gross_profit", "fiscal_year": 2022, "fiscal_period": "FY", "metric_value": 150.0},
            {"ticker": "GOOGL", "metric_name": "gross_profit", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": None},
            {"ticker": "GOOGL", "metric_name": "gross_profit", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 180.0},
        ]
    }

    charts = engine.extract_financial_charts(
        sql_data=data_with_missing,
        chart_preference="line",
        ticker="GOOGL",
    )

    assert len(charts) == 1
    chart = charts[0]
    values = chart.datasets[0]["values"]
    assert values[1] is None, "Missing FY2023 metric_value must remain None, NEVER 0!"

    # Check Plotly trace
    plotly_json = chart.plotly_figure_json
    assert plotly_json is not None
    assert plotly_json["data"][0]["y"] == [150.0, None, 180.0]
    assert plotly_json["data"][0]["connectgaps"] is False

    # Check ASCII
    assert "[No data]" in chart.ascii_chart


# ---------------------------------------------------------------------------
# 6. Multi-Ticker and Unit Consistency
# ---------------------------------------------------------------------------

def test_multi_ticker_and_unit_consistency() -> None:
    """Ensure multi-ticker comparisons are scaled consistently and plotted together."""
    engine = ChartEngine()

    peer_data = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383285000000},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 391035000000},
            {"ticker": "MSFT", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 211915000000},
            {"ticker": "MSFT", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 245122000000},
        ]
    }

    charts = engine.extract_financial_charts(
        sql_data=peer_data,
        chart_preference="bar",
    )

    assert len(charts) == 1
    chart = charts[0]
    assert len(chart.datasets) == 2
    assert chart.datasets[0]["label"] == "AAPL Total Revenues"
    assert chart.datasets[1]["label"] == "MSFT Total Revenues"
    assert chart.datasets[0]["values"] == [383.29, 391.04]
    assert chart.datasets[1]["values"] == [211.91, 245.12]
    assert chart.unit == "USD (Billions)"
    assert chart.title == "AAPL vs MSFT Total Revenues (USD (Billions))"

    # Check Plotly trace labels
    traces = chart.plotly_figure_json["data"]
    assert len(traces) == 2
    assert traces[0]["name"] == "AAPL Total Revenues"
    assert traces[1]["name"] == "MSFT Total Revenues"


# ---------------------------------------------------------------------------
# 7. Insufficient Data and Preference None Guardrails
# ---------------------------------------------------------------------------

def test_insufficient_data_skips_rendering() -> None:
    """Ensure charts are NOT generated when preference is none or data points < 2."""
    engine = ChartEngine()

    single_point = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383.29},
        ]
    }

    # Points < 2
    assert engine.extract_financial_charts(single_point, chart_preference="line") == []

    # Preference is 'none'
    multi_point = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383.29},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 391.04},
        ]
    }
    assert engine.extract_financial_charts(multi_point, chart_preference="none") == []

    # SQL data is empty
    assert engine.extract_financial_charts(None, chart_preference="line") == []
    assert engine.extract_financial_charts({"data": []}, chart_preference="line") == []


# ---------------------------------------------------------------------------
# 8. Duration Metrics YTD Conversion & Q4 Derivation
# ---------------------------------------------------------------------------

def test_quarterly_duration_ytd_derivation() -> None:
    """Verify duration metrics YTD conversion:
    Q1 = YTD Q1 (100)
    Q2 = YTD Q2 - YTD Q1 (220 - 100 = 120)
    Q3 = YTD Q3 - YTD Q2 (350 - 220 = 130)
    Q4 = FY - YTD Q3 (500 - 350 = 150)
    """
    normalizer = TimeSeriesNormalizer()

    records = [
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q1", "metric_value": 100.0, "duration_scope": "discrete_quarter"},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q2", "metric_value": 220.0, "is_cumulative": True, "duration_scope": "cumulative_6m"},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q3", "metric_value": 350.0, "is_cumulative": True, "duration_scope": "cumulative_9m"},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 500.0, "duration_scope": "annual_12m"},
    ]

    ts = normalizer.normalize(records, "revenues", "AAPL", target_periods=["Q1", "Q2", "Q3", "Q4"])
    assert ts is not None

    pts = {p.fiscal_period: p for p in ts.points}
    assert pts["Q1"].value == 100.0
    assert pts["Q1"].is_derived is False

    assert pts["Q2"].value == 120.0
    assert pts["Q2"].is_derived is True

    assert pts["Q3"].value == 130.0
    assert pts["Q3"].is_derived is True

    assert pts["Q4"].value == 150.0
    assert pts["Q4"].is_derived is True

    # Mathematical identity check: sum(Q1..Q4) == FY
    total_quarterly = sum(p.value for p in ts.points if p.value is not None)
    assert total_quarterly == 500.0


# ---------------------------------------------------------------------------
# 9. Point-in-Time Metrics (Assets) No Subtraction
# ---------------------------------------------------------------------------

def test_point_in_time_assets_no_subtraction() -> None:
    """Verify point-in-time metrics (e.g. total_assets) NEVER undergo cumulative subtraction."""
    normalizer = TimeSeriesNormalizer()

    records = [
        {"ticker": "AAPL", "metric_name": "total_assets", "fiscal_year": 2023, "fiscal_period": "Q1", "metric_value": 350.0},
        {"ticker": "AAPL", "metric_name": "total_assets", "fiscal_year": 2023, "fiscal_period": "Q2", "metric_value": 360.0},
        {"ticker": "AAPL", "metric_name": "total_assets", "fiscal_year": 2023, "fiscal_period": "Q3", "metric_value": 370.0},
        {"ticker": "AAPL", "metric_name": "total_assets", "fiscal_year": 2023, "fiscal_period": "Q4", "metric_value": 380.0},
        {"ticker": "AAPL", "metric_name": "total_assets", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 380.0},
    ]

    ts = normalizer.normalize(records, "total_assets", "AAPL", target_periods=["Q1", "Q2", "Q3", "Q4"])
    assert ts is not None
    assert ts.is_point_in_time is True

    pts = {p.fiscal_period: p for p in ts.points}
    # Values must remain raw snapshot observations (NO SUBTRACTION!)
    assert pts["Q1"].value == 350.0
    assert pts["Q2"].value == 360.0  # NOT 360 - 350 = 10
    assert pts["Q3"].value == 370.0  # NOT 370 - 360 = 10
    assert pts["Q4"].value == 380.0  # NOT 380 - 370 = 10


# ---------------------------------------------------------------------------
# 10. Missing or Incompatible Source Data Handling
# ---------------------------------------------------------------------------

def test_missing_or_incompatible_source_data() -> None:
    """Verify that if YTD data is missing or incompatible, values remain None without crashing."""
    normalizer = TimeSeriesNormalizer()

    # Missing YTD Q2 -> Q3 cannot be derived from YTD
    records_missing_q2 = [
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q1", "metric_value": 100.0},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q3", "metric_value": 350.0, "is_cumulative": True},
    ]
    ts = normalizer.normalize(records_missing_q2, "revenues", "AAPL", target_periods=["Q1", "Q2", "Q3", "Q4"])
    # Q1 exists (100.0), Q2 is None, Q3 cannot be derived (None) -> only 1 valid point -> returns None
    assert ts is None

    # Incompatible units (USD vs EUR)
    records_mismatched_units = [
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q1", "metric_value": 100.0, "unit": "usd"},
        {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "Q2", "metric_value": 220.0, "unit": "eur", "is_cumulative": True},
    ]
    derived_val, _is_derived, _ = normalizer._derive_quarterly_duration(
        fy=2023,
        quarter="Q2",
        ytd_data={(2023, "Q1"): records_mismatched_units[0], (2023, "Q2"): records_mismatched_units[1]},
        fy_data={},
        discrete_data={},
    )
    assert derived_val is None, "Incompatible units must abort derivation and remain None"


# ---------------------------------------------------------------------------
# 11. Regression with Existing Contracts & Citation Node
# ---------------------------------------------------------------------------

def test_citation_node_chart_preference_integration() -> None:
    """Verify citation_node extracts charts honoring query_plan.chart_preference."""
    sql_data = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2023, "fiscal_period": "FY", "metric_value": 383285000000},
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": 2024, "fiscal_period": "FY", "metric_value": 391035000000},
        ]
    }

    state: AgentState = {
        "query": "Show AAPL revenue trend from 2023 to 2024",
        "intent": "FINANCIAL_METRIC",
        "draft_answer": "Apple revenue was $383.29B in 2023 and $391.04B in 2024.",
        "sql_data": sql_data,
        "routing_decision": {
            "tickers": ["AAPL"],
            "chart_preference": "line",
        },
        "query_plan": {
            "chart_preference": "line",
            "fiscal_periods": ["FY"],
        },
        "evidence_items": [],
    }

    output = citation_node(state)
    assert len(output["charts"]) == 1
    chart_dict = output["charts"][0]
    assert chart_dict["chart_type"] == "line"
    assert chart_dict["plotly_figure_json"] is not None
    assert chart_dict["datasets"][0]["values"] == [383.29, 391.04]


def test_chart_engine_year_range_filtering() -> None:
    """Ensure ChartEngine filters out records outside the target fiscal years."""
    engine = ChartEngine()

    # Data containing all years from 2009 to 2026 (like the user experienced)
    all_years_data = {
        "data": [
            {"ticker": "AAPL", "metric_name": "revenues", "fiscal_year": y, "fiscal_period": "FY", "metric_value": 100.0 + y}
            for y in range(2009, 2027)
        ] + [
            {"ticker": "MSFT", "metric_name": "revenues", "fiscal_year": y, "fiscal_period": "FY", "metric_value": 50.0 + y}
            for y in range(2009, 2027)
        ]
    }

    # User requested 2020 to 2024
    charts = engine.extract_financial_charts(
        sql_data=all_years_data,
        chart_preference="bar",
        query_plan={"fiscal_years": [2020, 2024], "fiscal_periods": ["FY"]},
    )

    assert len(charts) == 1
    chart = charts[0]
    # MUST only have FY2020 to FY2024 (5 bars, NOT 18 bars from 2009 to 2026!)
    assert chart.x_labels == ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]
    assert len(chart.datasets[0]["values"]) == 5
    assert len(chart.datasets[1]["values"]) == 5

