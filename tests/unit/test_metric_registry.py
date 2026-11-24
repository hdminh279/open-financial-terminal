"""Unit tests for MetricRegistry and metadata resolution."""

import pytest

from src.core.financial.metric_registry import MetricRegistry


@pytest.fixture(scope="module")
def registry() -> MetricRegistry:
    """Fixture providing initialized MetricRegistry instance."""
    return MetricRegistry()


def test_registry_loads_all_taxonomy_metrics(registry: MetricRegistry):
    """Verify registry loads all 49 defined taxonomy metrics."""
    metrics = registry.list_metrics()
    assert len(metrics) == 49


def test_canonical_metric_retrieval(registry: MetricRegistry):
    """Verify canonical metric lookup returns correct attributes."""
    rev = registry.get("revenues")
    assert rev is not None
    assert rev.key == "revenues"
    assert rev.display_name == "Total Revenues"
    assert rev.classification == "gaap"
    assert rev.storage_layer == "sql_gold"
    assert rev.default_unit == "USD"
    assert not rev.is_dimensional


def test_bilingual_synonym_resolution(registry: MetricRegistry):
    """Verify English and Vietnamese synonyms resolve to canonical keys."""
    test_cases = [
        ("revenue", "revenues"),
        ("tổng doanh thu", "revenues"),
        ("doanh số", "revenues"),
        ("operating margin", "operating_margin"),
        ("biên lợi nhuận hoạt động", "operating_margin"),
        ("r&d", "research_and_development"),
        ("chi phí r&d", "research_and_development"),
        ("chi phí nghiên cứu", "research_and_development"),
        ("ebitda", "adjusted_ebitda"),
        ("ebitda điều chỉnh", "adjusted_ebitda"),
        ("iphone", "product_revenue_iphone"),
        ("doanh thu iphone", "product_revenue_iphone"),
        ("azure", "product_revenue_azure"),
        ("đám mây azure", "product_revenue_azure"),
    ]
    for term, expected_key in test_cases:
        resolved = registry.resolve(term)
        assert resolved is not None, f"Failed resolving '{term}'"
        assert resolved.key == expected_key, f"Resolved '{term}' to '{resolved.key}', expected '{expected_key}'"


def test_storage_layer_partitioning(registry: MetricRegistry):
    """Verify metrics partition accurately into SQL Gold vs RAG Textual layers."""
    sql_metrics = registry.list_metrics(storage_layer="sql_gold")
    rag_metrics = registry.list_metrics(storage_layer="rag_textual")

    assert len(sql_metrics) == 26
    assert len(rag_metrics) == 23
    assert len(sql_metrics) + len(rag_metrics) == 49

    assert registry.is_sql_metric("revenues") is True
    assert registry.is_sql_metric("adjusted_ebitda") is False
    assert registry.is_rag_metric("adjusted_ebitda") is True
    assert registry.is_rag_metric("product_revenue_iphone") is True


def test_unit_and_classification_resolution(registry: MetricRegistry):
    """Verify default measurement units and accounting classifications."""
    assert registry.get_default_unit("operating_margin") == "%"
    assert registry.get_default_unit("revenues") == "USD"
    assert registry.get_default_unit("current_ratio") == "ratio"
    assert registry.get_default_unit("eps_diluted") == "USD/shares"


def test_gold_table_ssot_matches_gold_parquet_and_dbt_marts(registry: MetricRegistry):
    """Verify STATEMENT_METRICS, RATIO_METRICS, and POINT_IN_TIME_METRICS align with Gold marts and Parquet schemas."""
    from pathlib import Path

    import duckdb

    from src.ai.agent.chart_engine import POINT_IN_TIME_METRICS
    from src.ai.agent.tools.financial_sql_tool import RATIO_METRICS, STATEMENT_METRICS

    assert set(registry.list_metrics(gold_table="statements")) == STATEMENT_METRICS
    assert set(registry.list_metrics(gold_table="ratios")) == RATIO_METRICS
    assert set(registry.list_metrics(period_type="instant")) == POINT_IN_TIME_METRICS

    # Unmaterialized metrics must NOT be mapped to a Gold SQL table
    for unmaterialized in (
        "research_and_development",
        "selling_general_and_admin",
        "total_operating_expenses",
        "capital_expenditures",
        "free_cash_flow",
    ):
        metric_def = registry.get(unmaterialized)
        assert metric_def is not None
        assert metric_def.gold_table is None
        assert unmaterialized not in STATEMENT_METRICS
        assert unmaterialized not in RATIO_METRICS

    # Every Gold table metric must exist in the authoritative dbt Gold mart SQL models
    stmt_sql = Path("transform/models/marts/fct_financial_statements.sql").read_text(encoding="utf-8")
    ratio_sql = Path("transform/models/marts/fct_financial_ratios.sql").read_text(encoding="utf-8")
    for metric in STATEMENT_METRICS:
        assert metric in stmt_sql, f"Statement metric '{metric}' missing from fct_financial_statements.sql"
    for metric in RATIO_METRICS:
        assert metric in ratio_sql, f"Ratio metric '{metric}' missing from fct_financial_ratios.sql"

    # Verify against committed Gold Parquet fixtures via DuckDB DESCRIBE
    conn = duckdb.connect(":memory:")
    stmt_meta = {
        "statement_key",
        "cik",
        "ticker",
        "market",
        "currency",
        "fiscal_year",
        "fiscal_period",
        "period_type",
        "start_date",
        "end_date",
        "is_derived",
        "derivation_method",
        "source_accession_number",
        "source_filed_date",
    }
    ratio_meta = {
        "ratio_key",
        "cik",
        "ticker",
        "market",
        "currency",
        "fiscal_year",
        "fiscal_period",
        "period_type",
        "start_date",
        "end_date",
        "source_accession_number",
        "source_filed_date",
    }
    fixture_stmt_cols = {
        row[0]
        for row in conn.execute(
            "DESCRIBE SELECT * FROM read_parquet('tests/fixtures/gold/fct_financial_statements.parquet')"
        ).fetchall()
    } - stmt_meta
    fixture_ratio_cols = {
        row[0]
        for row in conn.execute(
            "DESCRIBE SELECT * FROM read_parquet('tests/fixtures/gold/fct_financial_ratios.parquet')"
        ).fetchall()
    } - ratio_meta

    assert fixture_stmt_cols.issubset(STATEMENT_METRICS)
    assert fixture_ratio_cols == RATIO_METRICS

    # If full local exported Gold marts exist, verify every registry Gold column exists in Parquet
    local_stmt = Path("data/export_marts/fct_financial_statements.parquet")
    local_ratio = Path("data/export_marts/fct_financial_ratios.parquet")
    if local_stmt.exists():
        actual_stmt_cols = {
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{local_stmt}')").fetchall()
        }
        assert STATEMENT_METRICS.issubset(actual_stmt_cols)
    if local_ratio.exists():
        actual_ratio_cols = {
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{local_ratio}')").fetchall()
        }
        assert RATIO_METRICS.issubset(actual_ratio_cols)
    conn.close()
