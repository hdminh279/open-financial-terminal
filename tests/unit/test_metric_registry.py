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
