"""Unit tests for FinancialSQLTool and DuckDBFinancialRepository.

Verifies:
- Pydantic schema validation contracts and input normalization
- SQL Injection attack payload defense and whitelisting
- SEC EDGAR canonical provenance URL generation
- DuckDB repository query routing (statements vs ratios)
- Financial semantics: NULL preservation (NULL != 0) and Q4 derivation flags
- Multi-company comparison and query latency measurement
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.agent.tools.financial_sql_tool import (
    FinancialMetricRecord,
    FinancialQueryInput,
    FinancialQueryResponse,
    FinancialSQLTool,
    generate_edgar_url,
)


@pytest.fixture(autouse=True)
def force_local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unit tests resolve local Parquet fixtures regardless of .env settings."""
    monkeypatch.setenv("DBT_TARGET", "local")
    monkeypatch.setenv("APP_ENV", "local")


class TestFinancialQueryInputValidation:
    """Test suite verifying input payload contract and validation."""

    def test_valid_input_normalization(self):
        """Verify lower-case tickers and mixed-case metrics are normalized."""
        query = FinancialQueryInput(
            ticker="aapl",
            metric="Revenues",
            fiscal_year=2023,
            fiscal_period="FY",
        )
        assert query.tickers == ["AAPL"]
        assert query.metrics == ["revenues"]
        assert query.fiscal_years == [2023]
        assert query.fiscal_periods == ["FY"]
        assert query.period_type == "annual"

    def test_batch_input_normalization(self):
        """Verify batch lists of tickers, metrics, and years are normalized."""
        query = FinancialQueryInput(
            tickers=["aapl", "msft"],
            metrics=["Revenues", "Operating_Income"],
            fiscal_years=[2022, 2023],
            fiscal_periods=["FY"],
        )
        assert query.tickers == ["AAPL", "MSFT"]
        assert query.metrics == ["revenues", "operating_income"]
        assert query.fiscal_years == [2022, 2023]
        assert query.fiscal_periods == ["FY"]

    def test_empty_lists_rejected(self):
        """Verify empty tickers or metrics are rejected."""
        with pytest.raises(ValidationError):
            FinancialQueryInput(tickers=[], metrics=["revenues"])
        with pytest.raises(ValidationError):
            FinancialQueryInput(tickers=["AAPL"], metrics=[])

    def test_unsupported_ticker_rejection(self):
        """Verify tickers outside Group A are rejected with ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            FinancialQueryInput(ticker="TSLA", metric="revenues")
        assert "Unsupported ticker" in str(exc_info.value)

    def test_unsupported_metric_rejection(self):
        """Verify metrics not in whitelist are rejected with ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            FinancialQueryInput(ticker="AAPL", metric="ebitda")
        assert "Unsupported metric" in str(exc_info.value)

    def test_period_consistency_validation(self):
        """Verify incompatible fiscal_period and period_type combinations raise errors."""
        with pytest.raises(ValidationError) as exc_info:
            FinancialQueryInput(
                ticker="AAPL",
                metric="revenues",
                fiscal_period="FY",
                period_type="quarterly",
            )
        assert "cannot have period_type 'quarterly'" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            FinancialQueryInput(
                ticker="AAPL",
                metric="revenues",
                fiscal_period="Q1",
                period_type="annual",
            )
        assert "cannot have period_type 'annual'" in str(exc_info.value)

    def test_auto_infer_period_type(self):
        """Verify period_type is automatically inferred from fiscal_period."""
        q_fy = FinancialQueryInput(ticker="AAPL", metric="revenues", fiscal_period="FY")
        assert q_fy.period_type == "annual"

        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            q_quarter = FinancialQueryInput(ticker="AAPL", metric="revenues", fiscal_period=quarter)
            assert q_quarter.period_type == "quarterly"


class TestSecuritySQLInjection:
    """Negative test suite testing SQL Injection attack defense."""

    @pytest.mark.parametrize(
        "attack_ticker",
        [
            "AAPL'; DROP TABLE statements; --",
            "AAPL' OR '1'='1",
            "AAPL' UNION SELECT * FROM users --",
            "AAPL/*comment*/",
            "admin'--",
        ],
    )
    def test_ticker_sql_injection_payloads_rejected(self, attack_ticker: str):
        """Verify SQL injection payloads in ticker are blocked by Pydantic."""
        with pytest.raises(ValidationError):
            FinancialQueryInput(ticker=attack_ticker, metric="revenues")

    @pytest.mark.parametrize(
        "attack_metric",
        [
            "revenues; DROP TABLE statements; --",
            "1; SELECT * FROM fct_financial_statements",
            "revenues' OR '1'='1",
            "revenues UNION ALL SELECT",
        ],
    )
    def test_metric_sql_injection_payloads_rejected(self, attack_metric: str):
        """Verify SQL injection payloads in metric name are blocked by Pydantic."""
        with pytest.raises(ValidationError):
            FinancialQueryInput(ticker="AAPL", metric=attack_metric)

    @pytest.mark.parametrize(
        "attack_period",
        [
            "Q1' OR '1'='1",
            "FY; DROP TABLE--",
            "Q1/*injection*/",
        ],
    )
    def test_period_sql_injection_payloads_rejected(self, attack_period: str):
        """Verify SQL injection payloads in fiscal_period are blocked by Pydantic."""
        with pytest.raises(ValidationError):
            FinancialQueryInput(ticker="AAPL", metric="revenues", fiscal_period=attack_period)


class TestProvenanceUrlGeneration:
    """Test suite verifying SEC EDGAR URL construction."""

    def test_canonical_edgar_url_formatting(self):
        """Verify leading zeros in CIK and hyphens in accession number are sanitized."""
        url = generate_edgar_url(
            cik="0000320193",
            accession_number="0000320193-23-000106",
        )
        expected = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106-index.htm"
        assert url == expected

    def test_none_accession_returns_none(self):
        """Verify missing accession number returns None."""
        assert generate_edgar_url(cik="0000320193", accession_number=None) is None

    def test_empty_cik_returns_none(self):
        """Verify missing CIK returns None."""
        assert generate_edgar_url(cik="", accession_number="0000320193-23-000106") is None


class TestDuckDBFinancialRepository:
    """Test suite verifying DuckDB parameterized querying and routing."""

    def test_query_statement_metric(self):
        """Verify statement metric queries fct_financial_statements."""
        repo = DuckDBFinancialRepository()
        query = FinancialQueryInput(
            ticker="AAPL",
            metric="revenues",
            fiscal_year=2023,
            fiscal_period="FY",
        )
        rows = repo.query_metric(query)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["metric_name"] == "revenues"
        assert rows[0]["fiscal_year"] == 2023
        assert rows[0]["fiscal_period"] == "FY"
        assert rows[0]["metric_value"] == Decimal("383285000000.00")
        assert rows[0]["source_accession_number"] == "0000320193-23-000106"

    def test_query_ratio_metric(self):
        """Verify ratio metric queries fct_financial_ratios."""
        repo = DuckDBFinancialRepository()
        query = FinancialQueryInput(
            ticker="MSFT",
            metric="net_margin",
            fiscal_year=2024,
            fiscal_period="FY",
        )
        rows = repo.query_metric(query)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "MSFT"
        assert rows[0]["metric_name"] == "net_margin"
        assert rows[0]["metric_value"] is not None


class TestFinancialSQLToolExecution:
    """Test suite verifying FinancialSQLTool execution and semantic invariants."""

    def test_tool_execute_returns_valid_envelope(self):
        """Verify execute returns structured FinancialQueryResponse."""
        tool = FinancialSQLTool()
        response = tool.get_metric(
            ticker="AAPL",
            metric="revenues",
            fiscal_year=2023,
            fiscal_period="FY",
        )
        assert isinstance(response, FinancialQueryResponse)
        assert response.record_count == 1
        assert response.execution_time_ms >= 0.0

        rec = response.data[0]
        assert isinstance(rec, FinancialMetricRecord)
        assert rec.ticker == "AAPL"
        assert rec.metric_name == "revenues"
        assert rec.metric_value == Decimal("383285000000.00")
        assert rec.provenance_url is not None
        assert "sec.gov" in rec.provenance_url

    def test_null_preservation_for_alphabet_gross_profit(self):
        """Verify Alphabet gross_profit is preserved as None (NULL != 0)."""
        tool = FinancialSQLTool()
        response = tool.get_metric(
            ticker="GOOGL",
            metric="gross_profit",
            fiscal_year=2023,
            fiscal_period="FY",
        )
        assert response.record_count == 1
        rec = response.data[0]
        assert rec.metric_value is None, "GOOGL Gross Profit must be strictly NULL, never 0"

    def test_derived_q4_metadata_preserved(self):
        """Verify derived Q4 record has is_derived=True and derivation_method='fy_minus_9m'."""
        tool = FinancialSQLTool()
        response = tool.get_metric(
            ticker="AAPL",
            metric="revenues",
            fiscal_year=2023,
            fiscal_period="Q4",
        )
        assert response.record_count == 1
        rec = response.data[0]
        assert rec.is_derived is True
        assert rec.derivation_method == "fy_minus_9m"

    def test_compare_companies(self):
        """Verify compare_companies compares metric across multiple companies."""
        tool = FinancialSQLTool()
        records = tool.compare_companies(
            metric="revenues",
            fiscal_year=2023,
            fiscal_period="FY",
            tickers=["AAPL", "MSFT", "GOOGL"],
        )
        assert len(records) == 3
        tickers_found = {r.ticker for r in records}
        assert tickers_found == {"AAPL", "MSFT", "GOOGL"}
        for r in records:
            assert r.metric_value is not None
            assert r.metric_value > 0
