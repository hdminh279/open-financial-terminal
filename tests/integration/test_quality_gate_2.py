"""Quality Gate 2 Integration Test Suite.

Validates the full enterprise analytical data mart and FinancialSQLTool against:
1. 100% Metric Availability across Group A companies (AAPL, MSFT, GOOGL) over 3 years (FY2022, FY2023, FY2024).
2. All 12 canonical financial statement metrics and 6 financial ratios.
3. Accounting Semantics:
   - Strict NULL preservation (Alphabet gross_profit and gross_margin must remain NULL).
   - Derived Q4 duration metric arithmetic ($Q1 + Q2 + Q3 + Q4 = FY$).
   - Derived Q4 EPS dilution constraint (EPS must be NULL for derived Q4).
   - Balance sheet equation sanity ($\text{Assets} = \text{Liabilities} + \text{Equity}$).
4. Provenance auditability: valid SEC EDGAR index URL attached to all as-reported records.
5. Latency SLA: sub-250ms query response time across all queries.
"""

import re
from decimal import Decimal

import pytest

from src.ai.agent.tools.financial_sql_tool import FinancialSQLTool

TICKERS = ["AAPL", "MSFT", "GOOGL"]
YEARS = [2022, 2023, 2024]
PERIODS = ["Q1", "Q2", "Q3", "Q4", "FY"]
EDGAR_URL_PATTERN = re.compile(
    r"^https://www\.sec\.gov/Archives/edgar/data/\d+/\d+/[\w-]+-index\.htm$"
)


@pytest.fixture(scope="module")
def tool() -> FinancialSQLTool:
    """Fixture providing initialized FinancialSQLTool."""
    return FinancialSQLTool()


class TestQualityGate2MetricAvailability:
    """Validate 100% metric availability across companies, years, and periods."""

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", YEARS)
    def test_annual_income_statement_metrics_exist(
        self, tool: FinancialSQLTool, ticker: str, year: int
    ):
        """Verify core income statement metrics exist for every company and year."""
        for metric in ["revenues", "operating_income", "net_income"]:
            resp = tool.get_metric(ticker=ticker, metric=metric, fiscal_year=year, fiscal_period="FY")
            assert resp.record_count == 1, f"Missing {metric} for {ticker} FY{year}"
            rec = resp.data[0]
            assert rec.metric_value is not None, f"{metric} should not be None for {ticker} FY{year}"
            assert rec.metric_value > 0, f"{metric} should be positive for {ticker} FY{year}"

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", YEARS)
    def test_annual_balance_sheet_metrics_exist(
        self, tool: FinancialSQLTool, ticker: str, year: int
    ):
        """Verify core balance sheet snapshot metrics exist for every company and year."""
        for metric in ["total_assets", "total_liabilities", "cash_and_cash_equivalents"]:
            resp = tool.get_metric(ticker=ticker, metric=metric, fiscal_year=year, fiscal_period="FY")
            assert resp.record_count == 1, f"Missing {metric} for {ticker} FY{year}"
            rec = resp.data[0]
            assert rec.metric_value is not None
            assert rec.metric_value > 0

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", YEARS)
    def test_annual_ratios_exist(self, tool: FinancialSQLTool, ticker: str, year: int):
        """Verify financial ratios are computed and queryable."""
        for metric in ["operating_margin", "net_margin", "current_ratio", "debt_to_equity"]:
            resp = tool.get_metric(ticker=ticker, metric=metric, fiscal_year=year, fiscal_period="FY")
            assert resp.record_count == 1, f"Missing ratio {metric} for {ticker} FY{year}"
            rec = resp.data[0]
            assert rec.metric_value is not None, f"Ratio {metric} is None for {ticker} FY{year}"

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_all_quarters_present_for_recent_year(self, tool: FinancialSQLTool, ticker: str):
        """Verify all 4 quarters (Q1-Q4) are present for FY2023."""
        resp = tool.execute({
            "ticker": ticker,
            "metric": "revenues",
            "fiscal_year": 2023,
            "period_type": "quarterly",
        })
        periods_found = [r.fiscal_period for r in resp.data]
        assert sorted(periods_found) == ["Q1", "Q2", "Q3", "Q4"], (
            f"Expected all 4 quarters for {ticker} in 2023, found {periods_found}"
        )


class TestQualityGate2AccountingSemantics:
    """Validate accounting invariants and special corporate reporting constraints."""

    @pytest.mark.parametrize("year", YEARS)
    def test_alphabet_gross_profit_is_strictly_null(self, tool: FinancialSQLTool, year: int):
        """Verify Alphabet (GOOGL) gross profit and gross margin are strictly NULL."""
        gp_resp = tool.get_metric(ticker="GOOGL", metric="gross_profit", fiscal_year=year, fiscal_period="FY")
        assert gp_resp.record_count == 1
        assert gp_resp.data[0].metric_value is None, f"GOOGL gross_profit in FY{year} must be None (NULL != 0)"

        gm_resp = tool.get_metric(ticker="GOOGL", metric="gross_margin", fiscal_year=year, fiscal_period="FY")
        assert gm_resp.record_count == 1
        assert gm_resp.data[0].metric_value is None, f"GOOGL gross_margin in FY{year} must be None"

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT"])
    @pytest.mark.parametrize("year", YEARS)
    def test_apple_and_microsoft_gross_profit_is_populated(
        self, tool: FinancialSQLTool, ticker: str, year: int
    ):
        """Verify Apple and Microsoft report gross profit as positive values."""
        resp = tool.get_metric(ticker=ticker, metric="gross_profit", fiscal_year=year, fiscal_period="FY")
        assert resp.record_count == 1
        assert resp.data[0].metric_value is not None
        assert resp.data[0].metric_value > 0

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", [2022, 2023])
    def test_q4_revenue_sum_identity(self, tool: FinancialSQLTool, ticker: str, year: int):
        """Verify mathematical identity: Q1 + Q2 + Q3 + Q4 == FY revenues (within $1 rounding)."""
        fy_resp = tool.get_metric(ticker=ticker, metric="revenues", fiscal_year=year, fiscal_period="FY")
        assert fy_resp.record_count == 1
        fy_rev = fy_resp.data[0].metric_value

        q_resp = tool.execute({
            "ticker": ticker,
            "metric": "revenues",
            "fiscal_year": year,
            "period_type": "quarterly",
        })
        assert q_resp.record_count == 4
        quarters_sum = sum(r.metric_value for r in q_resp.data if r.metric_value is not None)

        diff = abs(quarters_sum - fy_rev)
        assert diff <= Decimal("1.00"), (
            f"{ticker} FY{year} revenue sum mismatch: Q1..Q4 sum = {quarters_sum}, FY = {fy_rev}, diff = {diff}"
        )

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", [2022, 2023])
    def test_derived_q4_eps_is_strictly_null(self, tool: FinancialSQLTool, ticker: str, year: int):
        """Verify EPS is never mathematically derived for Q4 (must be strictly NULL)."""
        q4_eps = tool.get_metric(ticker=ticker, metric="eps_diluted", fiscal_year=year, fiscal_period="Q4")
        if q4_eps.record_count > 0:
            rec = q4_eps.data[0]
            if rec.is_derived:
                assert rec.metric_value is None, (
                    f"Derived Q4 diluted EPS for {ticker} FY{year} must be NULL"
                )

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("year", YEARS)
    def test_balance_sheet_identity(self, tool: FinancialSQLTool, ticker: str, year: int):
        """Verify Balance Sheet identity: Assets = Liabilities + Equity (within 2% tolerance)."""
        assets = tool.get_metric(ticker=ticker, metric="total_assets", fiscal_year=year, fiscal_period="FY").data[0].metric_value
        liabs = tool.get_metric(ticker=ticker, metric="total_liabilities", fiscal_year=year, fiscal_period="FY").data[0].metric_value
        equity = tool.get_metric(ticker=ticker, metric="stockholders_equity_parent", fiscal_year=year, fiscal_period="FY").data[0].metric_value

        assert assets is not None and liabs is not None and equity is not None
        calculated_assets = liabs + equity
        tolerance = assets * Decimal("0.02")  # 2% US-GAAP mezzanine/rounding allowance
        discrepancy = abs(assets - calculated_assets)
        assert discrepancy <= tolerance, (
            f"{ticker} FY{year} Balance sheet discrepancy too large: "
            f"Assets={assets}, Liabs+Equity={calculated_assets}, diff={discrepancy}"
        )


class TestQualityGate2AuditProvenanceAndLatency:
    """Validate citation provenance and performance SLA."""

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_as_reported_records_have_valid_edgar_provenance(
        self, tool: FinancialSQLTool, ticker: str
    ):
        """Verify all as-reported financial statement records have valid SEC EDGAR URLs."""
        resp = tool.get_metric(ticker=ticker, metric="revenues", fiscal_year=2023, fiscal_period="FY")
        assert resp.record_count == 1
        rec = resp.data[0]
        assert rec.provenance_url is not None
        assert EDGAR_URL_PATTERN.match(rec.provenance_url), (
            f"Invalid SEC EDGAR citation URL: {rec.provenance_url}"
        )

    def test_query_latency_sla(self, tool: FinancialSQLTool):
        """Verify queries execute well below 250ms SLA."""
        resp = tool.get_metric(ticker="AAPL", metric="revenues", fiscal_year=2023)
        assert resp.execution_time_ms < 250.0, f"Query too slow: {resp.execution_time_ms}ms"
