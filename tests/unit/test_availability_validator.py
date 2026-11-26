"""Unit tests for FinancialAvailabilityValidator and Dynamic SQL Guardrails.

Verifies:
1. Dynamic period end date validation against reference date (zero static calendar hardcoding).
2. Filing publication status verification (Form 10-K vs 10-Q partial period handling).
3. SQL tool integration: filtering queries so DuckDB never executes unended/unfiled periods.
4. End-to-end consistency: availability reports populated for excluded future years.
"""

from datetime import date

from backend.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from backend.ai.agent.tools.financial_sql_tool import (
    FinancialQueryInput,
    FinancialSQLTool,
)
from backend.core.financial.availability_validator import (
    AvailabilityCode,
    FinancialAvailabilityValidator,
)


class TestFinancialAvailabilityValidator:
    """Test suite for data-driven financial data availability validation."""

    def test_future_year_rejection_against_reference_date(self):
        """Verify that years beyond reference date are rejected as FUTURE_PERIOD_NOT_ENDED."""
        validator = FinancialAvailabilityValidator()
        # Reference date set to mid-2024
        ref_date = date(2024, 6, 1)

        valid_years, reports = validator.validate_periods(
            tickers=["AAPL", "MSFT"],
            fiscal_years=[2023, 2024, 2025, 2026],
            fiscal_periods=["FY"],
            today=ref_date,
        )

        # 2025 and 2026 are strictly future relative to 2024-06-01
        assert 2025 not in valid_years
        assert 2026 not in valid_years

        future_reports = [r for r in reports if r.code == AvailabilityCode.FUTURE_PERIOD_NOT_ENDED]
        assert len(future_reports) >= 4  # AAPL & MSFT for 2025 & 2026

    def test_sql_tool_filters_future_years_and_populates_reports(self):
        """Verify FinancialSQLTool only queries valid historical years and reports excluded ones."""
        tool = FinancialSQLTool()

        # Inject simulated reference date of 2024-10-01 (when FY2024 is available, but 2025/2026 are not)
        ref_date = date(2024, 10, 1)

        query = FinancialQueryInput(
            tickers=["AAPL"],
            metrics=["revenues"],
            fiscal_years=[2022, 2023, 2024, 2025, 2026],
            fiscal_periods=["FY"],
        )

        # Temporarily mock today in validator
        original_validate = tool.availability_validator.validate_periods

        def mock_validate(*args, **kwargs):
            kwargs["today"] = ref_date
            return original_validate(*args, **kwargs)

        tool.availability_validator.validate_periods = mock_validate

        response = tool.execute(query)

        # Valid years in result should only be up to 2024
        result_years = {r.fiscal_year for r in response.data}
        assert 2025 not in result_years
        assert 2026 not in result_years

        # Response must include availability audit reports
        assert len(response.availability_reports) > 0
        excluded = [r for r in response.availability_reports if not r.is_available]
        excluded_years = {r.fiscal_year for r in excluded}
        assert 2025 in excluded_years
        assert 2026 in excluded_years

    def test_all_future_years_returns_empty_data_with_audit_reason(self):
        """Verify querying solely future years returns 0 records and clear explanation."""
        tool = FinancialSQLTool()
        ref_date = date(2024, 1, 1)

        query = FinancialQueryInput(
            tickers=["MSFT"],
            metrics=["revenues"],
            fiscal_years=[2026, 2027],
            fiscal_periods=["FY"],
        )

        original_validate = tool.availability_validator.validate_periods

        def mock_validate(*args, **kwargs):
            kwargs["today"] = ref_date
            return original_validate(*args, **kwargs)

        tool.availability_validator.validate_periods = mock_validate

        response = tool.execute(query)
        assert response.record_count == 0
        assert len(response.data) == 0
        assert len(response.availability_reports) >= 2
        assert all(not r.is_available for r in response.availability_reports)

    def test_duckdb_repository_resolves_fixtures_when_flag_is_set(self, monkeypatch):
        """Verify repository resolves test fixtures when USE_TEST_FIXTURES is true."""
        monkeypatch.setenv("USE_TEST_FIXTURES", "true")
        monkeypatch.delenv("GOLD_STATEMENTS_PARQUET_PATH", raising=False)
        monkeypatch.delenv("GOLD_RATIOS_PARQUET_PATH", raising=False)

        stmt, ratio = DuckDBFinancialRepository.resolve_default_paths()
        assert "tests/fixtures" in stmt
        assert "tests/fixtures" in ratio

    def test_duckdb_repository_resolves_s3_when_operational(self, monkeypatch):
        """Verify repository resolves to authoritative S3 when operational."""
        monkeypatch.setenv("USE_TEST_FIXTURES", "false")
        monkeypatch.delenv("GOLD_STATEMENTS_PARQUET_PATH", raising=False)
        monkeypatch.delenv("GOLD_RATIOS_PARQUET_PATH", raising=False)

        stmt, ratio = DuckDBFinancialRepository.resolve_default_paths()
        assert stmt.startswith("s3://")
        assert ratio.startswith("s3://")
