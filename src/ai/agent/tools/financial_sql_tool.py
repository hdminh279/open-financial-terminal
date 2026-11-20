"""Financial SQL Tool contracts and schema definitions."""

import time
from typing import TYPE_CHECKING, Any, Literal, Optional
if TYPE_CHECKING:
    from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository

from datetime import date
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.financial.company_registry import get_supported_tickers

STATEMENT_METRICS = {
    "revenues",
    "operating_income",
    "net_income",
    "gross_profit",
    "eps_diluted",
    "cash_and_cash_equivalents",
    "current_assets",
    "current_liabilities",
    "total_assets",
    "total_liabilities",
    "stockholders_equity_parent",
    "stockholders_equity_total",
}

RATIO_METRICS = {
    "operating_margin",
    "net_margin",
    "gross_margin",
    "current_ratio",
    "debt_to_equity",
    "revenue_growth_yoy",
}

SUPPORTED_METRICS = STATEMENT_METRICS | RATIO_METRICS
SUPPORTED_TICKERS = get_supported_tickers()


class FinancialQueryInput(BaseModel):
    """Input payload contract for querying standardized financial metrics in pure batch-first mode."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(
        default_factory=list,
        description="List of target company ticker symbols for batch execution.",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="List of target metric names for batch execution.",
    )
    fiscal_years: list[int] = Field(
        default_factory=list,
        description="List of fiscal years for multi-year trend queries.",
    )
    fiscal_periods: list[Literal["Q1", "Q2", "Q3", "Q4", "FY"]] = Field(
        default_factory=list,
        description="List of fiscal periods for multi-period queries.",
    )
    period_type: Optional[Literal["quarterly", "annual"]] = Field(
        default=None,
        description="Target period duration type. Automatically resolved if fiscal_period is specified.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_and_populate_lists(cls, data: Any) -> Any:
        """Convert scalar inputs into batch lists and strictly pop scalar keys without write-back."""
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # Pop scalar keys
        single_ticker = d.pop("ticker", None)
        single_metric = d.pop("metric", None)
        single_year = d.pop("fiscal_year", None)
        single_period = d.pop("fiscal_period", None)

        # Normalize tickers
        raw_tickers = list(d.get("tickers") or [])
        if single_ticker and single_ticker not in raw_tickers:
            raw_tickers.append(single_ticker)
        d["tickers"] = [t.strip().upper() for t in raw_tickers if isinstance(t, str)]

        # Normalize metrics
        raw_metrics = list(d.get("metrics") or [])
        if single_metric and single_metric not in raw_metrics:
            raw_metrics.append(single_metric)
        d["metrics"] = [m.strip().lower() for m in raw_metrics if isinstance(m, str)]

        # Normalize fiscal_years
        raw_years = list(d.get("fiscal_years") or [])
        if single_year is not None and single_year not in raw_years:
            raw_years.append(single_year)
        for y in raw_years:
            if not (isinstance(y, int) and 1900 <= y <= 2100):
                raise ValueError(f"Fiscal year must be a valid 4-digit calendar year between 1900 and 2100, got {y}")
        d["fiscal_years"] = raw_years

        # Normalize fiscal_periods
        raw_periods = list(d.get("fiscal_periods") or [])
        if single_period and single_period not in raw_periods:
            raw_periods.append(single_period)
        d["fiscal_periods"] = raw_periods

        return d

    @model_validator(mode="after")
    def validate_batch_rules(self) -> "FinancialQueryInput":
        """Validate ticker/metric whitelists and fiscal period consistency."""
        if not self.tickers:
            raise ValueError("At least one ticker must be specified.")
        supported = get_supported_tickers()
        for t in self.tickers:
            if t not in supported:
                raise ValueError(f"Unsupported ticker: '{t}'. Must be one of {sorted(supported)}")

        if not self.metrics:
            raise ValueError("At least one metric must be specified.")
        for m in self.metrics:
            if m not in SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: '{m}'. Must be one of {sorted(SUPPORTED_METRICS)}")

        for p in self.fiscal_periods:
            if p == "FY" and self.period_type == "quarterly":
                raise ValueError("Inconsistent query: fiscal_period 'FY' cannot have period_type 'quarterly'.")
            if p in ("Q1", "Q2", "Q3", "Q4") and self.period_type == "annual":
                raise ValueError(f"Inconsistent query: fiscal_period '{p}' cannot have period_type 'annual'.")

        if self.period_type is None and self.fiscal_periods:
            if all(p == "FY" for p in self.fiscal_periods):
                self.period_type = "annual"
            elif all(p in ("Q1", "Q2", "Q3", "Q4") for p in self.fiscal_periods):
                self.period_type = "quarterly"

        return self


class FinancialMetricRecord(BaseModel):
    """Standardized single financial observation with audit provenance."""

    record_key: str = Field(..., description="Surrogate primary key (statement_key or ratio_key)")
    ticker: str = Field(..., description="Company ticker symbol")
    cik: str = Field(..., description="10-digit SEC Central Index Key")
    fiscal_year: int = Field(..., description="SEC authoritative fiscal year")
    fiscal_period: str = Field(..., description="SEC authoritative fiscal period (Q1, Q2, Q3, Q4, FY)")
    period_type: str = Field(..., description="Duration scope: quarterly or annual")
    start_date: Optional[date] = Field(default=None, description="Start date of the period (null for instant)")
    end_date: Optional[date] = Field(default=None, description="End date of the period or balance sheet snapshot date")
    metric_name: str = Field(..., description="Standardized canonical metric name")
    metric_value: Optional[Decimal] = Field(
        default=None,
        description="Exact numerical metric value. Null if not reported (e.g. Alphabet gross profit).",
    )
    is_derived: bool = Field(
        default=False,
        description="True if record was derived via financial subtraction (e.g. Q4 = FY - 9M)",
    )
    derivation_method: Optional[str] = Field(
        default=None,
        description="Audit description of derivation formula if is_derived is true",
    )
    source_accession_number: Optional[str] = Field(
        default=None,
        description="SEC EDGAR accession number of the primary source filing",
    )
    source_filed_date: Optional[date] = Field(
        default=None,
        description="Filing date of the primary source document",
    )
    provenance_url: Optional[str] = Field(
        default=None,
        description="Direct clickable SEC EDGAR URL to the source filing index",
    )


class FinancialQueryResponse(BaseModel):
    """Standardized envelope response returned by FinancialSQLTool."""

    query: FinancialQueryInput = Field(..., description="Original validated query input")
    record_count: int = Field(..., description="Total matching records found")
    data: list[FinancialMetricRecord] = Field(default_factory=list, description="List of financial records")
    execution_time_ms: float = Field(default=0.0, description="Query execution latency in milliseconds")
    availability_reports: list[Any] = Field(
        default_factory=list,
        description="Audit reports on period availability and publication state",
    )


def generate_edgar_url(cik: str, accession_number: Optional[str]) -> Optional[str]:
    """Generate canonical SEC EDGAR filing index URL for audit verification.

    Format:
    https://www.sec.gov/Archives/edgar/data/{cik_without_leading_zeros}/{accession_nodash}/{accession_number}-index.htm

    Args:
        cik: SEC Central Index Key (e.g. '0000320193').
        accession_number: Primary document accession number (e.g. '0000320193-23-000106').

    Returns:
        Canonical URL string, or None if accession_number is missing.
    """
    if not accession_number or not cik:
        return None

    try:
        cik_clean = str(int(cik.strip()))
    except (ValueError, TypeError):
        cik_clean = cik.strip().lstrip("0") or "0"

    accession_clean = accession_number.strip()
    accession_nodash = accession_clean.replace("-", "")

    return f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_nodash}/{accession_clean}-index.htm"


class FinancialSQLTool:
    """Tool executing parameterized SQL queries over standardized financial marts."""

    name: str = "financial_sql_tool"
    description: str = (
        "Query standardized US-GAAP financial statement metrics and ratios for supported companies. "
        "Supports revenue, net income, operating income, gross profit, diluted EPS, balance sheet metrics, "
        "and ratios (margins, liquidity, debt-to-equity, YoY growth) with audit citations."
    )

    def __init__(
        self,
        repository: Optional["DuckDBFinancialRepository"] = None,
        availability_validator: Optional[Any] = None,
    ):
        """Initialize tool with repository instance or lazy-load default repository."""
        if repository is None:
            from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
            self.repository = DuckDBFinancialRepository()
        else:
            self.repository = repository

        if availability_validator is None:
            from src.core.financial.availability_validator import FinancialAvailabilityValidator
            self.availability_validator = FinancialAvailabilityValidator(repository=self.repository)
        else:
            self.availability_validator = availability_validator

    def execute(self, query: FinancialQueryInput | dict[str, Any]) -> FinancialQueryResponse:
        """Execute validated query, enrich with SEC provenance citations, and return envelope.

        Args:
            query: Either a validated FinancialQueryInput or raw dictionary matching the schema.

        Returns:
            FinancialQueryResponse containing matched records, execution latency, and citations.
        """
        start_time = time.perf_counter()

        validated_query = (
            query if isinstance(query, FinancialQueryInput) else FinancialQueryInput(**query)
        )

        # 1. Gatekeeper: Validate requested periods dynamically against SEC metadata BEFORE SQL
        valid_years, availability_reports = self.availability_validator.validate_periods(
            tickers=validated_query.tickers,
            fiscal_years=validated_query.fiscal_years,
            fiscal_periods=validated_query.fiscal_periods,
            conn=self.repository.conn,
            statements_path=self.repository.statements_path,
        )

        # If user explicitly requested years, but none are published/available (e.g. future periods)
        if validated_query.fiscal_years and not valid_years:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return FinancialQueryResponse(
                query=validated_query,
                record_count=0,
                data=[],
                execution_time_ms=elapsed_ms,
                availability_reports=availability_reports,
            )

        # Query only valid and available years from DuckDB
        query_to_execute = validated_query
        if validated_query.fiscal_years and len(valid_years) < len(validated_query.fiscal_years):
            query_to_execute = validated_query.model_copy(update={"fiscal_years": valid_years})

        raw_rows = self.repository.query_metric(query_to_execute)

        records: list[FinancialMetricRecord] = []
        for row in raw_rows:
            provenance_url = generate_edgar_url(
                cik=row.get("cik", ""),
                accession_number=row.get("source_accession_number"),
            )

            raw_val = row.get("metric_value")
            metric_val: Optional[Decimal] = None
            if raw_val is not None:
                metric_val = raw_val if isinstance(raw_val, Decimal) else Decimal(str(raw_val))

            record = FinancialMetricRecord(
                record_key=str(row["record_key"]),
                ticker=row["ticker"],
                cik=row["cik"],
                fiscal_year=int(row["fiscal_year"]),
                fiscal_period=row["fiscal_period"],
                period_type=row["period_type"],
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                metric_name=row["metric_name"],
                metric_value=metric_val,
                is_derived=bool(row.get("is_derived", False)),
                derivation_method=row.get("derivation_method"),
                source_accession_number=row.get("source_accession_number"),
                source_filed_date=row.get("source_filed_date"),
                provenance_url=provenance_url,
            )
            records.append(record)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return FinancialQueryResponse(
            query=validated_query,
            record_count=len(records),
            data=records,
            execution_time_ms=elapsed_ms,
            availability_reports=availability_reports,
        )

    def get_metric(
        self,
        ticker: str,
        metric: str,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[Literal["Q1", "Q2", "Q3", "Q4", "FY"]] = None,
    ) -> FinancialQueryResponse:
        """Convenience method for querying a single metric."""
        query = FinancialQueryInput(
            tickers=[ticker],
            metrics=[metric],
            fiscal_years=[fiscal_year] if fiscal_year is not None else [],
            fiscal_periods=[fiscal_period] if fiscal_period is not None else [],
        )
        return self.execute(query)

    def compare_companies(
        self,
        metric: str,
        fiscal_year: int,
        fiscal_period: Literal["Q1", "Q2", "Q3", "Q4", "FY"],
        tickers: Optional[list[str]] = None,
    ) -> list[FinancialMetricRecord]:
        """Convenience method comparing a specific metric across multiple companies for a period."""
        target_tickers = tickers if tickers is not None else sorted(list(get_supported_tickers()))
        query = FinancialQueryInput(
            tickers=target_tickers,
            metrics=[metric],
            fiscal_years=[fiscal_year],
            fiscal_periods=[fiscal_period],
        )
        return self.execute(query).data
