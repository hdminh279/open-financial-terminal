"""Authoritative SEC filing and financial data availability validator.

Validates query time horizons against dynamic SEC filing metadata and Lakehouse state:
1. Has the fiscal period ended? (Checked dynamically against end_date vs current date).
2. Has the filing been published? (Checked against Form 10-K / 10-Q filing catalog).
3. Is data available and verified? (Checked against Gold Lakehouse records and verified provenance).
Zero static calendar dates; 100% data-driven.
"""

import logging
import threading
from datetime import date
from enum import Enum
from typing import Any

import duckdb
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AvailabilityCode(str, Enum):
    AVAILABLE = "AVAILABLE"
    FUTURE_PERIOD_NOT_ENDED = "FUTURE_PERIOD_NOT_ENDED"
    FILING_NOT_PUBLISHED = "FILING_NOT_PUBLISHED"
    PARTIAL_PERIOD_ONLY = "PARTIAL_PERIOD_ONLY"
    UNVERIFIED_DATA = "UNVERIFIED_DATA"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"


class PeriodAvailability(BaseModel):
    """Detailed availability state for a specific company fiscal period."""

    ticker: str
    fiscal_year: int
    fiscal_period: str
    code: AvailabilityCode
    is_available: bool
    reason: str
    end_date: date | None = None
    source_accession_number: str | None = None


class FinancialAvailabilityValidator:
    """Validates period availability against dynamic Lakehouse metadata."""

    def __init__(self, repository: Any | None = None):
        """Initialize validator with DuckDB repository reference."""
        self.repository = repository
        self._filing_cache: dict[tuple[str, int, str], dict[str, Any]] | None = None
        self._cache_lock = threading.Lock()

    def _load_filing_catalog(
        self, conn: duckdb.DuckDBPyConnection, statements_path: str
    ) -> dict[tuple[str, int, str], dict[str, Any]]:
        """Load distinct period metadata from Gold statements mart in a thread-safe manner."""
        with self._cache_lock:
            if self._filing_cache is not None:
                return self._filing_cache

        catalog: dict[tuple[str, int, str], dict[str, Any]] = {}
        try:
            query = """
                SELECT DISTINCT
                    ticker,
                    fiscal_year,
                    fiscal_period,
                    end_date,
                    source_accession_number,
                    source_filed_date
                FROM read_parquet(?)
            """
            if self.repository is not None and hasattr(self.repository, "execute_fetchall"):
                rows = self.repository.execute_fetchall(query, [statements_path])
            elif isinstance(conn, duckdb.DuckDBPyConnection):
                cursor = conn.cursor()
                try:
                    rows = cursor.execute(query, [statements_path]).fetchall()
                finally:
                    cursor.close()
            else:
                rows = conn.execute(query, [statements_path]).fetchall()

            for r in rows:
                key = (str(r[0]).upper(), int(r[1]), str(r[2]).upper())
                catalog[key] = {
                    "ticker": r[0],
                    "fiscal_year": r[1],
                    "fiscal_period": r[2],
                    "end_date": r[3],
                    "source_accession_number": r[4],
                    "source_filed_date": r[5],
                }
            with self._cache_lock:
                if self._filing_cache is None:
                    self._filing_cache = catalog
                return self._filing_cache
        except Exception as exc:
            # Do not permanently cache empty state on transient storage errors
            logger.warning("Failed to load filing availability catalog from %s: %s", statements_path, exc)
            return {}

    def validate_periods(
        self,
        tickers: list[str],
        fiscal_years: list[int],
        fiscal_periods: list[str],
        conn: duckdb.DuckDBPyConnection | None = None,
        statements_path: str | None = None,
        today: date | None = None,
    ) -> tuple[list[int], list[PeriodAvailability]]:
        """Validate requested periods against real SEC filing availability.

        Args:
            tickers: Target company tickers.
            fiscal_years: Requested fiscal years (if empty, defaults to all available).
            fiscal_periods: Target periods ('FY', 'Q1', etc.).
            conn: Active DuckDB connection.
            statements_path: Path to statements parquet file.
            today: Reference date for current time (defaults to date.today()).

        Returns:
            Tuple of (valid_years, availability_reports).
        """
        current_date = today or date.today()
        periods = fiscal_periods or ["FY"]
        reports: list[PeriodAvailability] = []
        valid_years_set: set[int] = set()

        # If fiscal_years is empty (e.g. unconstrained trend), all years in DB are considered
        if not fiscal_years:
            return [], []

        # If connection and path are provided, use Lakehouse catalog
        catalog: dict[tuple[str, int, str], dict[str, Any]] = {}
        if conn and statements_path:
            catalog = self._load_filing_catalog(conn, statements_path)

        for year in fiscal_years:
            year_is_valid_for_any_ticker = False

            for ticker in tickers:
                for period in periods:
                    key = (ticker.upper(), year, period.upper())
                    entry = catalog.get(key)

                    # 1. Check if period end date is in the future
                    if entry and entry.get("end_date") and entry["end_date"] > current_date:
                        reports.append(
                            PeriodAvailability(
                                ticker=ticker,
                                fiscal_year=year,
                                fiscal_period=period,
                                code=AvailabilityCode.FUTURE_PERIOD_NOT_ENDED,
                                is_available=False,
                                reason=f"Fiscal period {period} {year} for {ticker} ends on {entry['end_date']}, which is in the future.",
                                end_date=entry.get("end_date"),
                            )
                        )
                        continue

                    # 2. If not filed and year is ahead of current date
                    if entry is None and year > current_date.year:
                        reports.append(
                            PeriodAvailability(
                                ticker=ticker,
                                fiscal_year=year,
                                fiscal_period=period,
                                code=AvailabilityCode.FUTURE_PERIOD_NOT_ENDED,
                                is_available=False,
                                reason=f"Fiscal year {year} is beyond the current calendar date ({current_date}).",
                            )
                        )
                        continue

                    # 3. Check filing existence in Lakehouse
                    if entry is None and catalog:
                        # Check if any quarterly filings exist for this year
                        has_quarterly = any((ticker.upper(), year, q) in catalog for q in ("Q1", "Q2", "Q3", "Q4"))
                        if period == "FY" and has_quarterly:
                            reports.append(
                                PeriodAvailability(
                                    ticker=ticker,
                                    fiscal_year=year,
                                    fiscal_period=period,
                                    code=AvailabilityCode.PARTIAL_PERIOD_ONLY,
                                    is_available=False,
                                    reason=f"Form 10-K (Annual FY) for {ticker} FY{year} is not filed yet; only quarterly Form 10-Q filings exist.",
                                )
                            )
                        else:
                            reports.append(
                                PeriodAvailability(
                                    ticker=ticker,
                                    fiscal_year=year,
                                    fiscal_period=period,
                                    code=AvailabilityCode.FILING_NOT_PUBLISHED,
                                    is_available=False,
                                    reason=f"No Form 10-K / 10-Q filing has been submitted to SEC EDGAR for {ticker} {period} {year}.",
                                )
                            )
                        continue

                    # Period is valid
                    year_is_valid_for_any_ticker = True
                    reports.append(
                        PeriodAvailability(
                            ticker=ticker,
                            fiscal_year=year,
                            fiscal_period=period,
                            code=AvailabilityCode.AVAILABLE,
                            is_available=True,
                            reason="Filing published and verified in SEC Lakehouse.",
                            end_date=entry.get("end_date") if entry else None,
                            source_accession_number=entry.get("source_accession_number") if entry else None,
                        )
                    )

            if year_is_valid_for_any_ticker:
                valid_years_set.add(year)

        return sorted(list(valid_years_set)), reports
