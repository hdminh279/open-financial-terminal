"""Financial metrics router providing direct Gold Parquet querying.

Provides ultra-fast (<20ms) DuckDB SQL queries over standardized financial statements
and financial ratios with SEC EDGAR provenance URLs.
"""

import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from backend.ai.agent.tools.financial_sql_tool import (
    RATIO_METRICS,
    SUPPORTED_METRICS,
    FinancialQueryInput,
    FinancialSQLTool,
)
from backend.contracts import MetricRecordResponse, MetricResponse
from backend.core.financial.company_registry import get_supported_tickers, resolve_filing_form
from backend.deps import get_duckdb_repository, get_sql_tool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["Financial Metrics"])


@router.get(
    "",
    response_model=MetricResponse,
    status_code=status.HTTP_200_OK,
    summary="Query standardized financial metrics",
    description="Direct SQL retrieval over Gold Parquet marts (Statements & Ratios) with sub-20ms latency.",
)
def get_financial_metric(
    ticker: str | None = Query(default=None, description="Scalar company ticker symbol"),
    tickers: list[str] | None = Query(default=None, description="List of company ticker symbols"),
    metric: str | None = Query(default=None, description="Scalar canonical metric name"),
    metrics: list[str] | None = Query(default=None, description="List of canonical metric names"),
    fiscal_year: int | None = Query(
        default=None, ge=2000, le=2050, description="Fiscal year of the filing (e.g. 2020-2026+)"
    ),
    fiscal_years: list[int] | None = Query(default=None, description="List of fiscal years"),
    fiscal_period: Literal["Q1", "Q2", "Q3", "Q4", "FY"] | None = Query(default=None, description="Fiscal period"),
    fiscal_periods: list[Literal["Q1", "Q2", "Q3", "Q4", "FY"]] | None = Query(
        default=None, description="List of fiscal periods"
    ),
    repository: DuckDBFinancialRepository = Depends(get_duckdb_repository),
) -> MetricResponse:
    """Retrieve financial observations with full SEC provenance URLs."""
    raw_tickers = list(tickers or [])
    if ticker and ticker not in raw_tickers:
        raw_tickers.append(ticker)
    normalized_tickers = [t.strip().upper() for t in raw_tickers if isinstance(t, str)]

    raw_metrics = list(metrics or [])
    if metric and metric not in raw_metrics:
        raw_metrics.append(metric)
    normalized_metrics = [m.strip().lower() for m in raw_metrics if isinstance(m, str)]

    if not normalized_tickers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one ticker must be specified.",
        )

    supported_tickers = get_supported_tickers()
    for t in normalized_tickers:
        if t not in supported_tickers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported ticker '{t}'. Supported tickers are: {sorted(supported_tickers)}",
            )

    if not normalized_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one metric must be specified.",
        )

    for m in normalized_metrics:
        if m not in SUPPORTED_METRICS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported metric '{m}'. Supported metrics include: {sorted(list(SUPPORTED_METRICS))}",
            )

    years = list(fiscal_years or [])
    if fiscal_year is not None and fiscal_year not in years:
        years.append(fiscal_year)

    periods = list(fiscal_periods or [])
    if fiscal_period and fiscal_period not in periods:
        periods.append(fiscal_period)

    try:
        sql_tool = get_sql_tool() if repository is get_duckdb_repository() else FinancialSQLTool(repository=repository)
        query_input = FinancialQueryInput(
            tickers=normalized_tickers,
            metrics=normalized_metrics,
            fiscal_years=years,
            fiscal_periods=periods,
        )

        result = sql_tool.execute(query_input)

        records: list[MetricRecordResponse] = []
        for r in result.data:
            val_float = float(r.metric_value) if r.metric_value is not None else None
            unit_label = "Ratio" if r.metric_name in RATIO_METRICS else (r.currency or "USD")
            records.append(
                MetricRecordResponse(
                    ticker=r.ticker,
                    fiscal_year=r.fiscal_year,
                    fiscal_period=r.fiscal_period,
                    period_type=r.period_type,
                    metric_name=r.metric_name,
                    value=val_float,
                    unit=unit_label,
                    filed_date=str(r.source_filed_date) if r.source_filed_date else None,
                    form=resolve_filing_form(r.ticker, r.cik, r.fiscal_period),
                    provenance_url=r.provenance_url,
                )
            )

        return MetricResponse(
            tickers=normalized_tickers,
            metrics=normalized_metrics,
            count=len(records),
            records=records,
            execution_time_ms=result.execution_time_ms,
        )

    except Exception as exc:
        logger.error("Failed querying metrics %s for %s: %s", normalized_metrics, normalized_tickers, exc)
        is_prod = os.getenv("APP_ENV", "local").lower() in ("prod", "production")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database query execution failed." if is_prod else f"Database query execution failed: {exc}",
        ) from exc
