"""Financial metrics router providing direct Gold Parquet querying.

Provides ultra-fast (<20ms) DuckDB SQL queries over standardized financial statements
and financial ratios with SEC EDGAR provenance URLs.
"""

import logging
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.agent.tools.financial_sql_tool import (
    FinancialQueryInput,
    FinancialSQLTool,
    SUPPORTED_METRICS,
    SUPPORTED_TICKERS,
)
from src.api.contracts import MetricRecordResponse, MetricResponse
from src.api.deps import get_duckdb_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["Financial Metrics"])


@router.get(
    "",
    response_model=MetricResponse,
    status_code=status.HTTP_200_OK,
    summary="Query standardized financial metrics",
    description="Direct SQL retrieval over Gold Parquet marts (Statements & Ratios) with sub-20ms latency.",
)
async def get_financial_metric(
    ticker: Optional[str] = Query(default=None, description="Scalar company ticker symbol"),
    tickers: Optional[list[str]] = Query(default=None, description="List of company ticker symbols"),
    metric: Optional[str] = Query(default=None, description="Scalar canonical metric name"),
    metrics: Optional[list[str]] = Query(default=None, description="List of canonical metric names"),
    fiscal_year: Optional[int] = Query(default=None, ge=2000, le=2050, description="Fiscal year of the filing (e.g. 2020-2026+)"),
    fiscal_years: Optional[list[int]] = Query(default=None, description="List of fiscal years"),
    fiscal_period: Optional[Literal["Q1", "Q2", "Q3", "Q4", "FY"]] = Query(
        default=None, description="Fiscal period"
    ),
    fiscal_periods: Optional[list[Literal["Q1", "Q2", "Q3", "Q4", "FY"]]] = Query(
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

    for t in normalized_tickers:
        if t not in SUPPORTED_TICKERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported ticker '{t}'. Supported tickers are: {sorted(list(SUPPORTED_TICKERS))}",
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
        sql_tool = FinancialSQLTool(repository=repository)
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
            records.append(
                MetricRecordResponse(
                    ticker=r.ticker,
                    fiscal_year=r.fiscal_year,
                    fiscal_period=r.fiscal_period,
                    period_type=r.period_type,
                    metric_name=r.metric_name,
                    value=val_float,
                    unit="USD" if "margin" not in r.metric_name and "ratio" not in r.metric_name else "Ratio",
                    filed_date=str(r.source_filed_date) if r.source_filed_date else None,
                    form="10-K" if r.fiscal_period == "FY" else "10-Q",
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query execution failed: {str(exc)}",
        )
        logger.error("Failed querying metric %s for %s: %s", normalized_metric, normalized_ticker, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query execution failed: {str(exc)}",
        )
