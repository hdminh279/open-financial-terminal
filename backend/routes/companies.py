"""Companies catalog router.

Provides metadata regarding covered enterprises, CIK identifiers, and reporting ranges.
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, Depends, status

from backend.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from backend.contracts import CompanyItem, CompanyListResponse
from backend.core.financial.company_registry import load_company_catalog
from backend.deps import get_duckdb_repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/companies", tags=["Companies"])

_LAKEHOUSE_YEARS_TTL_SECONDS = 600.0
_lakehouse_years_cache: dict[str, tuple[float, dict[str, list[int]]]] = {}
_lakehouse_years_lock = threading.Lock()


def _get_lakehouse_supported_years(repo: DuckDBFinancialRepository) -> dict[str, list[int]]:
    """Retrieve verified fiscal years for all tickers from DuckDB Lakehouse with a 10-min TTL cache."""
    cache_key = repo.statements_path
    now = time.monotonic()
    with _lakehouse_years_lock:
        cached = _lakehouse_years_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < _LAKEHOUSE_YEARS_TTL_SECONDS:
            return cached[1]

    years_by_ticker: dict[str, list[int]] = defaultdict(list)
    try:
        rows = repo.execute_fetchall(
            "SELECT ticker, fiscal_year FROM read_parquet(?) GROUP BY ticker, fiscal_year ORDER BY ticker, fiscal_year",
            [repo.statements_path],
        )
        for ticker_val, year_val in rows:
            years_by_ticker[str(ticker_val).upper()].append(int(year_val))
    except Exception as exc:
        logger.warning("Failed to inspect lakehouse years for companies catalog: %s", exc)
        return {}

    result = dict(years_by_ticker)
    with _lakehouse_years_lock:
        _lakehouse_years_cache[cache_key] = (time.monotonic(), result)
    return result


@router.get(
    "",
    response_model=CompanyListResponse,
    status_code=status.HTTP_200_OK,
    summary="List supported enterprise entities",
    description="Returns universe of analyzed public corporations with CIKs and dynamically verified fiscal years.",
)
def list_companies(
    repository: DuckDBFinancialRepository = Depends(get_duckdb_repository),
) -> CompanyListResponse:
    """Return catalog of covered companies from configuration and lakehouse."""
    companies_data = load_company_catalog()
    lakehouse_years = _get_lakehouse_supported_years(repository)
    current_year = datetime.now().year

    items = []
    for c in companies_data:
        ticker = str(c["ticker"]).upper()
        years = lakehouse_years.get(ticker) or c.get("supported_years") or list(range(2009, current_year + 1))
        form_type = str(c.get("form_type", "")).upper()
        if c.get("market") == "VN" or str(c.get("cik", "")).startswith("VN:"):
            available_forms = ["BCTC"]
        elif form_type == "20-F":
            available_forms = ["20-F", "6-K"]
        else:
            available_forms = ["10-K", "10-Q"]

        items.append(
            CompanyItem(
                ticker=ticker,
                name=c["name"],
                cik=str(c.get("cik") or f"VN:{ticker}"),
                supported_years=years,
                available_forms=available_forms,
            )
        )

    return CompanyListResponse(companies=items)
