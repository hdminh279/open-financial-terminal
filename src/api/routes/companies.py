"""Companies catalog router.

Provides metadata regarding covered enterprises, CIK identifiers, and reporting ranges.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter, status
import yaml

from src.api.contracts import CompanyItem, CompanyListResponse

router = APIRouter(prefix="/companies", tags=["Companies"])


def _get_lakehouse_supported_years() -> Dict[str, List[int]]:
    """Retrieve verified fiscal years for all tickers directly from DuckDB Lakehouse."""
    years_by_ticker: Dict[str, List[int]] = defaultdict(list)
    try:
        from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
        repo = DuckDBFinancialRepository()
        query = f"SELECT ticker, fiscal_year FROM '{repo.statements_path}' GROUP BY ticker, fiscal_year ORDER BY ticker, fiscal_year"
        rows = repo.conn.execute(query).fetchall()
        for t, y in rows:
            years_by_ticker[str(t).upper()].append(int(y))
    except Exception:
        pass
    return dict(years_by_ticker)


@router.get(
    "",
    response_model=CompanyListResponse,
    status_code=status.HTTP_200_OK,
    summary="List supported enterprise entities",
    description="Returns universe of analyzed public corporations with CIKs and dynamically verified fiscal years.",
)
async def list_companies() -> CompanyListResponse:
    """Return catalog of covered companies from configuration and lakehouse."""
    companies_path = Path("configs/companies.yaml")
    companies_data = []

    if companies_path.exists():
        with open(companies_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
            companies_data = raw_cfg.get("companies", [])

    lakehouse_years = _get_lakehouse_supported_years()
    current_year = datetime.now().year

    items = []
    for c in companies_data:
        ticker = c["ticker"].upper()
        # Prefer dynamic lakehouse years; fallback to config or dynamic range
        years = lakehouse_years.get(ticker) or c.get("supported_years") or list(range(2009, current_year + 1))
        items.append(
            CompanyItem(
                ticker=c["ticker"],
                name=c["name"],
                cik=str(c["cik"]),
                supported_years=years,
                available_forms=["10-K", "10-Q"],
            )
        )

    return CompanyListResponse(companies=items)
