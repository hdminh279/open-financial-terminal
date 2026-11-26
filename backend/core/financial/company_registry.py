"""Company Registry providing dynamic ticker and enterprise metadata resolution (SSOT)."""

import os
from functools import lru_cache
from typing import Any

import yaml

DEFAULT_COMPANIES_PATH = "configs/companies.yaml"
DEFAULT_VN_COMPANIES_PATH = "configs/companies_vn.yaml"


@lru_cache(maxsize=8)
def _load_company_catalog_cached(config_path: str, include_vn: bool = True) -> tuple[dict[str, Any], ...]:
    """Read and cache company definitions from YAML configuration."""
    catalogs: list[dict[str, Any]] = []
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        catalogs.extend(data.get("companies", []))

    # Also include VN30 catalog when loading standard catalog
    if include_vn and config_path == DEFAULT_COMPANIES_PATH and os.path.exists(DEFAULT_VN_COMPANIES_PATH):
        with open(DEFAULT_VN_COMPANIES_PATH, encoding="utf-8") as f:
            vn_data = yaml.safe_load(f) or {}
        catalogs.extend(vn_data.get("companies", []))

    return tuple(catalogs)


def load_company_catalog(
    config_path: str = DEFAULT_COMPANIES_PATH,
    include_vn: bool = True,
) -> list[dict[str, Any]]:
    """Load list of company dictionary definitions from configuration YAML."""
    return list(_load_company_catalog_cached(config_path, include_vn=include_vn))


def get_supported_tickers(config_path: str = DEFAULT_COMPANIES_PATH) -> set[str]:
    """Retrieve set of uppercase canonical company ticker symbols from configuration."""
    companies = _load_company_catalog_cached(config_path)
    return {str(c["ticker"]).upper() for c in companies if "ticker" in c}


def get_company_form_map(config_path: str = DEFAULT_COMPANIES_PATH) -> dict[str, str]:
    """Return mapping of uppercase ticker to annual filing form ('10-K', '20-F', etc.)."""
    return {
        str(c["ticker"]).upper(): str(c.get("form_type", "10-K"))
        for c in _load_company_catalog_cached(config_path)
        if c.get("ticker")
    }


def resolve_filing_form(ticker: str, cik: str, fiscal_period: str) -> str:
    """Resolve authoritative filing form label (BCTC, 20-F, 10-K, 10-Q, 6-K) for a record."""
    if cik.strip().upper().startswith("VN:"):
        return "BCTC"
    annual_form = get_company_form_map().get(ticker.upper(), "10-K")
    # SEC Q4 figures are derived from annual 10-K / 20-F filings (no separate Q4 10-Q exists)
    if annual_form == "20-F":
        return "20-F" if fiscal_period in ("FY", "Q4") else "6-K"
    return "10-K" if fiscal_period in ("FY", "Q4") else "10-Q"
