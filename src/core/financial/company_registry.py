"""Company Registry providing dynamic ticker and enterprise metadata resolution (SSOT)."""

import os
from typing import Any, Dict, List, Set
import yaml

DEFAULT_COMPANIES_PATH = "configs/companies.yaml"
DEFAULT_VN_COMPANIES_PATH = "configs/companies_vn.yaml"


def load_company_catalog(config_path: str = DEFAULT_COMPANIES_PATH) -> List[Dict[str, Any]]:
    """Load list of company dictionary definitions from configuration YAML."""
    catalogs: List[Dict[str, Any]] = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        catalogs.extend(data.get("companies", []))

    # Also include VN30 catalog when loading standard catalog
    if config_path == DEFAULT_COMPANIES_PATH and os.path.exists(DEFAULT_VN_COMPANIES_PATH):
        with open(DEFAULT_VN_COMPANIES_PATH, "r", encoding="utf-8") as f:
            vn_data = yaml.safe_load(f) or {}
        catalogs.extend(vn_data.get("companies", []))

    return catalogs


def get_supported_tickers(config_path: str = DEFAULT_COMPANIES_PATH) -> Set[str]:
    """Retrieve set of uppercase canonical company ticker symbols from configuration."""
    companies = load_company_catalog(config_path)
    return {str(c["ticker"]).upper() for c in companies if "ticker" in c}
