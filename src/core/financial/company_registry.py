"""Company Registry providing dynamic ticker and enterprise metadata resolution (SSOT)."""

import os
from typing import Any, Dict, List, Set
import yaml

DEFAULT_COMPANIES_PATH = "configs/companies.yaml"


def load_company_catalog(config_path: str = DEFAULT_COMPANIES_PATH) -> List[Dict[str, Any]]:
    """Load list of company dictionary definitions from configuration YAML."""
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("companies", [])


def get_supported_tickers(config_path: str = DEFAULT_COMPANIES_PATH) -> Set[str]:
    """Retrieve set of uppercase canonical company ticker symbols from configuration."""
    companies = load_company_catalog(config_path)
    return {str(c["ticker"]).upper() for c in companies if "ticker" in c}
