"""Unit tests for Vietnamese VN30 Registry and VAS Taxonomy Map."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def companies_vn_data() -> dict:
    """Load configs/companies_vn.yaml."""
    config_path = Path("configs/companies_vn.yaml")
    assert config_path.exists(), "configs/companies_vn.yaml must exist."
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def vas_taxonomy_data() -> dict:
    """Load configs/vas_taxonomy_map.yaml."""
    config_path = Path("configs/vas_taxonomy_map.yaml")
    assert config_path.exists(), "configs/vas_taxonomy_map.yaml must exist."
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def us_taxonomy_data() -> dict:
    """Load configs/xbrl_taxonomy_map.yaml for cross-market metric alignment."""
    config_path = Path("configs/xbrl_taxonomy_map.yaml")
    assert config_path.exists(), "configs/xbrl_taxonomy_map.yaml must exist."
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestVN30Registry:
    """Validate data contracts of configs/companies_vn.yaml."""

    def test_total_companies_count(self, companies_vn_data: dict):
        """Verify 50 Vietnam companies (30 VN30 + 20 expanded HOSE leaders) are defined."""
        companies = companies_vn_data.get("companies", [])
        assert len(companies) == 50, f"Expected 50 companies, found {len(companies)}"
        assert companies_vn_data.get("total_companies") == 50
        vn30 = [c for c in companies if c.get("index_membership", "VN30") == "VN30"]
        assert len(vn30) == 30, f"Expected 30 VN30 constituents, found {len(vn30)}"

    def test_unique_tickers(self, companies_vn_data: dict):
        """Ensure all 30 tickers are uppercase, alphanumeric, and strictly unique."""
        companies = companies_vn_data.get("companies", [])
        tickers = [c["ticker"] for c in companies]
        assert len(tickers) == len(set(tickers)), "Duplicate tickers found in companies_vn.yaml"
        for t in tickers:
            assert t.isupper(), f"Ticker {t} must be uppercase"
            assert len(t) == 3, f"HOSE ticker {t} must be 3 characters"

    def test_required_metadata_fields(self, companies_vn_data: dict):
        """Verify every constituent contains all required portfolio and legal fields."""
        required_fields = {
            "ticker",
            "name",
            "english_name",
            "sector",
            "industry",
            "exchange",
            "currency",
            "accounting_standard",
            "peer_benchmark",
        }
        for c in companies_vn_data.get("companies", []):
            missing = required_fields - set(c.keys())
            assert not missing, f"Company {c.get('ticker')} missing required fields: {missing}"
            assert c["exchange"] == "HOSE", f"{c['ticker']} must be listed on HOSE"
            assert c["currency"] == "VND", f"{c['ticker']} currency must be VND"
            assert c["accounting_standard"] in ("VAS", "VAS-BANK", "VAS-SECURITIES", "VAS-INSURANCE")


class TestVASTaxonomyMap:
    """Validate completeness and schema integrity of configs/vas_taxonomy_map.yaml."""

    def test_canonical_metrics_alignment_with_us_gaap(self, vas_taxonomy_data: dict, us_taxonomy_data: dict):
        """Ensure core financial metrics have exact naming parity across US and VN markets."""
        vas_metrics = set(vas_taxonomy_data.get("metrics", {}).keys())
        us_metrics = set(us_taxonomy_data.get("metrics", {}).keys())

        core_common_metrics = [
            "revenues",
            "gross_profit",
            "operating_income",
            "net_income",
            "eps_diluted",
            "total_assets",
            "current_assets",
            "current_liabilities",
            "cash_and_cash_equivalents",
            "total_liabilities",
            "stockholders_equity_total",
            "operating_cash_flow",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "debt_to_equity",
        ]
        for metric in core_common_metrics:
            assert metric in vas_metrics, f"Missing canonical metric {metric} in vas_taxonomy_map.yaml"
            assert metric in us_metrics, f"Metric {metric} not found in xbrl_taxonomy_map.yaml"

    def test_banking_specific_metrics_exist(self, vas_taxonomy_data: dict):
        """Verify specialized commercial banking metrics exist for 11 VN30 banks."""
        vas_metrics = vas_taxonomy_data.get("metrics", {})
        banking_metrics = [
            "net_interest_income",
            "net_fee_income",
            "total_operating_income",
            "provision_expense",
            "customer_loans",
            "customer_deposits",
        ]
        for bm in banking_metrics:
            assert bm in vas_metrics, f"Missing banking metric {bm} in vas_taxonomy_map.yaml"
            assert vas_metrics[bm]["default_unit"] == "VND"

    def test_accounting_equations_and_constraints(self, vas_taxonomy_data: dict):
        """Verify constraints and equations are defined for validation."""
        constraints = vas_taxonomy_data.get("constraints", {})
        assert "VND" in constraints.get("allowed_units", [])
        assert "VND/shares" in constraints.get("allowed_units", [])
        equations = constraints.get("accounting_equations", {})
        assert "corporate_balance_sheet" in equations
        assert "corporate_gross_profit" in equations
