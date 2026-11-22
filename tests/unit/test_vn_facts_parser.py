"""Unit tests for Vietnamese Financial Facts Parser and Silver Data Quality Gates."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.data.processing.vn_facts_parser import (
    ALLOWED_UNITS,
    VNFactsParser,
)


@pytest.fixture
def mock_storage():
    """Create a mock S3Storage instance."""
    storage = MagicMock()
    storage.upload_bytes.return_value = True
    return storage


@pytest.fixture
def parser(mock_storage):
    """Instantiate VNFactsParser with mock storage."""
    return VNFactsParser(
        s3_storage=mock_storage,
        bronze_bucket="test-bronze",
        silver_bucket="test-silver",
        companies_vn_path="configs/companies_vn.yaml",
    )


class TestVNFactsParserQualityGates:
    """Validate all 6 Data Quality Gate rules enforced during Bronze to Silver processing."""

    def test_rule_1_empty_payload_anomaly(self, parser, mock_storage):
        """Rule 1: Empty upstream JSON payload must be quarantined with EMPTY_PAYLOAD_ANOMALY."""
        raw_json = json.dumps(
            {
                "ticker": "FPT",
                "report_type": "incomestatement",
                "period": "yearly",
                "source_url": "https://api-finfo.vndirect.com.vn",
                "data": [],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file(
            bucket="test-bronze",
            key="vn_raw/financials/FPT_incomestatement_yearly.json",
        )

        assert len(clean) == 0
        assert len(quarantine) == 1
        assert "EMPTY_PAYLOAD_ANOMALY" in quarantine[0]["quarantine_reason"]
        assert quarantine[0]["ticker"] == "FPT"

    def test_rule_2_missing_period_gap(self, parser):
        """Rule 2: Detect chronological quarterly reporting gaps (e.g. Q1 and Q3 exist but Q2 missing)."""
        records = [
            {"reportType": "QUARTER", "fiscalDate": "2023-03-31", "itemCode": 21001, "numericValue": 100},
            {"reportType": "QUARTER", "fiscalDate": "2023-09-30", "itemCode": 21001, "numericValue": 120},
            {"reportType": "QUARTER", "fiscalDate": "2023-12-31", "itemCode": 21001, "numericValue": 150},
        ]
        anomalies = parser.validate_quarterly_continuity("FPT", records, "test_file.json")

        assert len(anomalies) == 1
        assert "MISSING_PERIOD_GAP" in anomalies[0]["quarantine_reason"]
        assert "FY2023" in anomalies[0]["quarantine_reason"]
        assert "Q2" in anomalies[0]["quarantine_reason"]

    def test_rule_3_unsupported_currency(self, parser, mock_storage):
        """Rule 3: Monetary units outside standard whitelist must be quarantined."""
        raw_json = json.dumps(
            {
                "ticker": "FPT",
                "report_type": "incomestatement",
                "period": "yearly",
                "data": [
                    {
                        "itemCode": 21001,
                        "numericValue": 50000000000,
                        "fiscalDate": "2023-12-31",
                    }
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        # Temporarily tamper allowed units to simulate unsupported currency
        original_units = set(ALLOWED_UNITS)
        try:
            ALLOWED_UNITS.clear()
            ALLOWED_UNITS.add("USD")  # VND is now disallowed

            clean, quarantine = parser.parse_bronze_file("test-bronze", "test.json")
            assert len(clean) == 0
            assert len(quarantine) == 1
            assert "UNSUPPORTED_CURRENCY" in quarantine[0]["quarantine_reason"]
        finally:
            ALLOWED_UNITS.clear()
            ALLOWED_UNITS.update(original_units)

    def test_rule_4_unmapped_taxonomy_tag(self, parser, mock_storage):
        """Rule 4: Unrecognized itemCode must be quarantined with UNMAPPED_TAXONOMY_TAG."""
        raw_json = json.dumps(
            {
                "ticker": "FPT",
                "report_type": "incomestatement",
                "period": "yearly",
                "data": [
                    {
                        "itemCode": 999999,  # Non-existent item code
                        "numericValue": 12345678,
                        "fiscalDate": "2023-12-31",
                    }
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file("test-bronze", "test.json")
        assert len(clean) == 0
        assert len(quarantine) == 1
        assert "UNMAPPED_TAXONOMY_TAG" in quarantine[0]["quarantine_reason"]
        assert "999999" in quarantine[0]["quarantine_reason"]

    def test_rule_5_duplicate_observation_deduplication(self, parser, mock_storage):
        """Rule 5: Deduplicate multiple observations for same period, keeping latest modifiedDate."""
        raw_json = json.dumps(
            {
                "ticker": "FPT",
                "report_type": "incomestatement",
                "period": "yearly",
                "data": [
                    {
                        "itemCode": 21001,  # revenues
                        "numericValue": 50000000000,
                        "fiscalDate": "2023-12-31",
                        "modifiedDate": "2024-01-15 00:00:00",
                    },
                    {
                        "itemCode": 21001,  # duplicate with newer amended date
                        "numericValue": 52617900827385,
                        "fiscalDate": "2023-12-31",
                        "modifiedDate": "2024-03-20 00:00:00",
                    },
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file("test-bronze", "test.json")
        assert len(clean) == 1
        assert len(quarantine) == 1
        # Preserved observation must have newer modified date and value
        assert clean[0]["value"] == Decimal("52617900827385.00")
        assert "DUPLICATE_OBSERVATION" in quarantine[0]["quarantine_reason"]

    def test_rule_6_strict_decimal_precision_and_null_preservation(self, parser, mock_storage):
        """Rule 6: Ensure DECIMAL(28, 2) fixed point quantization and preserve NULL != 0."""
        # Case A: Valid numeric string/float quantized cleanly
        assert parser.quantize_decimal("12345.678") == Decimal("12345.68")
        assert parser.quantize_decimal(50000) == Decimal("50000.00")

        # Case B: Unparseable NaN / Infinity quarantined
        raw_json = json.dumps(
            {
                "ticker": "FPT",
                "report_type": "incomestatement",
                "period": "yearly",
                "data": [
                    {
                        "itemCode": 21001,
                        "numericValue": "NaN_CORRUPTED",
                        "fiscalDate": "2023-12-31",
                    }
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file("test-bronze", "test.json")
        assert len(clean) == 0
        assert len(quarantine) == 1
        assert "INVALID_DECIMAL_VALUE" in quarantine[0]["quarantine_reason"]

        # Case C: NULL is preserved and not converted to 0
        assert parser.quantize_decimal(None) is None


class TestVNFactsParserTransformAndParquet:
    """Validate standard transformation and Parquet serialization."""

    def test_clean_transformation_and_fact_keys(self, parser, mock_storage):
        """Ensure clean records generate deterministic SHA-256 keys and correct schema."""
        raw_json = json.dumps(
            {
                "ticker": "VNM",
                "report_type": "balancesheet",
                "period": "yearly",
                "data": [
                    {
                        "itemCode": 12700,  # total_assets
                        "numericValue": 52673371104460.0,
                        "fiscalDate": "2023-12-31",
                        "modifiedDate": "2024-03-10 00:00:00",
                    }
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file("test-bronze", "test.json")
        assert len(clean) == 1
        assert len(quarantine) == 0

        obs = clean[0]
        assert obs["ticker"] == "VNM"
        assert obs["tag"] == "total_assets"
        assert obs["period_type"] == "instant"
        assert obs["duration_scope"] == "instant"
        assert obs["start_date"] is None
        assert obs["end_date"] == date(2023, 12, 31)
        assert obs["fiscal_year"] == 2023
        assert obs["fiscal_period"] == "FY"
        assert obs["market"] == "VN"
        assert obs["value"] == Decimal("52673371104460.00")
        assert len(obs["fact_key"]) == 64
        assert len(obs["observation_key"]) == 64

    def test_serialize_and_upload_parquet(self, parser, mock_storage):
        """Ensure valid PyArrow Parquet buffers are generated and uploaded to MinIO."""
        clean_rows = [
            {
                "fact_key": "a" * 64,
                "observation_key": "b" * 64,
                "cik": "VN:FPT",
                "ticker": "FPT",
                "taxonomy": "vas",
                "tag": "revenues",
                "value": Decimal("52617900827385.00"),
                "unit": "VND",
                "period_type": "duration",
                "start_date": date(2023, 1, 1),
                "end_date": date(2023, 12, 31),
                "fiscal_year": 2023,
                "fiscal_period": "FY",
                "duration_days": 364,
                "duration_scope": "annual_12m",
                "form": "BCTC-FY",
                "filed_date": date(2024, 3, 20),
                "accession_number": "c" * 18,
                "is_amendment": False,
                "source_file": "s3://test/fpt.json",
                "market": "VN",
            }
        ]
        quarantine_rows = [
            {
                "ticker": "FPT",
                "tag": "EMPTY_PAYLOAD",
                "raw_payload": "{}",
                "quarantine_reason": "EMPTY_PAYLOAD_ANOMALY",
                "source_file": "s3://test/fpt_empty.json",
                "quarantined_at": datetime.now(UTC),
            }
        ]

        stats = parser.serialize_and_upload(clean_rows, quarantine_rows)
        assert stats["clean_records"] == 1
        assert stats["quarantine_records"] == 1
        assert stats["facts_parquet_bytes"] > 0
        assert stats["quarantine_parquet_bytes"] > 0
        assert mock_storage.upload_bytes.call_count == 2

    def test_vas_bank_excludes_corporate_gross_profit(self, parser, mock_storage):
        """VAS-BANK issuers must never map Circular 200 gross_profit or revenues codes."""
        raw_json = json.dumps(
            {
                "ticker": "VCB",
                "report_type": "incomestatement",
                "period": "yearly",
                "data": [
                    {"itemCode": 23100, "numericValue": 8383018000000, "fiscalDate": "2024-12-31"},
                    {"itemCode": 21001, "numericValue": 999999, "fiscalDate": "2024-12-31"},
                    {"itemCode": 421701, "numericValue": 68578496000000, "fiscalDate": "2024-12-31"},
                ],
            }
        ).encode("utf-8")
        mock_storage.download_bytes.return_value = raw_json

        clean, quarantine = parser.parse_bronze_file("test-bronze", "vn_raw/financials/VCB_incomestatement_yearly.json")
        tags = {row["tag"] for row in clean}
        assert "gross_profit" not in tags
        assert "revenues" not in tags
        assert "total_operating_income" in tags
        assert len(quarantine) == 0
