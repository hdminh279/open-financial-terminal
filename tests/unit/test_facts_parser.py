"""Unit tests for XBRL FactsParser module.

Verifies:
- Canonical fact_key and observation_key generation
- Strict Decimal precision and fixed-point scale validation
- Accounting period duration scope classification
- Data quarantine triage (incomplete, invalid date range, precision violations)
- Retention of valid-unclassified observations (missing fy/fp)
- PyArrow Parquet table serialization and schema conformity
- Idempotent execution
"""

from datetime import date, datetime
from decimal import Decimal
import json
from typing import Any, List
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.processing.facts_parser import FactsParser


class TestFactsParserKeys:
    """Test suite for fact_key and observation_key canonical hashing."""

    def test_fact_key_deterministic(self) -> None:
        """Verify identical accounting inputs yield identical fact_key hashes."""
        key1 = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Revenues",
            unit="USD",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
            dimensions="consolidated",
        )
        key2 = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Revenues",
            unit="USD",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
            dimensions="consolidated",
        )
        assert key1 == key2
        assert len(key1) == 64

    def test_fact_key_sensitivity(self) -> None:
        """Verify modifying any dimension changes the fact_key hash."""
        base = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Revenues",
            unit="USD",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
        )
        different_cik = FactsParser.compute_fact_key(
            cik="0000789019",
            taxonomy="us-gaap",
            tag="Revenues",
            unit="USD",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
        )
        different_tag = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="NetIncomeLoss",
            unit="USD",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
        )
        different_unit = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Revenues",
            unit="EUR",
            period_type="duration",
            start_str="2023-01-01",
            end_str="2023-03-31",
        )
        assert base != different_cik
        assert base != different_tag
        assert base != different_unit

    def test_fact_key_instant_normalization(self) -> None:
        """Verify instant periods normalize start_date to 'null'."""
        instant_key1 = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Assets",
            unit="USD",
            period_type="instant",
            start_str=None,
            end_str="2023-09-30",
        )
        instant_key2 = FactsParser.compute_fact_key(
            cik="0000320193",
            taxonomy="us-gaap",
            tag="Assets",
            unit="USD",
            period_type="instant",
            start_str="",
            end_str="2023-09-30",
        )
        assert instant_key1 == instant_key2

    def test_observation_key_uniqueness(self) -> None:
        """Verify multiple filing accession numbers produce distinct observation keys for the same fact."""
        fact_key = "d04b32b85a3c945b37e8bbd12e8c2576a88b560df0a424a13247bb411cf11718"
        obs1 = FactsParser.compute_observation_key(fact_key, "0000320193-23-000106")
        obs2 = FactsParser.compute_observation_key(fact_key, "0000320193-24-000100")
        assert obs1 != obs2
        assert len(obs1) == 64


class TestFactsParserDecimalPrecision:
    """Test suite for Decimal precision and fixed-point validation."""

    def test_decimal_from_int(self) -> None:
        """Verify integer values quantize exactly to 2 decimal places."""
        res = FactsParser.validate_and_quantize_decimal(5000000)
        assert res == Decimal("5000000.00")
        assert isinstance(res, Decimal)

    def test_decimal_from_decimal_object(self) -> None:
        """Verify native Decimal objects preserve precision and scale."""
        res = FactsParser.validate_and_quantize_decimal(Decimal("12345.67"))
        assert res == Decimal("12345.67")

    def test_decimal_disallow_float(self) -> None:
        """Verify raw float values are rejected to prevent IEEE-754 precision bugs."""
        res = FactsParser.validate_and_quantize_decimal(12345.67)
        assert res is None

    def test_decimal_reject_fractional_cents(self) -> None:
        """Verify fractional cents beyond 2 decimal places are rejected for USD whole amounts."""
        res = FactsParser.validate_and_quantize_decimal(Decimal("100.1234"))
        assert res is None

    def test_decimal_reject_overflow(self) -> None:
        """Verify digits exceeding 28 are rejected to prevent decimal128 overflow."""
        huge_val = Decimal("9" * 30)
        res = FactsParser.validate_and_quantize_decimal(huge_val)
        assert res is None


class TestFactsParserPeriodClassification:
    """Test suite for accounting period duration scope categorization."""

    def test_classify_instant(self) -> None:
        """Verify start_date=None classifies as instant with duration_days=None."""
        days, scope = FactsParser.classify_duration(None, date(2023, 9, 30))
        assert days is None
        assert scope == "instant"

    def test_classify_discrete_quarter(self) -> None:
        """Verify ~90-day intervals classify as discrete_quarter."""
        days, scope = FactsParser.classify_duration(date(2023, 1, 1), date(2023, 4, 1))
        assert days == 90
        assert scope == "discrete_quarter"

    def test_classify_cumulative_6m(self) -> None:
        """Verify ~180-day intervals classify as cumulative_6m."""
        days, scope = FactsParser.classify_duration(date(2023, 1, 1), date(2023, 7, 1))
        assert days == 181
        assert scope == "cumulative_6m"

    def test_classify_cumulative_9m(self) -> None:
        """Verify ~270-day intervals classify as cumulative_9m."""
        days, scope = FactsParser.classify_duration(date(2023, 1, 1), date(2023, 9, 30))
        assert days == 272
        assert scope == "cumulative_9m"

    def test_classify_annual_12m(self) -> None:
        """Verify ~365-day intervals classify as annual_12m."""
        days, scope = FactsParser.classify_duration(date(2022, 10, 1), date(2023, 9, 30))
        assert days == 364
        assert scope == "annual_12m"

    def test_classify_unclassified(self) -> None:
        """Verify irregular durations classify as unclassified."""
        days, scope = FactsParser.classify_duration(date(2023, 1, 1), date(2023, 2, 15))
        assert days == 45
        assert scope == "unclassified"

    def test_classify_invalid_date_range(self) -> None:
        """Verify start_date > end_date marks as invalid."""
        days, scope = FactsParser.classify_duration(date(2023, 5, 1), date(2023, 1, 1))
        assert days < 0
        assert scope == "invalid"


class TestFactsParserQuarantineAndRetention:
    """Test suite for quarantine segregation and valid-unclassified fact retention."""

    @pytest.fixture
    def parser(self, mocker: Any) -> FactsParser:
        """Construct FactsParser with mocked S3 storage."""
        mock_s3 = mocker.MagicMock()
        return FactsParser(s3_storage=mock_s3, taxonomy_path="configs/xbrl_taxonomy_map.yaml")

    def test_incomplete_fact_key_quarantined(self, parser: FactsParser) -> None:
        """Verify missing essential fact key fields routes record to quarantine."""
        bad_payload = {
            "cik": "0000320193",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                # Missing 'end' and 'accn'
                                {"val": 100000, "form": "10-K", "filed": "2023-11-01"}
                            ]
                        }
                    }
                }
            },
        }
        obs, q = parser.parse_facts_payload(bad_payload, source_file="s3://test.json")
        assert len(obs) == 0
        assert len(q) == 1
        assert q[0]["quarantine_reason"] == "INCOMPLETE_FACT_KEY"

    def test_invalid_date_range_quarantined(self, parser: FactsParser) -> None:
        """Verify date range paradox routes record to quarantine."""
        inverted_dates_payload = {
            "cik": "0000320193",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100000,
                                    "start": "2023-12-31",
                                    "end": "2023-01-01",  # end before start
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-11-01",
                                    "form": "10-K",
                                }
                            ]
                        }
                    }
                }
            },
        }
        obs, q = parser.parse_facts_payload(inverted_dates_payload, source_file="s3://test.json")
        assert len(obs) == 0
        assert len(q) == 1
        assert q[0]["quarantine_reason"] == "INVALID_DATE_RANGE"

    def test_missing_fiscal_metadata_retained_in_silver(self, parser: FactsParser) -> None:
        """Verify observations lacking fy or fp are NOT quarantined, but kept with null metadata."""
        unclassified_payload = {
            "cik": "0000320193",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 500000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    # fy and fp omitted
                                }
                            ]
                        }
                    }
                }
            },
        }
        obs, q = parser.parse_facts_payload(unclassified_payload, source_file="s3://test.json")
        assert len(q) == 0
        assert len(obs) == 1
        assert obs[0]["fiscal_year"] is None
        assert obs[0]["fiscal_period"] is None
        assert obs[0]["duration_scope"] == "discrete_quarter"
        assert obs[0]["value"] == Decimal("500000.00")


class TestFactsParserPyArrowAndIdempotency:
    """Test suite for Parquet serialization and idempotent pipeline execution."""

    def test_pyarrow_facts_table_schema(self, mocker: Any) -> None:
        """Verify built facts table strictly adheres to PyArrow schema types."""
        mock_s3 = mocker.MagicMock()
        parser = FactsParser(s3_storage=mock_s3)

        sample_observations = [
            {
                "fact_key": "a" * 64,
                "observation_key": "b" * 64,
                "cik": "0000320193",
                "ticker": "AAPL",
                "taxonomy": "us-gaap",
                "tag": "Revenues",
                "value": Decimal("123456789.50"),
                "unit": "USD",
                "period_type": "duration",
                "start_date": date(2023, 1, 1),
                "end_date": date(2023, 3, 31),
                "fiscal_year": 2023,
                "fiscal_period": "Q1",
                "duration_days": 89,
                "duration_scope": "discrete_quarter",
                "form": "10-Q",
                "filed_date": date(2023, 4, 15),
                "accession_number": "0000320193-23-000001",
                "is_amendment": False,
                "source_file": "s3://sec-bronze/raw/company_facts/AAPL_0000320193.json",
            }
        ]

        table = parser.build_facts_table(sample_observations)
        assert table.schema == FactsParser.FACTS_SCHEMA
        assert table.num_rows == 1

        # Roundtrip Parquet serialization
        parquet_bytes = parser.table_to_parquet_bytes(table)
        reader = pq.ParquetFile(pa.BufferReader(parquet_bytes))
        roundtrip_table = reader.read()
        assert roundtrip_table.num_rows == 1
        assert roundtrip_table.column("value")[0].as_py() == Decimal("123456789.50")

    def test_run_idempotency(self, mocker: Any) -> None:
        """Verify executing run() twice on identical Bronze inputs yields identical Silver bytes."""
        mock_s3 = mocker.MagicMock()
        sample_json = {
            "cik": "0000320193",
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ]
                        }
                    }
                }
            },
        }

        mock_s3.list_objects.return_value = ["raw/company_facts/AAPL_0000320193.json"]
        mock_s3.download_bytes.return_value = json.dumps(sample_json).encode("utf-8")

        uploaded_buffers: List[bytes] = []

        def capture_upload(bucket: str, key: str, data: bytes, **kwargs: Any) -> bool:
            if "facts.parquet" in key:
                uploaded_buffers.append(data)
            return True

        mock_s3.upload_bytes.side_effect = capture_upload

        parser = FactsParser(s3_storage=mock_s3)
        res1 = parser.run()
        res2 = parser.run()

        assert res1["valid_observations"] == 1
        assert res2["valid_observations"] == 1
        assert len(uploaded_buffers) == 2
        # Identical bytes
        assert uploaded_buffers[0] == uploaded_buffers[1]


class TestFactsParserTickerResolution:
    """Test suite for mapping entityName and CIK to standardized ticker."""

    @pytest.fixture
    def parser(self, mocker: Any) -> FactsParser:
        """Construct FactsParser with mocked S3 storage."""
        mock_s3 = mocker.MagicMock()
        return FactsParser(s3_storage=mock_s3)

    def test_resolve_ticker_from_explicit_argument(self, parser: FactsParser) -> None:
        """Verify explicit ticker overrides any lookup."""
        ticker = parser.resolve_ticker(
            cik="0000320193", entity_name="Apple Inc.", explicit_ticker="aapl"
        )
        assert ticker == "AAPL"

    def test_resolve_ticker_from_cik(self, parser: FactsParser) -> None:
        """Verify CIK matches correct ticker."""
        assert parser.resolve_ticker(cik="0000320193") == "AAPL"
        assert parser.resolve_ticker(cik="0000789019") == "MSFT"
        assert parser.resolve_ticker(cik="0001652044") == "GOOGL"

    def test_resolve_ticker_from_entity_name(self, parser: FactsParser) -> None:
        """Verify company entity name maps correctly to ticker."""
        assert parser.resolve_ticker(entity_name="Apple Inc.") == "AAPL"
        assert parser.resolve_ticker(entity_name="MICROSOFT CORP") == "MSFT"
        assert parser.resolve_ticker(entity_name="Microsoft Corporation") == "MSFT"
        assert parser.resolve_ticker(entity_name="Alphabet Inc.") == "GOOGL"

    def test_parse_facts_payload_populates_mapped_ticker(self, parser: FactsParser) -> None:
        """Verify parse_facts_payload maps entityName to ticker, not company name."""
        payload = {
            "cik": "0000320193",
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ]
                        }
                    }
                }
            },
        }
        obs, _ = parser.parse_facts_payload(payload, source_file="s3://test.json")
        assert len(obs) == 1
        # Crucial check: ticker must be 'AAPL', never 'Apple Inc.'
        assert obs[0]["ticker"] == "AAPL"


class TestFactsParserTaxonomyAndUnits:
    """Test suite for taxonomy configuration loading, fail-fast validations, and unit filtering."""

    def test_missing_taxonomy_raises_filenotfound(self) -> None:
        """Verify parser raises FileNotFoundError when taxonomy file is missing."""
        with pytest.raises(FileNotFoundError, match="Taxonomy configuration not found"):
            FactsParser.load_taxonomy_config("non_existent_taxonomy_file.yaml")

    def test_invalid_allowed_units_raises_valueerror(self) -> None:
        """Verify parser raises ValueError if allowed_units is missing or invalid."""
        with pytest.raises(ValueError, match="constraints.allowed_units"):
            FactsParser.extract_allowed_units({"constraints": {"allowed_units": []}})
        with pytest.raises(ValueError, match="constraints.allowed_units"):
            FactsParser.extract_allowed_units({"constraints": {}})
        with pytest.raises(ValueError, match="non-empty strings"):
            FactsParser.extract_allowed_units({"constraints": {"allowed_units": [""]}})

    def test_unit_filtering_accepts_usd_and_usd_per_share(self) -> None:
        """Verify parser retains both USD and USD/shares, while dropping disallowed units."""
        parser = FactsParser()
        payload = {
            "cik": "0000320193",
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ],
                            "CAD": [
                                {
                                    "val": 95000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ],
                        }
                    },
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {
                                    "val": Decimal("1.52"),
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ],
                            "shares": [
                                {
                                    "val": 16000000000,
                                    "start": "2023-01-01",
                                    "end": "2023-03-31",
                                    "accn": "0000320193-23-000001",
                                    "filed": "2023-04-15",
                                    "form": "10-Q",
                                    "fy": 2023,
                                    "fp": "Q1",
                                }
                            ],
                        }
                    },
                }
            },
        }
        obs, _ = parser.parse_facts_payload(payload, source_file="s3://test.json")
        # Exactly 2 observations accepted: Revenues (USD) and EarningsPerShareDiluted (USD/shares)
        assert len(obs) == 2
        units = {o["unit"] for o in obs}
        assert units == {"USD", "USD/shares"}
        eps_obs = next(o for o in obs if o["tag"] == "EarningsPerShareDiluted")
        assert eps_obs["value"] == Decimal("1.52")
        assert eps_obs["unit"] == "USD/shares"


