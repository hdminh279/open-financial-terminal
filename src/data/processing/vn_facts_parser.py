"""Vietnamese Financial Facts Parser and Silver Data Quality Gate.

Processes raw Bronze financial statements from MinIO (`sec-bronze/vn_raw/financials/`),
evaluates 6 rigorous Data Quality Gate rules, canonicalizes VAS line items into
standard metrics (DECIMAL(28, 2)), separates quarantine anomalies, and serializes
clean observations to Parquet in MinIO (`sec-silver/facts/vn_facts.parquet`).
"""

import hashlib
import io
import json
import logging
import os
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)

# Standard allowed reporting units
ALLOWED_UNITS: set[str] = {"VND", "VND/shares", "%"}

# Corporate Circular 200 item codes that must never be mapped onto VAS-BANK (Model 101/102)
BANK_DISALLOWED_ITEM_CODES: set[int] = {
    21000,  # gross_revenues
    21001,  # revenues
    22000,  # operating_income
    22070,  # gross_profit_alt
    22100,  # cost_of_goods_sold
    22500,  # selling_expenses
    22510,  # administrative_expenses
    23100,  # gross_profit
    11000,  # current_assets
    12000,  # non_current_assets
    13100,  # current_liabilities
    13200,  # non_current_liabilities
}

# VNDirect Finfo itemCode mapping to canonical standard metric names
# Format: itemCode: (canonical_metric, period_type, default_unit)
ITEM_CODE_MAP: dict[int, tuple[str, str, str]] = {
    # -------------------------------------------------------------------------
    # Corporate Income Statement (Model 2 / Circular 200)
    # -------------------------------------------------------------------------
    21001: ("revenues", "duration", "VND"),
    21000: ("gross_revenues", "duration", "VND"),
    22100: ("cost_of_goods_sold", "duration", "VND"),
    23100: ("gross_profit", "duration", "VND"),
    22070: ("gross_profit_alt", "duration", "VND"),
    21500: ("financial_income", "duration", "VND"),
    22200: ("financial_expenses", "duration", "VND"),
    22210: ("interest_expense", "duration", "VND"),
    22500: ("selling_expenses", "duration", "VND"),
    22510: ("administrative_expenses", "duration", "VND"),
    22000: ("operating_income", "duration", "VND"),
    23800: ("profit_before_tax", "duration", "VND"),
    24000: ("profit_before_tax_alt", "duration", "VND"),
    23000: ("net_income", "duration", "VND"),
    23003: ("net_income_total", "duration", "VND"),
    23001: ("basic_eps", "duration", "VND/shares"),
    700087: ("basic_eps_alt", "duration", "VND/shares"),
    # -------------------------------------------------------------------------
    # Corporate Balance Sheet (Model 1 / Circular 200)
    # -------------------------------------------------------------------------
    11000: ("current_assets", "instant", "VND"),
    11100: ("cash_and_cash_equivalents", "instant", "VND"),
    12000: ("non_current_assets", "instant", "VND"),
    12700: ("total_assets", "instant", "VND"),
    13000: ("total_liabilities", "instant", "VND"),
    13100: ("current_liabilities", "instant", "VND"),
    13200: ("non_current_liabilities", "instant", "VND"),
    14000: ("stockholders_equity_total", "instant", "VND"),
    14110: ("contributed_capital", "instant", "VND"),
    14180: ("retained_earnings", "instant", "VND"),
    14400: ("total_liabilities_and_equity", "instant", "VND"),
    # -------------------------------------------------------------------------
    # Corporate Cash Flow (Model 3 / Circular 200)
    # -------------------------------------------------------------------------
    32000: ("operating_cash_flow", "duration", "VND"),
    33000: ("investing_cash_flow", "duration", "VND"),
    34000: ("financing_cash_flow", "duration", "VND"),
    35000: ("net_change_in_cash", "duration", "VND"),
    # -------------------------------------------------------------------------
    # Banking Specific (Models 101 & 102 / Circular 49 & Decision 16)
    # -------------------------------------------------------------------------
    421100: ("interest_income", "duration", "VND"),
    421900: ("interest_expense_bank", "duration", "VND"),
    422100: ("net_interest_income", "duration", "VND"),
    421701: ("total_operating_income", "duration", "VND"),
    88888: ("profit_before_tax_bank", "duration", "VND"),
    412100: ("customer_loans", "instant", "VND"),
    413300: ("customer_deposits", "instant", "VND"),
}


class VNFactsParser:
    """Standardizes Vietnamese financial data and enforces Silver Data Quality Gates."""

    FACTS_SCHEMA = pa.schema(
        [
            ("fact_key", pa.string()),
            ("observation_key", pa.string()),
            ("cik", pa.string()),
            ("ticker", pa.string()),
            ("taxonomy", pa.string()),
            ("tag", pa.string()),
            ("value", pa.decimal128(28, 2)),
            ("unit", pa.string()),
            ("period_type", pa.string()),
            ("start_date", pa.date32()),
            ("end_date", pa.date32()),
            ("fiscal_year", pa.int32()),
            ("fiscal_period", pa.string()),
            ("duration_days", pa.int32()),
            ("duration_scope", pa.string()),
            ("form", pa.string()),
            ("filed_date", pa.date32()),
            ("accession_number", pa.string()),
            ("is_amendment", pa.bool_()),
            ("source_file", pa.string()),
            ("market", pa.string()),
        ]
    )

    QUARANTINE_SCHEMA = pa.schema(
        [
            ("ticker", pa.string()),
            ("tag", pa.string()),
            ("raw_payload", pa.string()),
            ("quarantine_reason", pa.string()),
            ("source_file", pa.string()),
            ("quarantined_at", pa.timestamp("us")),
        ]
    )

    def __init__(
        self,
        s3_storage: S3Storage | None = None,
        bronze_bucket: str = "sec-bronze",
        silver_bucket: str = "sec-silver",
        companies_vn_path: str = "configs/companies_vn.yaml",
    ) -> None:
        """Initialize parser with storage connection and metadata registry."""
        self.s3 = s3_storage or S3Storage()
        self.bronze_bucket = bronze_bucket
        self.silver_bucket = silver_bucket
        self.companies_vn_path = companies_vn_path
        self.company_standards = self._load_company_standards(companies_vn_path)

    @staticmethod
    def _load_company_standards(config_path: str) -> dict[str, str]:
        """Load accounting standards per ticker from companies_vn.yaml."""
        if not os.path.exists(config_path):
            logger.warning("Config path %s not found. Defaulting to VAS.", config_path)
            return {}
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        mapping = {}
        for comp in data.get("companies", []):
            ticker = comp.get("ticker", "").strip().upper()
            standard = comp.get("accounting_standard", "VAS")
            if ticker:
                mapping[ticker] = standard
        return mapping

    @staticmethod
    def compute_fact_key(
        ticker: str,
        tag: str,
        unit: str,
        period_type: str,
        start_str: str | None,
        end_str: str,
    ) -> str:
        """Generate canonical SHA-256 hash identifying the underlying economic fact."""
        start_key = start_str if (start_str and period_type == "duration") else "null"
        payload = f"VN:{ticker}/vas/{tag}/{unit}/{period_type}/{start_key}/{end_str}/consolidated"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_observation_key(fact_key: str, accession_number: str) -> str:
        """Generate unique SHA-256 hash for a specific filed observation."""
        payload = f"{fact_key}/{accession_number}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def parse_period_dates(
        fiscal_date_str: str,
        report_type: str,
        period: str,
    ) -> tuple[date | None, date, int, str]:
        """Derive start_date, end_date, fiscal_year, and fiscal_period from fiscalDate."""
        end_date = datetime.strptime(fiscal_date_str[:10], "%Y-%m-%d").date()
        year = end_date.year

        if period.lower() == "yearly" or report_type.upper() == "ANNUAL":
            fiscal_period = "FY"
            start_date = date(year, 1, 1)
        else:
            month = end_date.month
            if month <= 3:
                fiscal_period = "Q1"
                start_date = date(year, 1, 1)
            elif month <= 6:
                fiscal_period = "Q2"
                start_date = date(year, 4, 1)
            elif month <= 9:
                fiscal_period = "Q3"
                start_date = date(year, 7, 1)
            else:
                fiscal_period = "Q4"
                start_date = date(year, 10, 1)

        return start_date, end_date, year, fiscal_period

    @staticmethod
    def quantize_decimal(val: Any) -> Decimal | None:
        """Strictly validate and quantize numerical values to DECIMAL(28, 2)."""
        if val is None:
            return None
        try:
            dec = Decimal(str(val))
            if not dec.is_finite():
                return None
            return dec.quantize(Decimal("0.01"))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def validate_quarterly_continuity(
        self,
        ticker: str,
        records: list[dict[str, Any]],
        source_file: str,
    ) -> list[dict[str, Any]]:
        """Gate Rule 2: Detect missing quarterly reporting gaps within fiscal years."""
        anomalies: list[dict[str, Any]] = []
        year_quarters: dict[int, set[str]] = {}

        for rec in records:
            if rec.get("reportType") == "QUARTER" and rec.get("fiscalDate"):
                try:
                    _, _, year, fp = self.parse_period_dates(rec["fiscalDate"], "QUARTER", "quarterly")
                    year_quarters.setdefault(year, set()).add(fp)
                except Exception:
                    continue

        current_year = datetime.now(UTC).year
        for year, quarters in year_quarters.items():
            # Check completed historical years (prior to current year)
            if year < current_year:
                expected = {"Q1", "Q2", "Q3", "Q4"}
                missing = expected - quarters
                # If there's an internal gap (e.g. Q1 and Q3 exist but Q2 is missing)
                if missing and len(quarters) >= 2:
                    anomalies.append(
                        {
                            "ticker": ticker,
                            "tag": "PERIOD_CONTINUITY",
                            "raw_payload": json.dumps(
                                {"fiscal_year": year, "existing": sorted(quarters), "missing": sorted(missing)}
                            ),
                            "quarantine_reason": f"MISSING_PERIOD_GAP: FY{year} missing {sorted(missing)}",
                            "source_file": source_file,
                            "quarantined_at": datetime.now(UTC),
                        }
                    )
        return anomalies

    def parse_bronze_file(
        self,
        bucket: str,
        key: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse raw Bronze JSON file and evaluate the 6 Silver Quality Gate rules."""
        clean_rows: list[dict[str, Any]] = []
        quarantine_rows: list[dict[str, Any]] = []

        try:
            raw_bytes = self.s3.download_bytes(bucket, key)
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            quarantine_rows.append(
                {
                    "ticker": key.split("/")[-1].split("_")[0],
                    "tag": "CORRUPT_PAYLOAD",
                    "raw_payload": str(exc),
                    "quarantine_reason": f"PAYLOAD_READ_FAILURE: {exc}",
                    "source_file": key,
                    "quarantined_at": datetime.now(UTC),
                }
            )
            return clean_rows, quarantine_rows

        ticker = payload.get("ticker", "").strip().upper()
        report_type = payload.get("report_type", "")
        period = payload.get("period", "")
        data_records = payload.get("data", [])

        # GATE RULE 1: Empty Payload Anomaly
        if not data_records:
            quarantine_rows.append(
                {
                    "ticker": ticker,
                    "tag": "EMPTY_PAYLOAD",
                    "raw_payload": json.dumps({"key": key, "record_count": 0}),
                    "quarantine_reason": "EMPTY_PAYLOAD_ANOMALY: Upstream returned 0 records",
                    "source_file": key,
                    "quarantined_at": datetime.now(UTC),
                }
            )
            return clean_rows, quarantine_rows

        # GATE RULE 2: Missing Period Gap Audit
        gap_anomalies = self.validate_quarterly_continuity(ticker, data_records, key)
        quarantine_rows.extend(gap_anomalies)

        # Deduplication tracker for GATE RULE 5: (ticker, tag, fiscal_year, fiscal_period, period_type)
        seen_observations: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        accounting_standard = self.company_standards.get(ticker, "VAS")

        for record in data_records:
            raw_item_code = record.get("itemCode")
            raw_val = record.get("numericValue")
            fiscal_date_str = record.get("fiscalDate")

            if raw_item_code is None or not fiscal_date_str:
                continue

            try:
                item_code_int = int(float(raw_item_code))
            except (ValueError, TypeError):
                quarantine_rows.append(
                    {
                        "ticker": ticker,
                        "tag": "INVALID_ITEM_CODE",
                        "raw_payload": json.dumps(record),
                        "quarantine_reason": "UNMAPPED_TAXONOMY_TAG: itemCode not numeric",
                        "source_file": key,
                        "quarantined_at": datetime.now(UTC),
                    }
                )
                continue

            # Ignore Circular 200 corporate line items for commercial banks (VAS-BANK)
            if accounting_standard == "VAS-BANK" and item_code_int in BANK_DISALLOWED_ITEM_CODES:
                continue

            # GATE RULE 4: Taxonomy Whitelist Mapping
            if item_code_int not in ITEM_CODE_MAP:
                # Retain raw record in quarantine if unmapped
                quarantine_rows.append(
                    {
                        "ticker": ticker,
                        "tag": f"VAS_CODE_{item_code_int}",
                        "raw_payload": json.dumps(record),
                        "quarantine_reason": f"UNMAPPED_TAXONOMY_TAG: Code {item_code_int} not in VAS registry",
                        "source_file": key,
                        "quarantined_at": datetime.now(UTC),
                    }
                )
                continue

            canonical_tag, period_type, unit = ITEM_CODE_MAP[item_code_int]

            # GATE RULE 3: Currency & Unit Verification
            if unit not in ALLOWED_UNITS:
                quarantine_rows.append(
                    {
                        "ticker": ticker,
                        "tag": canonical_tag,
                        "raw_payload": json.dumps(record),
                        "quarantine_reason": f"UNSUPPORTED_CURRENCY: Unit '{unit}' not in {ALLOWED_UNITS}",
                        "source_file": key,
                        "quarantined_at": datetime.now(UTC),
                    }
                )
                continue

            # GATE RULE 6: Strict DECIMAL(28, 2) and Numeric Validation
            dec_val = self.quantize_decimal(raw_val)
            if dec_val is None and raw_val is not None:
                quarantine_rows.append(
                    {
                        "ticker": ticker,
                        "tag": canonical_tag,
                        "raw_payload": json.dumps(record),
                        "quarantine_reason": f"INVALID_DECIMAL_VALUE: Unparseable value '{raw_val}'",
                        "source_file": key,
                        "quarantined_at": datetime.now(UTC),
                    }
                )
                continue

            # If the raw report has no data (missing metric), keep NULL; do not fabricate zero
            if dec_val is None:
                continue

            try:
                start_date, end_date, fiscal_year, fiscal_period = self.parse_period_dates(
                    fiscal_date_str, report_type, period
                )
            except Exception as exc:
                quarantine_rows.append(
                    {
                        "ticker": ticker,
                        "tag": canonical_tag,
                        "raw_payload": json.dumps(record),
                        "quarantine_reason": f"INVALID_DATE_FORMAT: {exc}",
                        "source_file": key,
                        "quarantined_at": datetime.now(UTC),
                    }
                )
                continue

            # Calculate duration scope
            duration_days: int | None = None
            if period_type == "duration" and start_date:
                duration_days = (end_date - start_date).days
                scope = "discrete_quarter" if fiscal_period != "FY" else "annual_12m"
            else:
                scope = "instant"
                start_date = None

            start_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_str = end_date.strftime("%Y-%m-%d")

            fact_key = self.compute_fact_key(
                ticker=ticker,
                tag=canonical_tag,
                unit=unit,
                period_type=period_type,
                start_str=start_str,
                end_str=end_str,
            )

            modified_date_str = record.get("modifiedDate") or record.get("createdDate") or fiscal_date_str
            accession_number = hashlib.sha256(
                f"{ticker}_{fiscal_date_str}_{report_type}_{period}_{item_code_int}".encode()
            ).hexdigest()[:18]
            observation_key = self.compute_observation_key(fact_key, accession_number)

            filed_dt = datetime.strptime(modified_date_str[:10], "%Y-%m-%d").date()

            candidate_row = {
                "fact_key": fact_key,
                "observation_key": observation_key,
                "cik": f"VN:{ticker}",
                "ticker": ticker,
                "taxonomy": self.company_standards.get(ticker, "VAS").lower(),
                "tag": canonical_tag,
                "value": dec_val,
                "unit": unit,
                "period_type": period_type,
                "start_date": start_date,
                "end_date": end_date,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "duration_days": duration_days,
                "duration_scope": scope,
                "form": f"BCTC-{fiscal_period}",
                "filed_date": filed_dt,
                "accession_number": accession_number,
                "is_amendment": False,
                "source_file": key,
                "market": "VN",
            }

            # GATE RULE 5: Deduplication Check
            dedup_key = (ticker, canonical_tag, fiscal_year, fiscal_period, period_type)
            if dedup_key in seen_observations:
                existing = seen_observations[dedup_key]
                # Compare filed_date / modifiedDate; keep the latest and quarantine the duplicate
                if candidate_row["filed_date"] >= existing["filed_date"]:
                    quarantine_rows.append(
                        {
                            "ticker": ticker,
                            "tag": canonical_tag,
                            "raw_payload": json.dumps({"superseded_val": str(existing["value"])}),
                            "quarantine_reason": f"DUPLICATE_OBSERVATION: Superseded by newer record on {candidate_row['filed_date']}",
                            "source_file": key,
                            "quarantined_at": datetime.now(UTC),
                        }
                    )
                    seen_observations[dedup_key] = candidate_row
                else:
                    quarantine_rows.append(
                        {
                            "ticker": ticker,
                            "tag": canonical_tag,
                            "raw_payload": json.dumps({"dropped_val": str(candidate_row["value"])}),
                            "quarantine_reason": f"DUPLICATE_OBSERVATION: Older than existing record from {existing['filed_date']}",
                            "source_file": key,
                            "quarantined_at": datetime.now(UTC),
                        }
                    )
            else:
                seen_observations[dedup_key] = candidate_row

        clean_rows = list(seen_observations.values())
        return clean_rows, quarantine_rows

    def serialize_and_upload(
        self,
        clean_rows: list[dict[str, Any]],
        quarantine_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Convert clean and quarantined records to Parquet and persist to MinIO Silver."""
        stats = {
            "clean_records": len(clean_rows),
            "quarantine_records": len(quarantine_rows),
            "facts_parquet_bytes": 0,
            "quarantine_parquet_bytes": 0,
        }

        # 1. Upload Clean Facts Parquet
        if clean_rows:
            table = pa.Table.from_pylist(clean_rows, schema=self.FACTS_SCHEMA)
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="snappy")
            data_bytes = buf.getvalue()
            facts_key = "facts/vn_facts.parquet"
            self.s3.upload_bytes(self.silver_bucket, facts_key, data_bytes, content_type="application/octet-stream")
            stats["facts_parquet_bytes"] = len(data_bytes)
            logger.info("Uploaded %d clean observations to s3://%s/%s", len(clean_rows), self.silver_bucket, facts_key)

        # 2. Upload Quarantine Anomalies Parquet
        if quarantine_rows:
            q_table = pa.Table.from_pylist(quarantine_rows, schema=self.QUARANTINE_SCHEMA)
            q_buf = io.BytesIO()
            pq.write_table(q_table, q_buf, compression="snappy")
            q_bytes = q_buf.getvalue()
            quarantine_key = "quarantine/vn_anomalies.parquet"
            self.s3.upload_bytes(self.silver_bucket, quarantine_key, q_bytes, content_type="application/octet-stream")
            stats["quarantine_parquet_bytes"] = len(q_bytes)
            logger.info(
                "Uploaded %d anomaly records to s3://%s/%s", len(quarantine_rows), self.silver_bucket, quarantine_key
            )

        return stats

    def run(
        self,
        prefix: str = "vn_raw/financials/",
        target_tickers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute parsing and quality gates across all Bronze statements, persisting to Silver."""
        all_objects = self.s3.list_objects(self.bronze_bucket, prefix=prefix)
        statement_keys = [k for k in all_objects if k.endswith(".json") and not k.endswith("_metadata.json")]
        if target_tickers:
            upper_targets = {t.upper() for t in target_tickers}
            statement_keys = [k for k in statement_keys if k.split("/")[-1].split("_")[0].upper() in upper_targets]

        logger.info("Found %d Bronze VN statement packages to parse.", len(statement_keys))

        all_clean_rows: list[dict[str, Any]] = []
        all_quarantine_rows: list[dict[str, Any]] = []

        for idx, key in enumerate(statement_keys, 1):
            clean_rows, q_rows = self.parse_bronze_file(self.bronze_bucket, key)
            all_clean_rows.extend(clean_rows)
            all_quarantine_rows.extend(q_rows)
            if idx % 20 == 0 or idx == len(statement_keys):
                logger.info(
                    "Parsed %d/%d files: accumulated %d clean observations, %d quarantine anomalies",
                    idx,
                    len(statement_keys),
                    len(all_clean_rows),
                    len(all_quarantine_rows),
                )

        # Global deduplication across all files before upload
        dedup_map: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        for row in all_clean_rows:
            key = (row["ticker"], row["tag"], row["fiscal_year"], row["fiscal_period"], row["period_type"])
            if key in dedup_map:
                existing = dedup_map[key]
                if row["filed_date"] >= existing["filed_date"]:
                    all_quarantine_rows.append(
                        {
                            "ticker": row["ticker"],
                            "tag": row["tag"],
                            "raw_payload": json.dumps({"superseded_val": str(existing["value"])}),
                            "quarantine_reason": f"DUPLICATE_OBSERVATION: Superseded across files by {row['filed_date']}",
                            "source_file": row["source_file"],
                            "quarantined_at": datetime.now(UTC),
                        }
                    )
                    dedup_map[key] = row
            else:
                dedup_map[key] = row

        final_clean = list(dedup_map.values())
        upload_stats = self.serialize_and_upload(final_clean, all_quarantine_rows)
        upload_stats["files_processed"] = len(statement_keys)
        return upload_stats
