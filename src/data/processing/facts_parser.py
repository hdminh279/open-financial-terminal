"""XBRL Facts Parser and Standardizer for Corporate Filing Intelligence.

Extracts raw XBRL company facts from MinIO sec-bronze, verifies facts against
a taxonomy whitelist, calculates canonical fact_key and observation_key hashes,
strictly handles fixed-point Decimal monetary quantities, classifies period durations,
and serializes clean observations to Parquet in MinIO sec-silver.
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


class FactsParser:
    """Parser and standardizer for transforming Bronze XBRL facts into Silver Parquet observations."""

    CANONICAL_DIMENSIONS = "consolidated"
    DEFAULT_TAXONOMY_PATH = "configs/xbrl_taxonomy_map.yaml"
    DEFAULT_COMPANIES_PATH = "configs/companies.yaml"
    ALLOWED_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A")

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
        ]
    )

    QUARANTINE_SCHEMA = pa.schema(
        [
            ("cik", pa.string()),
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
        taxonomy_path: str = DEFAULT_TAXONOMY_PATH,
        companies_path: str = DEFAULT_COMPANIES_PATH,
        bronze_bucket: str = "sec-bronze",
        silver_bucket: str = "sec-silver",
    ) -> None:
        """Initialize FactsParser with S3 client, taxonomy rules, and company mappings."""
        self.s3 = s3_storage or S3Storage()
        self.bronze_bucket = bronze_bucket
        self.silver_bucket = silver_bucket
        self.taxonomy_path = taxonomy_path
        self.companies_path = companies_path
        self.taxonomy_config = self.load_taxonomy_config(taxonomy_path)
        self.allowed_units = self.extract_allowed_units(self.taxonomy_config)
        self.whitelist_tags = self.extract_whitelist_tags(self.taxonomy_config)
        self.name_to_ticker, self.cik_to_ticker = self.load_company_mappings(companies_path)

    @staticmethod
    def load_company_mappings(
        companies_path: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Load company entity name and CIK to ticker mappings from YAML."""
        name_to_ticker: dict[str, str] = {}
        cik_to_ticker: dict[str, str] = {}
        if not os.path.exists(companies_path):
            logger.warning("Companies config file not found at %s", companies_path)
            return name_to_ticker, cik_to_ticker

        with open(companies_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for comp in data.get("companies", []):
            ticker = comp.get("ticker")
            name = comp.get("name")
            cik = comp.get("cik")
            if ticker and name:
                name_to_ticker[name.strip().lower()] = ticker
                # Handle standard naming variations (e.g. 'MICROSOFT CORP' vs 'Microsoft Corporation')
                lower_name = name.lower()
                if "corporation" in lower_name:
                    name_to_ticker[lower_name.replace("corporation", "corp").strip()] = ticker
                elif "corp" in lower_name:
                    name_to_ticker[lower_name.replace("corp", "corporation").strip()] = ticker
            if ticker and cik:
                cik_to_ticker[str(cik).zfill(10)] = ticker

        logger.info("Loaded %d company mappings from %s", len(name_to_ticker), companies_path)
        return name_to_ticker, cik_to_ticker

    def resolve_ticker(
        self,
        cik: str | None = None,
        entity_name: str | None = None,
        explicit_ticker: str | None = None,
    ) -> str | None:
        """Resolve trading ticker from explicit argument, CIK, or company entity name."""
        if explicit_ticker and explicit_ticker.strip():
            return explicit_ticker.strip().upper()

        if cik and cik in self.cik_to_ticker:
            return self.cik_to_ticker[cik]

        if entity_name:
            norm_name = entity_name.strip().lower()
            if norm_name in self.name_to_ticker:
                return self.name_to_ticker[norm_name]
            cleaned_name = norm_name.rstrip(".").strip()
            if cleaned_name in self.name_to_ticker:
                return self.name_to_ticker[cleaned_name]

        return None

    @staticmethod
    def load_taxonomy_config(taxonomy_path: str) -> dict[str, Any]:
        """Load and validate the YAML taxonomy mapping and validation configuration.

        Fails fast if the file is missing or contains invalid YAML content.
        """
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Taxonomy configuration not found: {taxonomy_path}")
        with open(taxonomy_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML content in {taxonomy_path}: expected dictionary root")
        return config

    @staticmethod
    def extract_allowed_units(config: dict[str, Any]) -> set[str]:
        """Extract and validate allowed measurement units from YAML constraints."""
        constraints = config.get("constraints", {})
        units = constraints.get("allowed_units")
        if not isinstance(units, list) or not units:
            raise ValueError("Configuration 'constraints.allowed_units' must be a non-empty list")
        for u in units:
            if not isinstance(u, str) or not u.strip():
                raise ValueError("All entries in 'constraints.allowed_units' must be non-empty strings")
        return set(units)

    @staticmethod
    def extract_whitelist_tags(config: dict[str, Any]) -> set[str]:
        """Extract set of unique allowed GAAP tags across all metric rules in the YAML config."""
        tags: set[str] = set()
        for metric_info in config.get("metrics", {}).values():
            if isinstance(metric_info, dict):
                for rule in metric_info.get("rules", []):
                    if isinstance(rule, dict):
                        tag = rule.get("tag")
                        if tag and isinstance(tag, str):
                            tags.add(tag.strip())

        logger.info("Loaded %d whitelist tags from taxonomy config", len(tags))
        return tags

    @classmethod
    def load_whitelist_tags(cls, taxonomy_path: str) -> set[str]:
        """Backward-compatible helper to extract whitelist tags from a file path."""
        cfg = cls.load_taxonomy_config(taxonomy_path)
        return cls.extract_whitelist_tags(cfg)

    @staticmethod
    def compute_fact_key(
        cik: str,
        taxonomy: str,
        tag: str,
        unit: str,
        period_type: str,
        start_str: str | None,
        end_str: str,
        dimensions: str = CANONICAL_DIMENSIONS,
    ) -> str:
        """Compute invariant SHA-256 hash identifying the underlying economic measurement."""
        normalized_start = start_str if (start_str and period_type == "duration") else "null"
        payload = f"{cik}/{taxonomy}/{tag}/{unit}/{period_type}/{normalized_start}/{end_str}/{dimensions}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_observation_key(fact_key: str, accession_number: str) -> str:
        """Compute unique SHA-256 hash identifying a specific filing observation of a fact."""
        payload = f"{fact_key}/{accession_number}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def classify_duration(start_date: date | None, end_date: date) -> tuple[int | None, str]:
        """Classify the accounting period duration scope without altering official SEC metadata."""
        if start_date is None:
            return None, "instant"

        duration_days = (end_date - start_date).days
        if duration_days < 0:
            return duration_days, "invalid"

        if 80 <= duration_days <= 105:
            scope = "discrete_quarter"
        elif 160 <= duration_days <= 200:
            scope = "cumulative_6m"
        elif 250 <= duration_days <= 295:
            scope = "cumulative_9m"
        elif 350 <= duration_days <= 375:
            scope = "annual_12m"
        else:
            scope = "unclassified"

        return duration_days, scope

    @classmethod
    def validate_and_quantize_decimal(cls, val: Any) -> Decimal | None:
        """Convert and validate numerical value to Decimal with scale 2.

        Guarantees zero floating-point intermediate representation.
        Returns None if value exceeds precision or has unparseable precision.
        """
        try:
            if isinstance(val, Decimal):
                dec = val
            elif isinstance(val, (int, str)):
                dec = Decimal(str(val))
            else:
                # Disallow raw floats to prevent precision corruption
                return None

            # Validate max precision and scale
            exp = dec.as_tuple().exponent
            if not isinstance(exp, int) or exp < -2:
                # Fractional cents beyond 2 decimal places are out of scope for USD whole metrics
                return None
            if len(dec.as_tuple().digits) > 28:
                return None

            return dec.quantize(Decimal("0.01"))
        except (InvalidOperation, TypeError):
            return None

    def parse_facts_payload(
        self,
        data: dict[str, Any],
        source_file: str,
        whitelist_tags: set[str] | None = None,
        ticker: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse raw company facts JSON payload into valid observations and quarantine anomalies."""
        tags_to_parse = whitelist_tags or self.whitelist_tags
        valid_observations: dict[str, dict[str, Any]] = {}
        quarantine_records: list[dict[str, Any]] = []

        cik_raw = data.get("cik")
        cik = str(cik_raw).zfill(10) if (cik_raw is not None and str(cik_raw).strip() != "") else None
        entity_name = data.get("entityName")
        resolved_ticker = self.resolve_ticker(
            cik=cik,
            entity_name=entity_name,
            explicit_ticker=ticker,
        )

        facts_dict = data.get("facts", {})
        for taxonomy, tag_group in facts_dict.items():
            if not isinstance(tag_group, dict):
                continue

            for tag_name, tag_obj in tag_group.items():
                if tag_name not in tags_to_parse or not isinstance(tag_obj, dict):
                    continue

                for unit_key, items in tag_obj.get("units", {}).items():
                    # Filter for permitted units defined dynamically in taxonomy constraints
                    if unit_key not in self.allowed_units or not isinstance(items, list):
                        continue

                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        val = item.get("val")
                        end_str = item.get("end")
                        start_str = item.get("start")
                        accn = item.get("accn")
                        filed_str = item.get("filed")
                        form = item.get("form")
                        fy = item.get("fy")
                        fp = item.get("fp")

                        # Validate minimum required fields for Fact & Observation Identity
                        if not cik or not tag_name or not end_str or val is None or not accn or not filed_str:
                            quarantine_records.append(
                                {
                                    "cik": cik,
                                    "ticker": resolved_ticker,
                                    "tag": tag_name,
                                    "raw_payload": json.dumps(
                                        {k: str(v) if isinstance(v, Decimal) else v for k, v in item.items()}
                                    ),
                                    "quarantine_reason": "INCOMPLETE_FACT_KEY",
                                    "source_file": source_file,
                                    "quarantined_at": datetime.now(UTC).replace(tzinfo=None),
                                }
                            )
                            continue

                        # Filter strictly to periodic reporting forms
                        if form not in self.ALLOWED_FORMS:
                            continue

                        # Validate Decimal
                        decimal_val = self.validate_and_quantize_decimal(val)
                        if decimal_val is None:
                            quarantine_records.append(
                                {
                                    "cik": cik,
                                    "ticker": resolved_ticker,
                                    "tag": tag_name,
                                    "raw_payload": json.dumps(
                                        {k: str(v) if isinstance(v, Decimal) else v for k, v in item.items()}
                                    ),
                                    "quarantine_reason": "UNSUPPORTED_DECIMAL_PRECISION",
                                    "source_file": source_file,
                                    "quarantined_at": datetime.now(UTC).replace(tzinfo=None),
                                }
                            )
                            continue

                        # Parse End Date and Filed Date
                        try:
                            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                            filed_date = datetime.strptime(filed_str, "%Y-%m-%d").date()
                        except ValueError:
                            quarantine_records.append(
                                {
                                    "cik": cik,
                                    "ticker": resolved_ticker,
                                    "tag": tag_name,
                                    "raw_payload": json.dumps(
                                        {k: str(v) if isinstance(v, Decimal) else v for k, v in item.items()}
                                    ),
                                    "quarantine_reason": "INVALID_DATE_FORMAT",
                                    "source_file": source_file,
                                    "quarantined_at": datetime.now(UTC).replace(tzinfo=None),
                                }
                            )
                            continue

                        # Period Type and Start Date
                        period_type = "instant" if not start_str else "duration"
                        start_date = None

                        if period_type == "duration":
                            try:
                                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                            except ValueError:
                                quarantine_records.append(
                                    {
                                        "cik": cik,
                                        "ticker": resolved_ticker,
                                        "tag": tag_name,
                                        "raw_payload": json.dumps(
                                            {k: str(v) if isinstance(v, Decimal) else v for k, v in item.items()}
                                        ),
                                        "quarantine_reason": "INVALID_DATE_FORMAT",
                                        "source_file": source_file,
                                        "quarantined_at": datetime.now(UTC).replace(tzinfo=None),
                                    }
                                )
                                continue

                            if start_date > end_date:
                                quarantine_records.append(
                                    {
                                        "cik": cik,
                                        "ticker": resolved_ticker,
                                        "tag": tag_name,
                                        "raw_payload": json.dumps(
                                            {k: str(v) if isinstance(v, Decimal) else v for k, v in item.items()}
                                        ),
                                        "quarantine_reason": "INVALID_DATE_RANGE",
                                        "source_file": source_file,
                                        "quarantined_at": datetime.now(UTC).replace(tzinfo=None),
                                    }
                                )
                                continue

                        duration_days, duration_scope = self.classify_duration(start_date, end_date)

                        # Compute keys
                        fact_key = self.compute_fact_key(
                            cik=cik,
                            taxonomy=taxonomy,
                            tag=tag_name,
                            unit=unit_key,
                            period_type=period_type,
                            start_str=start_str,
                            end_str=end_str,
                            dimensions=self.CANONICAL_DIMENSIONS,
                        )
                        obs_key = self.compute_observation_key(fact_key, accn)

                        fiscal_year = int(fy) if fy is not None else None
                        fiscal_period = fp.upper() if fp is not None else None
                        is_amendment = bool(form and "/A" in form)

                        obs = {
                            "fact_key": fact_key,
                            "observation_key": obs_key,
                            "cik": cik,
                            "ticker": resolved_ticker,
                            "taxonomy": taxonomy,
                            "tag": tag_name,
                            "value": decimal_val,
                            "unit": unit_key,
                            "period_type": period_type,
                            "start_date": start_date,
                            "end_date": end_date,
                            "fiscal_year": fiscal_year,
                            "fiscal_period": fiscal_period,
                            "duration_days": duration_days,
                            "duration_scope": duration_scope,
                            "form": form,
                            "filed_date": filed_date,
                            "accession_number": accn,
                            "is_amendment": is_amendment,
                            "source_file": source_file,
                        }
                        valid_observations[obs_key] = obs

        return list(valid_observations.values()), quarantine_records

    def build_facts_table(self, observations: list[dict[str, Any]]) -> pa.Table:
        """Construct strongly typed PyArrow Table conforming to FACTS_SCHEMA."""
        columns = {field.name: [obs.get(field.name) for obs in observations] for field in self.FACTS_SCHEMA}
        return pa.Table.from_pydict(columns, schema=self.FACTS_SCHEMA)

    def build_quarantine_table(self, anomalies: list[dict[str, Any]]) -> pa.Table:
        """Construct strongly typed PyArrow Table conforming to QUARANTINE_SCHEMA."""
        columns = {field.name: [rec.get(field.name) for rec in anomalies] for field in self.QUARANTINE_SCHEMA}
        return pa.Table.from_pydict(columns, schema=self.QUARANTINE_SCHEMA)

    @staticmethod
    def table_to_parquet_bytes(table: pa.Table) -> bytes:
        """Serialize PyArrow Table to compressed Parquet binary bytes."""
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        return buffer.getvalue()

    def run(
        self,
        bronze_prefix: str = "raw/company_facts/",
        facts_output_key: str = "facts/facts.parquet",
        quarantine_output_key: str = "quarantine/anomalies.parquet",
    ) -> dict[str, int]:
        """Execute complete Bronze to Silver ETL parsing and persistence pipeline."""
        logger.info("Executing FactsParser pipeline from %s to %s...", self.bronze_bucket, self.silver_bucket)
        self.s3.ensure_buckets_exist([self.silver_bucket])

        fact_keys = [
            k
            for k in self.s3.list_objects(self.bronze_bucket, prefix=bronze_prefix)
            if k.endswith(".json") and not k.endswith("_metadata.json")
        ]

        all_observations: dict[str, dict[str, Any]] = {}
        all_quarantine: list[dict[str, Any]] = []

        for key in fact_keys:
            raw_bytes = self.s3.download_bytes(self.bronze_bucket, key)
            payload = json.loads(
                raw_bytes.decode("utf-8"),
                parse_float=Decimal,
                parse_int=Decimal,
            )
            ticker_from_key = key.split("/")[-1].split("_")[0]

            source_file = f"s3://{self.bronze_bucket}/{key}"
            obs_list, q_list = self.parse_facts_payload(payload, source_file=source_file, ticker=ticker_from_key)

            for obs in obs_list:
                all_observations[obs["observation_key"]] = obs

            for q in q_list:
                all_quarantine.append(q)

        obs_records = list(all_observations.values())
        logger.info(
            "Parsed %d total valid observations and %d quarantined records",
            len(obs_records),
            len(all_quarantine),
        )

        # Upload facts parquet
        facts_table = self.build_facts_table(obs_records)
        facts_bytes = self.table_to_parquet_bytes(facts_table)
        self.s3.upload_bytes(
            bucket=self.silver_bucket,
            key=facts_output_key,
            data=facts_bytes,
            content_type="application/vnd.apache.parquet",
            metadata={"record_count": str(len(obs_records)), "pipeline": "FactsParser"},
        )

        # Upload quarantine parquet if any anomalies exist
        if all_quarantine:
            q_table = self.build_quarantine_table(all_quarantine)
            q_bytes = self.table_to_parquet_bytes(q_table)
            self.s3.upload_bytes(
                bucket=self.silver_bucket,
                key=quarantine_output_key,
                data=q_bytes,
                content_type="application/vnd.apache.parquet",
                metadata={"record_count": str(len(all_quarantine)), "pipeline": "FactsParser"},
            )

        return {
            "valid_observations": len(obs_records),
            "quarantined_records": len(all_quarantine),
        }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = FactsParser()
    stats = parser.run()
    print(f"\nExecution finished successfully: {stats}")


"""
docker compose run --rm app uv run python -m src.cli process --facts
"""
