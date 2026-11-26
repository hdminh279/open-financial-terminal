"""Pipeline to ingest SEC company submissions and XBRL facts into MinIO Bronze Layer.

Extracts data from SEC EDGAR API for target companies defined in configs/companies.yaml,
hashes the payload with SHA-256 for data provenance, and stores it in MinIO `sec-bronze`.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from backend.core.rate_limiter import RateLimiter
from backend.core.storage.s3_storage import S3Storage
from pipelines.ingestion.sec_client import SECClient

logger = logging.getLogger(__name__)

# SEC EDGAR base endpoints
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_10}.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_10}.json"


class FactsIngestionPipeline:
    """Orchestrates fetching and storing raw company facts and submissions."""

    def __init__(
        self,
        sec_client: SECClient,
        s3_storage: S3Storage,
        bronze_bucket: str = "sec-bronze",
    ) -> None:
        """Initialize the pipeline with required clients and destination bucket."""
        self.sec_client = sec_client
        self.s3_storage = s3_storage
        self.bronze_bucket = bronze_bucket

    @staticmethod
    def _compute_sha256(data: bytes) -> str:
        """Compute hex-encoded SHA-256 digest of payload."""
        return hashlib.sha256(data).hexdigest()

    def _fetch_and_store_endpoint(
        self,
        url: str,
        s3_raw_key: str,
        s3_meta_key: str,
        ticker: str,
        cik_10: str,
        force: bool = False,
    ) -> bool:
        """Fetch a single SEC endpoint and upload raw JSON + companion metadata to MinIO Bronze.

        If not forced and the object already exists in S3, compares the remote payload SHA-256
        against recorded metadata. If identical, skips uploading to avoid redundant I/O.
        """
        logger.info("Fetching: %s", url)
        response = self.sec_client.get(url)
        payload_bytes = response.content

        # Compute audit provenance
        sha256_hash = self._compute_sha256(payload_bytes)
        fetched_at = datetime.now(UTC).isoformat()

        # Idempotency check: if not forced and object exists, verify if content actually changed
        if not force and self.s3_storage.object_exists(self.bronze_bucket, s3_raw_key):
            try:
                existing_meta = self.s3_storage.get_metadata(self.bronze_bucket, s3_raw_key)
                existing_sha256 = existing_meta.get("metadata", {}).get("sha256")
                if existing_sha256 and existing_sha256.lower() == sha256_hash.lower():
                    logger.info(
                        "Payload for %s is unchanged (SHA256: %s). Skipping S3 upload (idempotent).",
                        s3_raw_key,
                        sha256_hash[:8],
                    )
                    return False
            except Exception as exc:
                logger.warning(
                    "Could not read metadata for s3://%s/%s: %s. Overwriting.", self.bronze_bucket, s3_raw_key, exc
                )

        metadata_dict: dict[str, Any] = {
            "ticker": ticker,
            "cik": cik_10,
            "source_url": url,
            "fetched_at": fetched_at,
            "sha256": sha256_hash,
            "byte_size": len(payload_bytes),
        }

        # 1. Upload raw payload
        self.s3_storage.upload_bytes(
            bucket=self.bronze_bucket,
            key=s3_raw_key,
            data=payload_bytes,
            content_type="application/json",
            metadata={
                "ticker": ticker,
                "cik": cik_10,
                "sha256": sha256_hash,
                "fetched_at": fetched_at,
            },
        )

        # 2. Upload companion metadata file for data lake auditability
        meta_bytes = json.dumps(metadata_dict, indent=2).encode("utf-8")
        self.s3_storage.upload_bytes(
            bucket=self.bronze_bucket,
            key=s3_meta_key,
            data=meta_bytes,
            content_type="application/json",
        )

        logger.info("Successfully ingested s3://%s/%s (SHA256: %s)", self.bronze_bucket, s3_raw_key, sha256_hash[:8])
        return True

    def ingest_company(self, ticker: str, cik: str, force: bool = False) -> dict[str, bool]:
        """Fetch and store submissions and XBRL facts for a given company."""
        cik_10 = str(cik).strip().zfill(10)
        ticker_clean = ticker.strip().upper()

        logger.info("Starting ingestion for %s (CIK: %s)", ticker_clean, cik_10)

        # 1. Submissions JSON
        sub_url = SEC_SUBMISSIONS_URL.format(cik_10=cik_10)
        sub_raw_key = f"raw/submissions/{ticker_clean}_{cik_10}.json"
        sub_meta_key = f"raw/submissions/{ticker_clean}_{cik_10}_metadata.json"
        sub_status = self._fetch_and_store_endpoint(
            url=sub_url,
            s3_raw_key=sub_raw_key,
            s3_meta_key=sub_meta_key,
            ticker=ticker_clean,
            cik_10=cik_10,
            force=force,
        )

        # 2. Company Facts JSON
        facts_url = SEC_COMPANY_FACTS_URL.format(cik_10=cik_10)
        facts_raw_key = f"raw/company_facts/{ticker_clean}_{cik_10}.json"
        facts_meta_key = f"raw/company_facts/{ticker_clean}_{cik_10}_metadata.json"
        facts_status = self._fetch_and_store_endpoint(
            url=facts_url,
            s3_raw_key=facts_raw_key,
            s3_meta_key=facts_meta_key,
            ticker=ticker_clean,
            cik_10=cik_10,
            force=force,
        )

        return {"submissions_ingested": sub_status, "facts_ingested": facts_status}

    def run_from_config(self, config_path: str = "configs/companies.yaml", force: bool = False) -> None:
        """Run the ingestion pipeline for all companies defined in the configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        companies: list[dict[str, Any]] = cfg.get("companies", [])
        if not companies:
            logger.warning("No companies configured in %s", config_path)
            return

        # Ensure MinIO buckets are ready
        self.s3_storage.ensure_buckets_exist()

        logger.info("Found %d companies to ingest from %s", len(companies), config_path)
        for item in companies:
            ticker = item["ticker"]
            cik = item["cik"]
            self.ingest_company(ticker=ticker, cik=cik, force=force)


def main() -> None:
    """CLI execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv()

    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError(
            "Missing SEC_USER_AGENT environment variable. "
            "Please configure it in .env (e.g. 'CorporateFilingIntelligence user@example.com')."
        )

    rate_limiter = RateLimiter(requests_per_second=10.0)
    with SECClient(user_agent=user_agent, rate_limiter=rate_limiter) as sec_client:
        s3_storage = S3Storage()
        pipeline = FactsIngestionPipeline(sec_client=sec_client, s3_storage=s3_storage)
        pipeline.run_from_config("configs/companies.yaml")


if __name__ == "__main__":
    main()

"""
docker compose run --rm app uv run python -m pipelines.ingestion.fetch_facts
"""
