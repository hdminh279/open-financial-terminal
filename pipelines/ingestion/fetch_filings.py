"""Pipeline to ingest raw SEC Form 10-K and Form 10-Q HTML filings into MinIO Bronze Layer.

Leverages FilingResolver to discover the most recent filings for target companies,
downloads the primary HTML document via SECClient, computes SHA-256 integrity hashes,
and uploads both the raw document and companion metadata to `sec-bronze`.
"""

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from backend.core.rate_limiter import RateLimiter
from backend.core.storage.s3_storage import S3Storage
from pipelines.ingestion.filing_resolver import (
    DEFAULT_TARGET_FORMS,
    FilingResolver,
    ResolvedFiling,
)
from pipelines.ingestion.sec_client import SECClient

logger = logging.getLogger(__name__)


class FilingsIngestionPipeline:
    """Orchestrates fetching and storing raw HTML filings in MinIO Bronze."""

    def __init__(
        self,
        sec_client: SECClient,
        s3_storage: S3Storage,
        resolver: FilingResolver | None = None,
        bronze_bucket: str = "sec-bronze",
    ) -> None:
        """Initialize the pipeline with required clients and destination bucket."""
        self.sec_client = sec_client
        self.s3_storage = s3_storage
        self.bronze_bucket = bronze_bucket
        self.resolver = resolver or FilingResolver(
            s3_storage=s3_storage,
            bronze_bucket=bronze_bucket,
            sec_client=sec_client,
        )

    @staticmethod
    def _compute_sha256(data: bytes) -> str:
        """Compute hex-encoded SHA-256 digest of payload."""
        return hashlib.sha256(data).hexdigest()

    def ingest_filing(self, filing: ResolvedFiling, force: bool = False) -> bool:
        """Download and store a single filing document if not already cached.

        Args:
            filing: Resolved filing descriptor containing source URL and destination key.
            force: If True, overwrite existing objects in S3.

        Returns:
            True if downloaded and stored, False if skipped due to idempotency.
        """
        # Determine companion metadata key
        base_path = filing.s3_key.rsplit(".", 1)[0]
        meta_key = f"{base_path}_metadata.json"

        # Pre-fetch check: skip if already validly stored in S3
        if not force and self.s3_storage.object_exists(self.bronze_bucket, filing.s3_key):
            try:
                meta = self.s3_storage.get_metadata(self.bronze_bucket, filing.s3_key)
                content_len = meta.get("content_length")
                has_sha256 = bool(meta.get("metadata", {}).get("sha256"))
                if isinstance(content_len, (int, float)) and content_len > 0 and has_sha256:
                    logger.info(
                        "Filing already exists with valid SHA-256 at s3://%s/%s. Skipping.",
                        self.bronze_bucket,
                        filing.s3_key,
                    )
                    return False
            except Exception as exc:
                logger.warning(
                    "Filing s3://%s/%s failed metadata check (%s). Re-downloading.",
                    self.bronze_bucket,
                    filing.s3_key,
                    exc,
                )

        logger.info(
            "Fetching %s [%s] (%s) from %s",
            filing.ticker,
            filing.form,
            filing.report_date,
            filing.sec_url,
        )

        response = self.sec_client.get(filing.sec_url)
        payload_bytes = response.content

        if not payload_bytes:
            logger.error("Empty content received for %s from %s", filing.accession_number, filing.sec_url)
            raise ValueError(f"Empty payload received from SEC for {filing.sec_url}")

        sha256_hash = self._compute_sha256(payload_bytes)
        fetched_at = datetime.now(UTC).isoformat()

        metadata_dict: dict[str, Any] = {
            "ticker": filing.ticker,
            "cik": filing.cik,
            "form": filing.form,
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "accession_number": filing.accession_number,
            "primary_document": filing.primary_document,
            "source_url": filing.sec_url,
            "fetched_at": fetched_at,
            "sha256": sha256_hash,
            "byte_size": len(payload_bytes),
        }

        # 1. Upload raw HTML document
        self.s3_storage.upload_bytes(
            bucket=self.bronze_bucket,
            key=filing.s3_key,
            data=payload_bytes,
            content_type="text/html",
            metadata={
                "ticker": filing.ticker,
                "cik": filing.cik,
                "form": filing.form,
                "sha256": sha256_hash,
                "fetched_at": fetched_at,
            },
        )

        # 2. Upload companion metadata file for lakehouse auditability
        meta_bytes = json.dumps(metadata_dict, indent=2).encode("utf-8")
        self.s3_storage.upload_bytes(
            bucket=self.bronze_bucket,
            key=meta_key,
            data=meta_bytes,
            content_type="application/json",
        )

        logger.info(
            "Successfully stored s3://%s/%s (Size: %.2f KB, SHA256: %s)",
            self.bronze_bucket,
            filing.s3_key,
            len(payload_bytes) / 1024,
            sha256_hash[:8],
        )
        return True

    def ingest_company_filings(
        self,
        ticker: str,
        cik: str,
        forms: Sequence[str] = DEFAULT_TARGET_FORMS,
        limit_per_form: int | None = None,
        min_year: int | None = 2009,
        force: bool = False,
    ) -> dict[str, int]:
        """Resolve and ingest target filings for a specific company."""
        logger.info("Resolving target filings for %s (CIK: %s)", ticker, cik)
        filings = self.resolver.resolve_from_s3(
            ticker=ticker,
            cik=cik,
            target_forms=forms,
            limit_per_form=limit_per_form,
            min_year=min_year,
        )

        stats = {"downloaded": 0, "skipped": 0, "total": len(filings)}
        for filing in filings:
            was_downloaded = self.ingest_filing(filing=filing, force=force)
            if was_downloaded:
                stats["downloaded"] += 1
            else:
                stats["skipped"] += 1

        logger.info("Completed %s: %d downloaded, %d skipped", ticker, stats["downloaded"], stats["skipped"])
        return stats

    def run_from_config(
        self,
        config_path: str = "configs/companies.yaml",
        forms: Sequence[str] = DEFAULT_TARGET_FORMS,
        limit_per_form: int | None = None,
        min_year: int | None = 2009,
        force: bool = False,
    ) -> None:
        """Run the filing ingestion pipeline for all companies defined in the config file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        companies: list[dict[str, Any]] = cfg.get("companies", [])
        if not companies:
            logger.warning("No companies configured in %s", config_path)
            return

        self.s3_storage.ensure_buckets_exist()

        limit_desc = f"{limit_per_form}/form" if limit_per_form else "all available"
        logger.info(
            "Starting filings ingestion for %d companies (limit=%s, min_year=%s)", len(companies), limit_desc, min_year
        )
        for item in companies:
            ticker = item["ticker"]
            cik = item["cik"]
            self.ingest_company_filings(
                ticker=ticker,
                cik=cik,
                forms=forms,
                limit_per_form=limit_per_form,
                min_year=min_year,
                force=force,
            )


def main() -> None:
    """CLI execution entrypoint for fetching raw filings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv()

    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError("Missing SEC_USER_AGENT environment variable in .env.")

    rate_limiter = RateLimiter(requests_per_second=10.0)
    with SECClient(user_agent=user_agent, rate_limiter=rate_limiter) as sec_client:
        s3_storage = S3Storage()
        pipeline = FilingsIngestionPipeline(sec_client=sec_client, s3_storage=s3_storage)
        pipeline.run_from_config(
            config_path="configs/companies.yaml",
            forms=("10-K", "10-Q"),
            limit_per_form=4,
            force=False,
        )


if __name__ == "__main__":
    main()


"""
docker compose run --rm app uv run python -m pipelines.ingestion.fetch_filings
"""
