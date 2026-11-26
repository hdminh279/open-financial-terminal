"""VN30 Financial Data Ingestion Pipeline for MinIO Bronze Layer.

Fetches audited annual and quarterly financial statements (Income Statement,
Balance Sheet, Cash Flow) for target Vietnamese companies defined in
configs/companies_vn.yaml, validates payload integrity, computes SHA-256
provenance hashes, and idempotently uploads them to MinIO `sec-bronze`.
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx
import yaml
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.rate_limiter import RateLimiter
from backend.core.storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)

REPORT_TYPES = ["incomestatement", "balancesheet", "cashflow"]
PERIOD_TYPES = ["yearly", "quarterly"]

# Dynamic modelType registry by accounting standard (zero hardcoding of tickers)
# Governed by Circular 200/2014/TT-BTC, Circular 49/2014/TT-NHNN, Circular 334/2016/TT-BTC, and Circular 199/2014/TT-BTC
ACCOUNTING_STANDARD_MODELS: dict[str, dict[str, int]] = {
    "VAS": {
        "balancesheet": 1,
        "incomestatement": 2,
        "cashflow": 3,
    },
    "VAS-BANK": {
        "balancesheet": 101,
        "incomestatement": 102,
        "cashflow": 103,
    },
    "VAS-SECURITIES": {
        "balancesheet": 89,
        "incomestatement": 90,
        "cashflow": 91,
    },
    "VAS-INSURANCE": {
        "balancesheet": 411,
        "incomestatement": 412,
        "cashflow": 413,
    },
}


def _is_retryable_error(exc: BaseException) -> bool:
    """Determine whether an error is transient and safe to retry.

    Retries on network-level connectivity errors, timeouts, rate limiting (429),
    and transient upstream server failures (500, 502, 503, 504).
    Fails fast without retry on client errors (400, 401, 403, 404).
    """
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


class VNFinancialClient:
    """Resilient HTTP client for fetching Vietnamese stock financial statements."""

    BASE_URL = "https://api-finfo.vndirect.com.vn/v4/financial_statements"
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    }

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        timeout_seconds: float = 15.0,
        min_year: int = 2019,
    ) -> None:
        """Initialize HTTP client with rate limiting and retry configuration."""
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_second=3.0)
        self.min_year = min_year
        self.client = httpx.Client(
            headers=self.DEFAULT_HEADERS,
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        reraise=True,
    )
    def fetch_financial_report(
        self,
        ticker: str,
        report_type: str,
        period: str = "yearly",
        accounting_standard: str = "VAS",
    ) -> dict[str, Any]:
        """Fetch raw financial statement JSON for a specific ticker and report type.

        Dynamically routes to the compliant modelType based on the company's accounting
        standard without hardcoding individual tickers.
        Retryable errors (429, 5xx, timeouts) bubble up directly to the tenacity
        retry decorator. Terminal client errors (4xx) fail fast.
        """
        self.rate_limiter.acquire()
        ticker_clean = ticker.strip().upper()
        report_clean = report_type.strip().lower()

        # Dynamic modelType lookup from accounting standard registry
        std_clean = accounting_standard.strip().upper() if accounting_standard else "VAS"
        standard_models = ACCOUNTING_STANDARD_MODELS.get(std_clean, ACCOUNTING_STANDARD_MODELS["VAS"])
        model_type = standard_models.get(report_clean, 2)

        report_type_param = "ANNUAL" if period.lower() == "yearly" else "QUARTER"
        query_filter = (
            f"code:{ticker_clean}~reportType:{report_type_param}~"
            f"modelType:{model_type}~fiscalDate:gte:{self.min_year}-01-01"
        )
        params = {
            "q": query_filter,
            "size": 3000,
            "sort": "fiscalDate:desc",
        }

        logger.debug("Requesting VN financial report: %s with params: %s", self.BASE_URL, params)

        # Do NOT catch retryable errors internally so Tenacity can intercept them
        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        body = response.json()
        raw_data = body.get("data", []) if isinstance(body, dict) else body

        return {
            "ticker": ticker_clean,
            "report_type": report_clean,
            "period": period,
            "source_url": str(response.url),
            "data": raw_data,
        }

    def close(self) -> None:
        """Close underlying HTTP connection pool."""
        self.client.close()

    def __enter__(self) -> "VNFinancialClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class VNIngestionPipeline:
    """Orchestrates ingesting raw VN30 statements into MinIO Bronze Layer with full atomicity."""

    def __init__(
        self,
        client: VNFinancialClient,
        storage: S3Storage,
        bronze_bucket: str = "sec-bronze",
    ) -> None:
        """Initialize ingestion pipeline."""
        self.client = client
        self.storage = storage
        self.bronze_bucket = bronze_bucket

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Compute hex-encoded SHA-256 hash of payload bytes."""
        return hashlib.sha256(data).hexdigest()

    def ingest_ticker(
        self,
        ticker: str,
        force: bool = False,
        accounting_standard: str = "VAS",
    ) -> dict[str, int]:
        """Ingest all 3 financial reports (yearly and quarterly) for a given ticker.

        Strictly differentiates between successful uploads, skipped existing items,
        genuinely empty API responses, HTTP failures, network failures, and parsing errors.
        Guarantees two-object atomicity (raw payload + metadata) with rollback on failure.
        """
        ticker = ticker.strip().upper()
        stats = {
            "uploaded": 0,
            "skipped": 0,
            "empty": 0,
            "failed_http": 0,
            "failed_network": 0,
            "failed_parse": 0,
            "failed_upload": 0,
        }

        for report_type in REPORT_TYPES:
            for period in PERIOD_TYPES:
                s3_raw_key = f"vn_raw/financials/{ticker}_{report_type}_{period}.json"
                s3_meta_key = f"vn_raw/financials/{ticker}_{report_type}_{period}_metadata.json"

                # 1. Fetch data from upstream API with Tenacity retry handling
                try:
                    result = self.client.fetch_financial_report(
                        ticker,
                        report_type,
                        period,
                        accounting_standard=accounting_standard,
                    )
                except httpx.HTTPStatusError as exc:
                    logger.error(
                        "HTTP error fetching %s %s (%s): %s",
                        ticker,
                        report_type,
                        period,
                        exc,
                    )
                    stats["failed_http"] += 1
                    continue
                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    logger.error(
                        "Network failure fetching %s %s (%s) after retries: %s",
                        ticker,
                        report_type,
                        period,
                        exc,
                    )
                    stats["failed_network"] += 1
                    continue
                except json.JSONDecodeError as exc:
                    logger.error(
                        "JSON parse failure for %s %s (%s): %s",
                        ticker,
                        report_type,
                        period,
                        exc,
                    )
                    stats["failed_parse"] += 1
                    continue
                except Exception as exc:
                    logger.error(
                        "Unexpected failure fetching %s %s (%s): %s",
                        ticker,
                        report_type,
                        period,
                        exc,
                    )
                    stats["failed_http"] += 1
                    continue

                # 2. Strict Pre-Bronze Check: Distinguish genuine empty responses
                data = result.get("data")
                if not data or (isinstance(data, list) and len(data) == 0):
                    logger.warning(
                        "Quality Gate Alert: Upstream API returned empty records for %s %s (%s). Marked as empty.",
                        ticker,
                        report_type,
                        period,
                    )
                    stats["empty"] += 1
                    continue

                payload_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
                sha256_hash = self.compute_sha256(payload_bytes)
                fetched_at = datetime.now(UTC).isoformat()

                # 3. Robust Idempotency Check: Require BOTH raw payload and companion metadata to exist
                raw_exists = self.storage.object_exists(self.bronze_bucket, s3_raw_key)
                meta_exists = self.storage.object_exists(self.bronze_bucket, s3_meta_key)

                if not force and raw_exists and meta_exists:
                    try:
                        existing_meta = self.storage.get_metadata(self.bronze_bucket, s3_raw_key)
                        existing_sha = existing_meta.get("metadata", {}).get("sha256")
                        if existing_sha and existing_sha.lower() == sha256_hash.lower():
                            logger.info(
                                "Payload unchanged for %s %s (%s) [SHA: %s]. Skipped.",
                                ticker,
                                report_type,
                                period,
                                sha256_hash[:8],
                            )
                            stats["skipped"] += 1
                            continue
                    except Exception as exc:
                        logger.warning(
                            "Could not verify existing metadata for %s: %s. Overwriting.",
                            s3_raw_key,
                            exc,
                        )

                # 4. Atomic Two-Object Upload with Rollback Compensation
                metadata_dict: dict[str, Any] = {
                    "market": "VN",
                    "ticker": ticker,
                    "report_type": report_type,
                    "period": period,
                    "record_count": len(data) if isinstance(data, list) else 1,
                    "source_url": result.get("source_url", ""),
                    "fetched_at": fetched_at,
                    "sha256": sha256_hash,
                    "byte_size": len(payload_bytes),
                }
                meta_bytes = json.dumps(metadata_dict, indent=2).encode("utf-8")

                raw_uploaded = False
                try:
                    # Upload raw payload first
                    self.storage.upload_bytes(
                        bucket=self.bronze_bucket,
                        key=s3_raw_key,
                        data=payload_bytes,
                        content_type="application/json",
                        metadata={
                            "ticker": ticker,
                            "sha256": sha256_hash,
                            "fetched_at": fetched_at,
                        },
                    )
                    raw_uploaded = True

                    # Upload companion metadata second
                    self.storage.upload_bytes(
                        bucket=self.bronze_bucket,
                        key=s3_meta_key,
                        data=meta_bytes,
                        content_type="application/json",
                    )

                    logger.info(
                        "Successfully ingested s3://%s/%s (%d records)", self.bronze_bucket, s3_raw_key, len(data)
                    )
                    stats["uploaded"] += 1

                except Exception as upload_exc:
                    logger.error(
                        "Upload error for %s (%s, %s): %s",
                        ticker,
                        report_type,
                        period,
                        upload_exc,
                    )
                    # Compensation rollback: Remove orphan raw object if companion metadata failed
                    if raw_uploaded:
                        try:
                            logger.warning(
                                "Rolling back orphan raw object s3://%s/%s",
                                self.bronze_bucket,
                                s3_raw_key,
                            )
                            self.storage.delete_object(self.bronze_bucket, s3_raw_key)
                        except Exception as del_exc:
                            logger.error(
                                "Failed to rollback s3://%s/%s: %s",
                                self.bronze_bucket,
                                s3_raw_key,
                                del_exc,
                            )
                    stats["failed_upload"] += 1

        return stats

    def run(
        self,
        config_path: str = "configs/companies_vn.yaml",
        target_tickers: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute ingestion pipeline for companies listed in configuration."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        companies = cfg.get("companies", [])
        available_tickers = [c["ticker"].upper() for c in companies if "ticker" in c]
        standards_map = {c["ticker"].upper(): c.get("accounting_standard", "VAS") for c in companies if "ticker" in c}

        if target_tickers:
            tickers_to_process = [t.upper() for t in target_tickers if t.upper() in available_tickers]
        else:
            tickers_to_process = available_tickers

        logger.info(
            "Starting VN30 Ingestion for %d companies (force=%s)...",
            len(tickers_to_process),
            force,
        )

        overall_stats = {
            "total_companies": len(tickers_to_process),
            "uploaded": 0,
            "skipped": 0,
            "empty": 0,
            "failed_http": 0,
            "failed_network": 0,
            "failed_parse": 0,
            "failed_upload": 0,
        }

        for idx, ticker in enumerate(tickers_to_process, 1):
            std = standards_map.get(ticker, "VAS")
            logger.info("[%d/%d] Ingesting ticker: %s (standard: %s)", idx, len(tickers_to_process), ticker, std)
            ticker_stats = self.ingest_ticker(ticker=ticker, force=force, accounting_standard=std)
            for k in [
                "uploaded",
                "skipped",
                "empty",
                "failed_http",
                "failed_network",
                "failed_parse",
                "failed_upload",
            ]:
                overall_stats[k] += ticker_stats[k]

        logger.info("VN30 Ingestion complete: %s", overall_stats)
        return overall_stats


def main() -> None:
    """CLI entrypoint for standalone testing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="VN30 Financial Ingestion CLI")
    parser.add_argument("--ticker", type=str, help="Specific ticker to ingest (e.g. FPT)")
    parser.add_argument("--all", action="store_true", help="Ingest all 30 VN30 constituents")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite S3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/companies_vn.yaml",
        help="Path to companies_vn.yaml",
    )
    args = parser.parse_args()

    storage = S3Storage()
    with VNFinancialClient() as client:
        pipeline = VNIngestionPipeline(client=client, storage=storage)
        target = [args.ticker] if args.ticker else None
        pipeline.run(config_path=args.config, target_tickers=target, force=args.force)


if __name__ == "__main__":
    main()
