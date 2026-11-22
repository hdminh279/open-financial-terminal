"""Integration tests for Bronze Layer Ingestion Idempotency and Quality Gate 1.

Verifies end-to-end data completeness, SHA-256 cryptographic integrity,
and pipeline idempotency across MinIO Bronze storage.
"""

import pytest
from dotenv import load_dotenv

from src.cli import main
from src.data.storage.s3_storage import S3Storage

load_dotenv()

TARGET_TICKERS = ["AAPL", "MSFT", "GOOGL"]
EXPECTED_FILING_COUNTS = {
    "AAPL": 8,  # 4 10-K + 4 10-Q
    "MSFT": 8,  # 4 10-K + 4 10-Q
    "GOOGL": 7,  # 3 10-K + 4 10-Q (GOOGL reorganized under Alphabet in 2015)
}


@pytest.fixture(scope="module")
def s3_storage() -> S3Storage:
    """Fixture providing an initialized S3Storage adapter connected to MinIO."""
    storage = S3Storage()
    storage.ensure_buckets_exist()
    return storage


def test_bronze_lakehouse_data_completeness(s3_storage: S3Storage):
    """Quality Gate 1 - Check 1: Verify all required Bronze objects exist for target companies."""
    bronze_keys: list[str] = s3_storage.list_objects(bucket="sec-bronze")
    key_set: set[str] = set(bronze_keys)

    # 1. Verify Submissions & Facts exist for all companies
    for ticker in TARGET_TICKERS:
        matching_subs = [
            k
            for k in key_set
            if f"raw/submissions/{ticker}_" in k and k.endswith(".json") and not k.endswith("_metadata.json")
        ]
        matching_facts = [
            k
            for k in key_set
            if f"raw/company_facts/{ticker}_" in k and k.endswith(".json") and not k.endswith("_metadata.json")
        ]

        assert len(matching_subs) >= 1, f"Missing submissions JSON for {ticker}"
        assert len(matching_facts) >= 1, f"Missing company facts JSON for {ticker}"

        # Verify companion audit metadata files exist
        assert f"{matching_subs[0].rsplit('.', 1)[0]}_metadata.json" in key_set, (
            f"Missing companion metadata for submissions {ticker}"
        )
        assert f"{matching_facts[0].rsplit('.', 1)[0]}_metadata.json" in key_set, (
            f"Missing companion metadata for facts {ticker}"
        )

    # 2. Verify 10-K & 10-Q HTML Filings count per company
    for ticker, expected_count in EXPECTED_FILING_COUNTS.items():
        filing_keys = [k for k in key_set if k.startswith(f"raw/filings/{ticker}/") and k.endswith(".htm")]
        assert len(filing_keys) >= expected_count, (
            f"Expected at least {expected_count} filings for {ticker}, but found {len(filing_keys)}: {filing_keys}"
        )


def test_bronze_lakehouse_cryptographic_integrity(s3_storage: S3Storage):
    """Quality Gate 1 - Check 2: Verify SHA-256 integrity for 100% of primary Bronze objects."""
    all_keys = s3_storage.list_objects(bucket="sec-bronze")
    primary_keys = [k for k in all_keys if not k.endswith("_metadata.json")]

    assert len(primary_keys) >= 29, f"Expected at least 29 primary objects, found {len(primary_keys)}"

    failed_keys = []
    for key in primary_keys:
        if not s3_storage.verify_integrity(bucket="sec-bronze", key=key):
            failed_keys.append(key)

    assert not failed_keys, f"Cryptographic integrity failed for {len(failed_keys)} objects: {failed_keys}"


def test_cli_verify_command():
    """Quality Gate 1 - Check 3: Verify the unified CLI 'verify' command exits with code 0."""
    exit_code = main(["verify", "--bucket", "sec-bronze"])
    assert exit_code == 0, "CLI verify command reported integrity failures"


def test_ingestion_pipeline_idempotency(s3_storage: S3Storage):
    """Quality Gate 1 - Check 4: Running 'ingest --all' second time creates zero duplicate objects."""
    initial_keys = set(s3_storage.list_objects(bucket="sec-bronze"))

    # Execute full pipeline again
    exit_code = main(["ingest", "--all"])
    assert exit_code == 0, "Second ingestion run failed with non-zero exit code"

    # Assert object state is completely preserved without duplicates
    post_run_keys = set(s3_storage.list_objects(bucket="sec-bronze"))
    assert post_run_keys == initial_keys, f"Idempotency violated! New objects created: {post_run_keys - initial_keys}"
