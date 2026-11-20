"""Unit tests for SHA-256 idempotency and object integrity verification."""

import hashlib
from unittest.mock import MagicMock
import pytest

from src.data.ingestion.fetch_facts import FactsIngestionPipeline
from src.data.storage.s3_storage import S3Storage


def test_verify_integrity_matching_hash():
    """Test verify_integrity returns True when payload matches metadata sha256."""
    mock_s3 = MagicMock(spec=S3Storage)
    sample_bytes = b'{"facts": "sample_data"}'
    sample_sha256 = hashlib.sha256(sample_bytes).hexdigest()

    # Bind original method to mock instance
    mock_s3.get_metadata.return_value = {
        "content_length": len(sample_bytes),
        "metadata": {"sha256": sample_sha256},
    }
    mock_s3.download_bytes.return_value = sample_bytes
    mock_s3.verify_integrity = S3Storage.verify_integrity.__get__(mock_s3, S3Storage)

    assert mock_s3.verify_integrity("sec-bronze", "raw/test.json") is True


def test_verify_integrity_mismatched_hash():
    """Test verify_integrity returns False when payload has been corrupted."""
    mock_s3 = MagicMock(spec=S3Storage)
    sample_bytes = b'{"facts": "sample_data"}'
    corrupted_bytes = b'{"facts": "corrupted_data"}'
    original_sha256 = hashlib.sha256(sample_bytes).hexdigest()

    mock_s3.get_metadata.return_value = {
        "content_length": len(corrupted_bytes),
        "metadata": {"sha256": original_sha256},
    }
    mock_s3.download_bytes.return_value = corrupted_bytes
    mock_s3.verify_integrity = S3Storage.verify_integrity.__get__(mock_s3, S3Storage)

    assert mock_s3.verify_integrity("sec-bronze", "raw/test.json") is False


def test_verify_integrity_missing_metadata():
    """Test verify_integrity returns False when no sha256 metadata exists."""
    mock_s3 = MagicMock(spec=S3Storage)
    mock_s3.get_metadata.return_value = {"content_length": 100, "metadata": {}}
    mock_s3.verify_integrity = S3Storage.verify_integrity.__get__(mock_s3, S3Storage)

    assert mock_s3.verify_integrity("sec-bronze", "raw/test.json") is False


def test_facts_idempotency_skips_when_sha256_matches():
    """Test FactsIngestionPipeline skips S3 upload if payload hash matches existing object."""
    mock_sec = MagicMock()
    mock_s3 = MagicMock(spec=S3Storage)

    payload_bytes = b'{"company": "AAPL", "facts": {}}'
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()

    mock_response = MagicMock()
    mock_response.content = payload_bytes
    mock_sec.get.return_value = mock_response

    mock_s3.object_exists.return_value = True
    mock_s3.get_metadata.return_value = {
        "metadata": {"sha256": payload_hash}
    }

    pipeline = FactsIngestionPipeline(sec_client=mock_sec, s3_storage=mock_s3)
    result = pipeline._fetch_and_store_endpoint(
        url="https://data.sec.gov/sample.json",
        s3_raw_key="raw/sample.json",
        s3_meta_key="raw/sample_metadata.json",
        ticker="AAPL",
        cik_10="0000320193",
        force=False,
    )

    assert result is False
    mock_s3.upload_bytes.assert_not_called()


def test_facts_idempotency_uploads_when_sha256_differs():
    """Test FactsIngestionPipeline uploads to S3 if payload hash differs from existing object."""
    mock_sec = MagicMock()
    mock_s3 = MagicMock(spec=S3Storage)

    old_hash = hashlib.sha256(b"old_data").hexdigest()
    new_bytes = b'{"company": "AAPL", "facts": {"new": 1}}'

    mock_response = MagicMock()
    mock_response.content = new_bytes
    mock_sec.get.return_value = mock_response

    mock_s3.object_exists.return_value = True
    mock_s3.get_metadata.return_value = {
        "metadata": {"sha256": old_hash}
    }

    pipeline = FactsIngestionPipeline(sec_client=mock_sec, s3_storage=mock_s3)
    result = pipeline._fetch_and_store_endpoint(
        url="https://data.sec.gov/sample.json",
        s3_raw_key="raw/sample.json",
        s3_meta_key="raw/sample_metadata.json",
        ticker="AAPL",
        cik_10="0000320193",
        force=False,
    )

    assert result is True
    assert mock_s3.upload_bytes.call_count == 2
