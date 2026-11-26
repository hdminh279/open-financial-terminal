"""Unit tests for FilingsIngestionPipeline."""

from unittest.mock import MagicMock

import pytest

from pipelines.ingestion.fetch_filings import FilingsIngestionPipeline
from pipelines.ingestion.filing_resolver import ResolvedFiling


@pytest.fixture
def mock_filing():
    """Create a sample ResolvedFiling instance."""
    return ResolvedFiling(
        ticker="AAPL",
        cik="0000320193",
        form="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        accession_number="0000320193-24-000106",
        primary_document="aapl-20240928.htm",
        sec_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000106/aapl-20240928.htm",
        s3_key="raw/filings/AAPL/10-K_2024-09-28_0000320193-24-000106.htm",
    )


def test_ingest_filing_success(mock_filing):
    """Test successful download and upload of filing HTML and companion metadata."""
    mock_sec_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = b"<html><body>Sample 10-K Report</body></html>"
    mock_sec_client.get.return_value = mock_response

    mock_s3 = MagicMock()
    mock_s3.object_exists.return_value = False

    pipeline = FilingsIngestionPipeline(sec_client=mock_sec_client, s3_storage=mock_s3)
    result = pipeline.ingest_filing(mock_filing)

    assert result is True
    mock_sec_client.get.assert_called_once_with(mock_filing.sec_url)

    # Verify 2 uploads: HTML document and companion metadata JSON
    assert mock_s3.upload_bytes.call_count == 2


def test_ingest_filing_idempotent_skip(mock_filing):
    """Test that existing filings in S3 are skipped when force=False."""
    mock_sec_client = MagicMock()
    mock_s3 = MagicMock()
    mock_s3.object_exists.return_value = True
    mock_s3.get_metadata.return_value = {
        "content_length": 1024,
        "metadata": {"sha256": "dummy_valid_hash"},
    }

    pipeline = FilingsIngestionPipeline(sec_client=mock_sec_client, s3_storage=mock_s3)
    result = pipeline.ingest_filing(mock_filing, force=False)

    assert result is False
    mock_sec_client.get.assert_not_called()
    mock_s3.upload_bytes.assert_not_called()


"""
docker compose run --rm app uv run pytest tests/unit/test_fetch_filings.py -v
"""
