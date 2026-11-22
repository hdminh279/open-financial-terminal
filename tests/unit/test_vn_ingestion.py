"""Unit tests for VN30 Ingestion Pipeline, Tenacity Retries, and Atomicity."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.ingestion.vn_ingestion import (
    VNFinancialClient,
    VNIngestionPipeline,
    _is_retryable_error,
)


class TestVNFinancialClientRetry:
    """Validate Tenacity retry behavior and error classification."""

    def test_retryable_error_predicate(self):
        """Ensure retryable errors are properly separated from terminal client errors."""
        req = httpx.Request("GET", "https://test.com")

        # 5xx and 429 must be retryable
        res_503 = httpx.Response(503, request=req)
        assert _is_retryable_error(httpx.HTTPStatusError("503", request=req, response=res_503)) is True

        res_429 = httpx.Response(429, request=req)
        assert _is_retryable_error(httpx.HTTPStatusError("429", request=req, response=res_429)) is True

        # 4xx client errors must NOT be retried (fail fast)
        res_404 = httpx.Response(404, request=req)
        assert _is_retryable_error(httpx.HTTPStatusError("404", request=req, response=res_404)) is False

        res_400 = httpx.Response(400, request=req)
        assert _is_retryable_error(httpx.HTTPStatusError("400", request=req, response=res_400)) is False

        # Network disconnect and timeout must be retryable
        assert _is_retryable_error(httpx.ConnectError("Connection refused")) is True
        assert _is_retryable_error(httpx.TimeoutException("Read timed out")) is True

    @patch("src.data.ingestion.rate_limiter.RateLimiter.acquire")
    def test_tenacity_retries_on_transient_503(self, mock_rate_limit):
        """Verify Tenacity actually executes retries on transient HTTP 503 until success."""
        client = VNFinancialClient()
        req = httpx.Request("GET", "https://test.com")
        res_503 = httpx.Response(503, request=req)
        res_200 = httpx.Response(200, request=req, json=[{"year": 2023, "revenue": 1000}])

        # Fail twice with 503, succeed on 3rd attempt
        client.client.get = MagicMock(
            side_effect=[
                httpx.HTTPStatusError("503", request=req, response=res_503),
                httpx.HTTPStatusError("503", request=req, response=res_503),
                res_200,
            ]
        )

        # Patch tenacity sleep to run instantly (< 0.01s)
        with patch("tenacity.nap.time.sleep", return_value=None):
            result = client.fetch_financial_report(ticker="FPT", report_type="incomestatement", period="yearly")

        assert client.client.get.call_count == 3
        assert result["ticker"] == "FPT"
        assert len(result["data"]) == 1

    @patch("src.data.ingestion.rate_limiter.RateLimiter.acquire")
    def test_tenacity_fails_fast_on_404(self, mock_rate_limit):
        """Verify client errors (404) fail fast without retrying 5 times."""
        client = VNFinancialClient()
        req = httpx.Request("GET", "https://test.com")
        res_404 = httpx.Response(404, request=req)

        client.client.get = MagicMock(side_effect=httpx.HTTPStatusError("404", request=req, response=res_404))

        with pytest.raises(httpx.HTTPStatusError):
            client.fetch_financial_report(ticker="FPT", report_type="incomestatement", period="yearly")

        # Must fail fast on attempt 1 with zero retries
        assert client.client.get.call_count == 1


class TestVNIngestionPipelineAtomicityAndClassification:
    """Validate distinct error classification and atomic two-object rollback."""

    def test_distinguish_empty_from_http_error(self):
        """Ensure empty API responses are counted in 'empty', while 503 is counted in 'failed_http'."""
        mock_client = MagicMock()
        mock_storage = MagicMock()
        pipeline = VNIngestionPipeline(client=mock_client, storage=mock_storage)

        # Scenario 1: API returns HTTP 200 with empty list []
        mock_client.fetch_financial_report.return_value = {
            "ticker": "FPT",
            "report_type": "incomestatement",
            "period": "yearly",
            "source_url": "https://test.com",
            "data": [],
        }

        with (
            patch("src.data.ingestion.vn_ingestion.REPORT_TYPES", ["incomestatement"]),
            patch("src.data.ingestion.vn_ingestion.PERIOD_TYPES", ["yearly"]),
        ):
            stats = pipeline.ingest_ticker("FPT")

        assert stats["empty"] == 1
        assert stats["failed_http"] == 0
        assert stats["uploaded"] == 0

        # Scenario 2: API returns 503
        req = httpx.Request("GET", "https://test.com")
        mock_client.fetch_financial_report.side_effect = httpx.HTTPStatusError(
            "503", request=req, response=httpx.Response(503, request=req)
        )

        with (
            patch("src.data.ingestion.vn_ingestion.REPORT_TYPES", ["incomestatement"]),
            patch("src.data.ingestion.vn_ingestion.PERIOD_TYPES", ["yearly"]),
        ):
            stats_fail = pipeline.ingest_ticker("FPT")

        assert stats_fail["failed_http"] == 1
        assert stats_fail["empty"] == 0
        assert stats_fail["uploaded"] == 0

    def test_atomic_upload_rollback_on_metadata_failure(self):
        """Verify orphan raw object is rolled back if companion metadata upload fails."""
        mock_client = MagicMock()
        mock_storage = MagicMock()
        pipeline = VNIngestionPipeline(client=mock_client, storage=mock_storage)

        mock_client.fetch_financial_report.return_value = {
            "ticker": "FPT",
            "report_type": "incomestatement",
            "period": "yearly",
            "source_url": "https://test.com",
            "data": [{"revenue": 50000}],
        }
        mock_storage.object_exists.return_value = False

        # Raw upload succeeds, metadata upload raises exception
        mock_storage.upload_bytes.side_effect = [
            None,  # First call (raw upload) succeeds
            Exception("S3 metadata write timeout"),  # Second call (meta upload) fails
        ]

        with (
            patch("src.data.ingestion.vn_ingestion.REPORT_TYPES", ["incomestatement"]),
            patch("src.data.ingestion.vn_ingestion.PERIOD_TYPES", ["yearly"]),
        ):
            stats = pipeline.ingest_ticker("FPT")

        # Compensation rollback must be called to delete raw object
        mock_storage.delete_object.assert_called_once_with(
            "sec-bronze",
            "vn_raw/financials/FPT_incomestatement_yearly.json",
        )
        assert stats["failed_upload"] == 1
        assert stats["uploaded"] == 0

    def test_idempotency_requires_both_raw_and_metadata_to_skip(self):
        """Verify pipeline will NOT skip if companion metadata is missing from previous partial run."""
        mock_client = MagicMock()
        mock_storage = MagicMock()
        pipeline = VNIngestionPipeline(client=mock_client, storage=mock_storage)

        mock_client.fetch_financial_report.return_value = {
            "ticker": "FPT",
            "report_type": "incomestatement",
            "period": "yearly",
            "source_url": "https://test.com",
            "data": [{"revenue": 50000}],
        }

        # Case 1: Raw exists, but meta does NOT exist -> Must NOT skip, must upload
        mock_storage.object_exists.side_effect = lambda bucket, key: "_metadata.json" not in key
        with (
            patch("src.data.ingestion.vn_ingestion.REPORT_TYPES", ["incomestatement"]),
            patch("src.data.ingestion.vn_ingestion.PERIOD_TYPES", ["yearly"]),
        ):
            stats = pipeline.ingest_ticker("FPT")

        assert stats["uploaded"] == 1
        assert stats["skipped"] == 0

        # Case 2: Both exist and SHA matches -> Must skip
        mock_storage.object_exists.side_effect = None
        mock_storage.object_exists.return_value = True
        payload_bytes = json.dumps(
            mock_client.fetch_financial_report.return_value, ensure_ascii=False, indent=2
        ).encode("utf-8")
        calc_sha = pipeline.compute_sha256(payload_bytes)

        mock_storage.get_metadata.return_value = {"metadata": {"sha256": calc_sha}}

        with (
            patch("src.data.ingestion.vn_ingestion.REPORT_TYPES", ["incomestatement"]),
            patch("src.data.ingestion.vn_ingestion.PERIOD_TYPES", ["yearly"]),
        ):
            stats_skip = pipeline.ingest_ticker("FPT")

        assert stats_skip["skipped"] == 1
        assert stats_skip["uploaded"] == 0
