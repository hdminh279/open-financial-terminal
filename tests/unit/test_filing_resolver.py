"""Unit tests for the SEC FilingResolver module."""

import pytest

from pipelines.ingestion.filing_resolver import FilingResolver


@pytest.fixture
def mock_submissions_dict():
    """Create a minimal mock submissions dictionary with mixed forms."""
    return {
        "cik": "0000320193",
        "entityType": "operating",
        "sic": "3571",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000106",
                    "0000320193-24-000080",
                    "0000320193-24-000050",
                    "0000320193-24-000020",
                    "0000320193-23-000106",
                    "0000320193-23-000099",  # 8-K to be filtered out
                    "0000320193-23-000075",
                    "0000320193-23-000050",
                    "0000320193-22-000108",
                    "0000320193-21-000105",
                ],
                "filingDate": [
                    "2024-11-01",
                    "2024-08-02",
                    "2024-05-03",
                    "2024-02-02",
                    "2023-11-03",
                    "2023-10-15",
                    "2023-08-04",
                    "2023-05-05",
                    "2022-10-28",
                    "2021-10-29",
                ],
                "reportDate": [
                    "2024-09-28",
                    "2024-06-29",
                    "2024-03-30",
                    "2023-12-30",
                    "2023-09-30",
                    "2023-10-15",
                    "2023-07-01",
                    "2023-04-01",
                    "2022-09-24",
                    "2021-09-25",
                ],
                "form": [
                    "10-K",
                    "10-Q",
                    "10-Q",
                    "10-Q",
                    "10-K",
                    "8-K",
                    "10-Q",
                    "10-Q",
                    "10-K",
                    "10-K",
                ],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240629.htm",
                    "aapl-20240330.htm",
                    "aapl-20231230.htm",
                    "aapl-20230930.htm",
                    "aapl-20231015.htm",
                    "aapl-20230701.htm",
                    "aapl-20230401.htm",
                    "aapl-20220924.htm",
                    "aapl-20210925.htm",
                ],
            }
        },
    }


def test_build_sec_archive_url():
    """Verify official SEC archive URL formats without leading zeros or hyphens."""
    url = FilingResolver.build_sec_archive_url(
        cik="0000320193",
        accession_number="0000320193-23-000106",
        primary_document="aapl-20230930.htm",
    )
    expected = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    assert url == expected


def test_build_s3_key():
    """Verify S3 key format for raw filing HTML in Bronze bucket."""
    key = FilingResolver.build_s3_key(
        ticker="AAPL",
        form="10-K",
        report_date="2023-09-30",
        accession_number="0000320193-23-000106",
    )
    assert key == "raw/filings/AAPL/10-K_2023-09-30_0000320193-23-000106.htm"


def test_resolve_top_4_filings_per_form(mock_submissions_dict):
    """Verify resolver extracts up to 4 recent filings for both 10-K and 10-Q."""
    resolver = FilingResolver()
    results = resolver.resolve_from_submissions_dict(
        submissions_data=mock_submissions_dict,
        target_forms=("10-K", "10-Q"),
        limit_per_form=4,
    )

    forms = [r.form for r in results]
    k_count = forms.count("10-K")
    q_count = forms.count("10-Q")

    # In mock data, there are 4 10-Ks and 5 10-Qs. We expect exactly 4 10-Ks and 4 10-Qs.
    assert k_count == 4
    assert q_count == 4
    assert len(results) == 8

    # Ensure 8-K was completely filtered out
    assert "8-K" not in forms

    # Verify first filing is the most recent 10-K
    latest_k = results[0]
    assert latest_k.form == "10-K"
    assert latest_k.accession_number == "0000320193-24-000106"
    assert latest_k.primary_document == "aapl-20240928.htm"
    assert latest_k.ticker == "AAPL"
    assert latest_k.cik == "0000320193"


def test_resolve_empty_submissions():
    """Verify resolver handles empty submissions gracefully without errors."""
    resolver = FilingResolver()
    results = resolver.resolve_from_submissions_dict({"cik": "0000320193", "tickers": ["AAPL"], "filings": {}})
    assert results == []


def test_resolve_uses_explicit_ticker_when_submissions_tickers_empty(mock_submissions_dict):
    """Verify explicit registry ticker overrides empty SEC submissions tickers array."""
    payload = {**mock_submissions_dict, "cik": "0000034088", "tickers": []}
    resolver = FilingResolver()
    results = resolver.resolve_from_submissions_dict(
        submissions_data=payload,
        target_forms=("10-K",),
        limit_per_form=1,
        ticker="XOM",
    )
    assert len(results) == 1
    assert results[0].ticker == "XOM"
    assert results[0].s3_key.startswith("raw/filings/XOM/")


"""
docker compose run --rm app uv run pytest tests/unit/test_filing_resolver.py -v
"""
