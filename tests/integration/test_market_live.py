"""Live integration smoke tests for market quote providers (Finnhub & SSI).

Marked with @pytest.mark.live so it can be executed on-demand with active internet connection.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.data.market.contracts import Market, QuoteStatus
from src.data.market.providers import FinnhubQuoteProvider, SSIQuoteProvider
from src.data.market.service import QuoteService


@pytest.mark.live
def test_live_finnhub_quote() -> None:
    """Smoke test fetching live quote for AAPL via Finnhub with configured key."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        pytest.skip("FINNHUB_API_KEY environment variable not set")

    provider = FinnhubQuoteProvider(api_key=api_key)
    quote = provider.fetch_quote("AAPL")
    assert quote is not None
    assert quote.ticker == "AAPL"
    assert quote.market == Market.US
    assert quote.currency == "USD"
    assert quote.status == QuoteStatus.REAL_TIME
    assert quote.price is not None
    assert quote.price > 0


@pytest.mark.live
def test_live_ssi_quote() -> None:
    """Smoke test fetching live quote for FPT via SSI iBoard public endpoint."""
    provider = SSIQuoteProvider()
    quote = provider.fetch_quote("FPT")
    assert quote is not None
    assert quote.ticker == "FPT"
    assert quote.market == Market.VN
    assert quote.currency == "VND"
    assert quote.status == QuoteStatus.REAL_TIME
    assert quote.price is not None
    assert quote.price > 1000  # Stock price in VND is in thousands


@pytest.mark.live
def test_live_multi_market_service() -> None:
    """Smoke test fetching batch multi-market quotes (US + VN) via QuoteService."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        pytest.skip("FINNHUB_API_KEY environment variable not set")

    svc = QuoteService()
    quotes = svc.get_quotes_batch(["AAPL", "FPT"])
    assert len(quotes) == 2
    by_ticker = {q.ticker: q for q in quotes}
    assert by_ticker["AAPL"].market == Market.US
    assert by_ticker["AAPL"].price is not None
    assert by_ticker["FPT"].market == Market.VN
    assert by_ticker["FPT"].price is not None
