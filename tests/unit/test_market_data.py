"""Unit tests for market data contracts, providers, quote service, watchlist repository, and API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.market.contracts import (
    AddWatchlistRequest,
    Market,
    QuoteStatus,
    StockQuote,
)
from src.data.market.providers import (
    FinnhubQuoteProvider,
    MockQuoteProvider,
    QuoteProviderFactory,
    SSIQuoteProvider,
)
from src.data.market.service import QuoteService, WatchlistRepository


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing FastAPI test client."""
    return TestClient(app)


class TestMarketContracts:
    """Validation test suite for Pydantic market models."""

    def test_stock_quote_sanitization(self) -> None:
        """Ensure ticker whitespace is trimmed and converted to uppercase."""
        quote = StockQuote(
            ticker="  aapl  ",
            market=Market.US,
            price=220.5,
            change=3.2,
            percent_change=1.47,
            as_of="2026-11-20T12:00:00Z",
        )
        assert quote.ticker == "AAPL"
        assert quote.currency == "USD"
        assert quote.status == QuoteStatus.REAL_TIME

    def test_empty_ticker_validation_error(self) -> None:
        """Ensure empty ticker raises ValueError."""
        with pytest.raises(ValueError):
            StockQuote(
                ticker="   ",
                market=Market.US,
                as_of="2026-11-20T12:00:00Z",
            )

    def test_add_watchlist_request_validation(self) -> None:
        """Ensure AddWatchlistRequest enforces uppercase and non-empty ticker."""
        req = AddWatchlistRequest(ticker="fpt")
        assert req.ticker == "FPT"
        assert req.market is None


class TestQuoteProviders:
    """Test suite for Strategy Pattern quote providers."""

    def test_mock_provider_deterministic(self) -> None:
        """Ensure MockQuoteProvider returns valid quotes without network."""
        provider = MockQuoteProvider(market=Market.US)
        quote = provider.fetch_quote("MSFT")
        assert quote is not None
        assert quote.ticker == "MSFT"
        assert quote.market == Market.US
        assert quote.price == 150.0

        vn_provider = MockQuoteProvider(market=Market.VN)
        vn_quote = vn_provider.fetch_quote("FPT")
        assert vn_quote is not None
        assert vn_quote.ticker == "FPT"
        assert vn_quote.market == Market.VN
        assert vn_quote.currency == "VND"
        assert vn_quote.price == 45000.0

    def test_finnhub_provider_parsing(self) -> None:
        """Test FinnhubQuoteProvider parsing mock HTTP response."""
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "c": 225.50,
            "d": 4.50,
            "dp": 2.04,
            "h": 227.00,
            "l": 221.00,
            "o": 222.00,
            "pc": 221.00,
            "t": 1795193000,
        }
        mock_resp.raise_for_status.return_value = None
        mock_http.get.return_value = mock_resp

        provider = FinnhubQuoteProvider(api_key="test_token", client=mock_http)
        quote = provider.fetch_quote("AAPL")
        assert quote is not None
        assert quote.ticker == "AAPL"
        assert quote.company_name == "Apple Inc."
        assert quote.price == 225.50
        assert quote.change == 4.50
        assert quote.percent_change == 2.04
        assert quote.currency == "USD"
        assert quote.status == QuoteStatus.REAL_TIME

    def test_finnhub_unsupported_symbol(self) -> None:
        """Test Finnhub handling when symbol returns 0 values."""
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"c": 0, "d": None, "dp": None, "pc": 0, "t": 0}
        mock_resp.raise_for_status.return_value = None
        mock_http.get.return_value = mock_resp

        provider = FinnhubQuoteProvider(api_key="test_token", client=mock_http)
        quote = provider.fetch_quote("UNKNOWN_SYM")
        assert quote is not None
        assert quote.status == QuoteStatus.UNSUPPORTED
        assert quote.price is None

    def test_ssi_provider_parsing(self) -> None:
        """Test SSIQuoteProvider parsing mock HTTP response."""
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "stockSymbol": "FPT",
                "clientNameEn": "FPT Corporation",
                "matchedPrice": 45000,
                "priceChange": -700,
                "priceChangePercent": -1.53,
                "refPrice": 45700,
                "highest": 46000,
                "lowest": 44650,
                "openPrice": 45600,
                "stockVol": 15670800,
            }
        }
        mock_resp.raise_for_status.return_value = None
        mock_http.get.return_value = mock_resp

        provider = SSIQuoteProvider(client=mock_http)
        quote = provider.fetch_quote("FPT")
        assert quote is not None
        assert quote.ticker == "FPT"
        assert quote.price == 45000.0
        assert quote.change == -700.0
        assert quote.percent_change == -1.53
        assert quote.currency == "VND"
        assert quote.market == Market.VN

    def test_quote_provider_factory_routing(self) -> None:
        """Ensure QuoteProviderFactory dispatches VN30 tickers to VN provider and others to US."""
        mock_us = MockQuoteProvider(market=Market.US)
        mock_vn = MockQuoteProvider(market=Market.VN)
        factory = QuoteProviderFactory(us_provider=mock_us, vn_provider=mock_vn)

        assert factory.get_provider("FPT").market == Market.VN
        assert factory.get_provider("VNM").market == Market.VN
        assert factory.get_provider("AAPL").market == Market.US
        assert factory.get_provider("NVDA").market == Market.US


class TestQuoteServiceAndRepository:
    """Test suite for QuoteService TTL caching and WatchlistRepository."""

    def test_quote_service_ttl_caching(self) -> None:
        """Ensure QuoteService serves subsequent calls from cache within TTL."""
        mock_provider = MagicMock()
        mock_provider.market = Market.US
        mock_quote = StockQuote(
            ticker="AAPL",
            market=Market.US,
            price=220.0,
            change=2.0,
            percent_change=0.92,
            as_of="2026-11-20T12:00:00Z",
        )
        mock_provider.fetch_quote.return_value = mock_quote

        factory = QuoteProviderFactory(us_provider=mock_provider, vn_provider=mock_provider)
        svc = QuoteService(factory=factory, ttl_seconds=60.0)

        # First call fetches from provider
        q1 = svc.get_quote("AAPL")
        assert q1.price == 220.0
        assert mock_provider.fetch_quote.call_count == 1

        # Second call within TTL hits cache
        q2 = svc.get_quote("AAPL")
        assert q2.price == 220.0
        assert mock_provider.fetch_quote.call_count == 1

        # Force refresh bypasses cache
        q3 = svc.get_quote("AAPL", force_refresh=True)
        assert q3.price == 220.0
        assert mock_provider.fetch_quote.call_count == 2

    def test_watchlist_repository_lifecycle(self) -> None:
        """Test WatchlistRepository add, remove, and duplicate handling in memory."""
        from src.data.storage.s3_storage import S3Storage

        mock_s3 = MagicMock(spec=S3Storage)
        mock_s3.object_exists.return_value = False
        mock_quote_svc = MagicMock()
        mock_quote_svc.get_quote.side_effect = lambda sym, **kwargs: StockQuote(
            ticker=sym,
            company_name=f"{sym} Corp",
            market=Market.VN if sym in ("FPT", "VNM", "HPG") else Market.US,
            price=100.0,
            as_of="2026-11-20T12:00:00Z",
        )
        mock_quote_svc.get_quotes_batch.side_effect = lambda syms, **kwargs: [mock_quote_svc.get_quote(s) for s in syms]

        repo = WatchlistRepository(s3_storage=mock_s3, quote_service=mock_quote_svc)
        initial_cfg = repo.get_watchlist()
        assert len(initial_cfg.items) == 7

        # Add new ticker
        updated = repo.add_ticker("VIC")
        assert any(item.ticker == "VIC" for item in updated.items)
        assert len(updated.items) == 8

        # Adding duplicate does not increase count
        dup = repo.add_ticker("VIC")
        assert len(dup.items) == 8

        # Remove ticker
        removed = repo.remove_ticker("VIC")
        assert not any(item.ticker == "VIC" for item in removed.items)
        assert len(removed.items) == 7

        # Enriched response contains live quotes
        enriched = repo.get_enriched_watchlist()
        assert enriched.total == 7
        assert len(enriched.items) == 7
        assert enriched.items[0].price == 100.0


class TestWatchlistAPI:
    """Integration test suite for FastAPI watchlist endpoints."""

    @pytest.fixture(autouse=True)
    def mock_backend_services(self) -> None:
        """Inject mock providers for lightning fast deterministic API tests."""
        from src.api.routes import watchlist as wl_module

        mock_quote_svc = MagicMock()
        mock_quote_svc.get_quote.side_effect = lambda sym, **kwargs: StockQuote(
            ticker=sym,
            company_name=f"{sym} Corp",
            market=Market.VN if sym in ("FPT", "VNM", "HPG", "VIC") else Market.US,
            price=150.0,
            change=2.5,
            percent_change=1.69,
            currency="VND" if sym in ("FPT", "VNM", "HPG", "VIC") else "USD",
            as_of="2026-11-20T12:00:00Z",
            status=QuoteStatus.REAL_TIME,
        )
        mock_quote_svc.get_quotes_batch.side_effect = lambda syms, **kwargs: [mock_quote_svc.get_quote(s) for s in syms]

        mock_s3 = MagicMock()
        mock_s3.exists.return_value = False
        repo = WatchlistRepository(s3_storage=mock_s3, quote_service=mock_quote_svc)

        with (
            patch.object(wl_module, "get_watchlist_repository", return_value=repo),
            patch.object(wl_module, "get_quote_service", return_value=mock_quote_svc),
        ):
            yield

    def test_get_watchlist_endpoint(self, test_client: TestClient) -> None:
        """Test GET /api/watchlist returns valid response structure."""
        resp = test_client.get("/api/watchlist")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1
        first_item = data["items"][0]
        assert "ticker" in first_item
        assert "market" in first_item
        assert "currency" in first_item

    def test_add_and_delete_watchlist_endpoint(self, test_client: TestClient) -> None:
        """Test POST /api/watchlist and DELETE /api/watchlist/{ticker}."""
        # 1. Add NVDA (or re-add)
        add_resp = test_client.post("/api/watchlist", json={"ticker": "TESTSYM"})
        assert add_resp.status_code == 201
        data = add_resp.json()
        assert any(item["ticker"] == "TESTSYM" for item in data["items"])

        # 2. Delete TESTSYM
        del_resp = test_client.delete("/api/watchlist/TESTSYM")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert not any(item["ticker"] == "TESTSYM" for item in del_data["items"])

    def test_get_quotes_endpoint(self, test_client: TestClient) -> None:
        """Test GET /api/quotes with comma-separated symbols."""
        resp = test_client.get("/api/quotes?tickers=AAPL,FPT")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        tickers = {item["ticker"] for item in data}
        assert "AAPL" in tickers
        assert "FPT" in tickers
