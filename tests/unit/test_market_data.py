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
        default_count = len(repo.DEFAULT_TICKERS)
        assert default_count == 150
        assert len(initial_cfg.items) == default_count
        tickers_in_default = {item.ticker for item in initial_cfg.items}
        assert {"AAPL", "MSFT", "NVDA", "TSM", "INFY", "FPT", "VNM", "HPG", "VCB", "TCB"}.issubset(tickers_in_default)

        # Add new ticker outside default 150 catalog
        updated = repo.add_ticker("NEWCO")
        assert any(item.ticker == "NEWCO" for item in updated.items)
        assert len(updated.items) == default_count + 1

        # Adding duplicate does not increase count
        dup = repo.add_ticker("NEWCO")
        assert len(dup.items) == default_count + 1

        # Remove ticker
        removed = repo.remove_ticker("NEWCO")
        assert not any(item.ticker == "NEWCO" for item in removed.items)
        assert len(removed.items) == default_count

        # Enriched response contains live quotes
        enriched = repo.get_enriched_watchlist()
        assert enriched.total == default_count
        assert len(enriched.items) == default_count
        assert enriched.items[0].price == 100.0


class TestWatchlistAPI:
    """Integration test suite for FastAPI watchlist endpoints."""

    @pytest.fixture(autouse=True)
    def mock_backend_services(self) -> None:
        """Inject mock providers for lightning fast deterministic API tests."""
        from src.api.routes import watchlist as wl_module
        from src.core.financial.company_registry import load_company_catalog

        vn_tickers = {
            str(c["ticker"]).upper()
            for c in load_company_catalog("configs/companies_vn.yaml", include_vn=False)
            if c.get("ticker")
        }
        mock_quote_svc = MagicMock()
        mock_quote_svc.get_quote.side_effect = lambda sym, **kwargs: StockQuote(
            ticker=sym,
            company_name=f"{sym} Corp",
            market=Market.VN if sym in vn_tickers else Market.US,
            price=150.0,
            change=2.5,
            percent_change=1.69,
            currency="VND" if sym in vn_tickers else "USD",
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

    def test_get_watchlist_pagination_and_market_filter(self, test_client: TestClient) -> None:
        """Verify GET /api/watchlist paginates items cleanly and filters by market."""
        resp_p1 = test_client.get("/api/watchlist?page=1&page_size=10")
        assert resp_p1.status_code == 200
        d1 = resp_p1.json()
        assert d1["page"] == 1
        assert d1["page_size"] == 10
        assert len(d1["items"]) == 10
        assert d1["total_pages"] >= 10

        resp_vn = test_client.get("/api/watchlist?page=1&page_size=10&market=VN")
        assert resp_vn.status_code == 200
        d_vn = resp_vn.json()
        assert d_vn["total"] == 50
        assert len(d_vn["items"]) == 10
        assert all(item["market"] == "VN" for item in d_vn["items"])


def test_quote_service_finnhub_limiter_and_no_429_cache() -> None:
    """Verify QuoteService enforces <=0.9 req/s Finnhub limiter, skips caching ERROR, and serves stale quotes."""
    mock_provider = MagicMock()
    mock_provider.market = Market.US
    factory = QuoteProviderFactory(us_provider=mock_provider, vn_provider=mock_provider)

    svc = QuoteService(factory=factory, ttl_seconds=0.05)
    assert svc.finnhub_rate_limiter.rate <= 0.9

    # 1. Transient ERROR (e.g. HTTP 429) must NOT be cached
    mock_provider.fetch_quote.return_value = StockQuote(
        ticker="AAPL",
        market=Market.US,
        price=None,
        as_of="2026-11-23T12:00:00Z",
        status=QuoteStatus.ERROR,
    )
    err_q = svc.get_quote("AAPL")
    assert err_q.status == QuoteStatus.ERROR
    assert "AAPL" not in svc._cache

    # 2. Successful fetch gets cached; when TTL expires and next fetch fails, stale valid price is served
    mock_provider.fetch_quote.return_value = StockQuote(
        ticker="AAPL",
        market=Market.US,
        price=225.0,
        as_of="2026-11-23T12:00:01Z",
        status=QuoteStatus.REAL_TIME,
    )
    ok_q = svc.get_quote("AAPL")
    assert ok_q.price == 225.0

    import time as _time

    _time.sleep(0.06)
    mock_provider.fetch_quote.return_value = StockQuote(
        ticker="AAPL",
        market=Market.US,
        price=None,
        as_of="2026-11-23T12:00:02Z",
        status=QuoteStatus.ERROR,
    )
    stale_q = svc.get_quote("AAPL")
    assert stale_q.price == 225.0
    assert stale_q.status == QuoteStatus.DELAYED
    assert stale_q.is_delayed is True


def test_quote_service_non_blocking_rate_limit_and_lru_bound() -> None:
    """Verify QuoteService never blocks on exhausted tokens (returns SWR or ERROR) and bounds LRU cache."""
    import time as _time

    from src.data.ingestion.rate_limiter import RateLimiter

    mock_provider = MagicMock()
    mock_provider.market = Market.US
    mock_provider.fetch_quote.side_effect = lambda sym: StockQuote(
        ticker=sym,
        market=Market.US,
        price=100.0,
        as_of="2026-11-24T00:00:00Z",
        status=QuoteStatus.REAL_TIME,
    )
    factory = QuoteProviderFactory(us_provider=mock_provider, vn_provider=mock_provider)

    # 1. Exhausted limiter with no stale cache returns ERROR immediately without calling provider
    tight_limiter = RateLimiter(requests_per_second=0.5, capacity=1.0)
    svc = QuoteService(factory=factory, ttl_seconds=0.05, finnhub_rate_limiter=tight_limiter)
    assert svc.get_quote("AAPL").status == QuoteStatus.REAL_TIME

    start = _time.monotonic()
    limited_q = svc.get_quote("MSFT")
    elapsed = _time.monotonic() - start
    assert elapsed < 0.05, f"QuoteService must not block worker threads, took {elapsed:.4f}s"
    assert limited_q.status == QuoteStatus.ERROR
    assert limited_q.price is None
    assert limited_q.company_name == "Microsoft Corporation"
    assert "MSFT" not in svc._cache
    assert mock_provider.fetch_quote.call_count == 1

    # 2. Exhausted limiter with expired fresh TTL but valid stale cache returns DELAYED quote immediately
    _time.sleep(0.06)
    start = _time.monotonic()
    swr_q = svc.get_quote("AAPL")
    elapsed = _time.monotonic() - start
    assert elapsed < 0.05
    assert swr_q.price == 100.0
    assert swr_q.status == QuoteStatus.DELAYED
    assert swr_q.is_delayed is True
    assert mock_provider.fetch_quote.call_count == 1

    # 3. LRU OrderedDict bounds cache size and evicts least-recently-used entries
    lru_svc = QuoteService(
        factory=factory,
        ttl_seconds=60.0,
        finnhub_rate_limiter=RateLimiter(requests_per_second=100.0, capacity=10.0),
        max_cache_size=2,
    )
    assert QuoteService.DEFAULT_MAX_CACHE_SIZE == 500
    lru_svc.get_quote("AAPL")
    lru_svc.get_quote("MSFT")
    # Touch AAPL so MSFT becomes least-recently-used
    lru_svc.get_quote("AAPL")
    lru_svc.get_quote("NVDA")
    assert list(lru_svc._cache.keys()) == ["AAPL", "NVDA"]
