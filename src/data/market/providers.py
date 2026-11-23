"""Market Quote Providers implementing the Strategy Pattern for multi-market data ingestion."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.financial.company_registry import load_company_catalog
from src.data.market.contracts import Market, QuoteStatus, StockQuote

logger = logging.getLogger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Determine whether an HTTP error is transient and safe to retry."""
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {500, 502, 503, 504}
    return False


class BaseQuoteProvider(ABC):
    """Abstract Strategy interface for real-time market quote providers."""

    @property
    @abstractmethod
    def market(self) -> Market:
        """Market jurisdiction served by this provider."""
        pass

    @abstractmethod
    def fetch_quote(self, ticker: str) -> StockQuote | None:
        """Fetch real-time or snapshot quote for a single ticker symbol.

        Args:
            ticker: Uppercase canonical ticker symbol.

        Returns:
            StockQuote instance if successful, or None if ticker not found/unsupported.
        """
        pass


class FinnhubQuoteProvider(BaseQuoteProvider):
    """US Equities live quote provider using the official Finnhub Stock API ($0 Free Tier)."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize Finnhub provider.

        Args:
            api_key: Optional Finnhub API token. Defaults to FINNHUB_API_KEY environment variable.
            timeout: Request timeout in seconds.
            client: Optional injected httpx.Client for testing and connection reuse.
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.timeout = timeout
        self._client = client
        self._name_cache: dict[str, str] = {}
        self._load_catalog_names()

    def _load_catalog_names(self) -> None:
        """Preload company names from local catalog (configs/companies.yaml)."""
        try:
            catalog = load_company_catalog("configs/companies.yaml")
            for c in catalog:
                if "ticker" in c:
                    sym = str(c["ticker"]).upper()
                    name = c.get("english_name") or c.get("name")
                    if name:
                        self._name_cache[sym] = name
        except Exception as exc:
            logger.debug("Failed to preload catalog names in Finnhub provider: %s", exc)

    def _resolve_company_name(self, client: httpx.Client, ticker: str) -> str | None:
        """Resolve human-readable company name from cache or Finnhub profile2."""
        symbol = ticker.upper()
        if symbol in self._name_cache:
            return self._name_cache[symbol]

        if not self.api_key:
            return None

        # Fetch profile from Finnhub if not in local catalog
        try:
            url = f"{self.BASE_URL}/stock/profile2"
            resp = client.get(url, params={"symbol": symbol, "token": self.api_key})
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("name")
                if name:
                    self._name_cache[symbol] = name
                    return name
        except Exception as exc:
            logger.debug("Could not fetch Finnhub profile for %s: %s", symbol, exc)

        return None

    @property
    def market(self) -> Market:
        """Market jurisdiction."""
        return Market.US

    def _get_client(self) -> httpx.Client:
        """Obtain active HTTP client."""
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.timeout)

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
        reraise=True,
    )
    def _execute_request(self, client: httpx.Client, ticker: str) -> dict[str, Any]:
        """Execute HTTP GET to Finnhub /quote endpoint with retry."""
        url = f"{self.BASE_URL}/quote"
        params = {"symbol": ticker, "token": self.api_key}
        resp = client.get(url, params=params)
        if resp.status_code == 429:
            logger.warning("Finnhub HTTP 429 rate limit hit for ticker %s", ticker)
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    def fetch_quote(self, ticker: str) -> StockQuote | None:
        """Fetch quote from Finnhub API."""
        if not self.api_key:
            logger.warning("Finnhub API key is not configured; cannot fetch quote for %s", ticker)
            return StockQuote(
                ticker=ticker,
                company_name=self._name_cache.get(ticker.upper()),
                market=self.market,
                price=None,
                change=None,
                percent_change=None,
                currency="USD",
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.ERROR,
            )

        client = self._get_client()
        should_close = self._client is None
        try:
            data = self._execute_request(client, ticker)
            current_price = data.get("c")
            prev_close = data.get("pc")

            # In Finnhub, unsupported or empty symbols return c=0 and pc=0
            if not current_price and not prev_close:
                logger.info("Finnhub returned no price data for ticker %s (unsupported or inactive)", ticker)
                return StockQuote(
                    ticker=ticker,
                    company_name=self._name_cache.get(ticker.upper()),
                    market=self.market,
                    price=None,
                    change=None,
                    percent_change=None,
                    currency="USD",
                    as_of=datetime.now(UTC).isoformat(),
                    status=QuoteStatus.UNSUPPORTED,
                )

            change = data.get("d")
            percent_change = data.get("dp")
            day_high = data.get("h")
            day_low = data.get("l")
            day_open = data.get("o")
            ts = data.get("t")
            as_of = datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else datetime.now(UTC).isoformat()

            company_name = self._resolve_company_name(client, ticker)

            return StockQuote(
                ticker=ticker,
                company_name=company_name,
                market=self.market,
                price=float(current_price) if current_price is not None else None,
                change=float(change) if change is not None else None,
                percent_change=float(percent_change) if percent_change is not None else None,
                currency="USD",
                day_high=float(day_high) if day_high is not None else None,
                day_low=float(day_low) if day_low is not None else None,
                day_open=float(day_open) if day_open is not None else None,
                previous_close=float(prev_close) if prev_close is not None else None,
                volume=None,
                as_of=as_of,
                status=QuoteStatus.REAL_TIME,
                is_delayed=False,
            )
        except Exception as exc:
            logger.error("Failed to fetch quote for %s from Finnhub: %s", ticker, exc)
            return StockQuote(
                ticker=ticker,
                company_name=self._name_cache.get(ticker.upper()),
                market=self.market,
                price=None,
                change=None,
                percent_change=None,
                currency="USD",
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.ERROR,
            )
        finally:
            if should_close:
                client.close()


class SSIQuoteProvider(BaseQuoteProvider):
    """Vietnam Equities (VN30) live quote provider using SSI iBoard public endpoint ($0)."""

    BASE_URL = "https://iboard-query.ssi.com.vn/stock"

    def __init__(
        self,
        timeout: float = 6.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize SSI quote provider."""
        self.timeout = timeout
        self._client = client

    @property
    def market(self) -> Market:
        """Market jurisdiction."""
        return Market.VN

    def _get_client(self) -> httpx.Client:
        """Obtain active HTTP client."""
        if self._client is not None:
            return self._client
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept": "application/json, text/plain, */*",
        }
        return httpx.Client(timeout=self.timeout, headers=headers)

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.4, min=0.4, max=2.0),
        reraise=True,
    )
    def _execute_request(self, client: httpx.Client, ticker: str) -> dict[str, Any]:
        """Execute HTTP GET to SSI iBoard endpoint."""
        url = f"{self.BASE_URL}/{ticker}"
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()

    def fetch_quote(self, ticker: str) -> StockQuote | None:
        """Fetch quote from SSI iBoard API."""
        client = self._get_client()
        should_close = self._client is None
        try:
            payload = self._execute_request(client, ticker)
            data = payload.get("data")
            if not data or not isinstance(data, dict):
                logger.info("SSI returned empty data object for ticker %s", ticker)
                return StockQuote(
                    ticker=ticker,
                    market=self.market,
                    price=None,
                    change=None,
                    percent_change=None,
                    currency="VND",
                    as_of=datetime.now(UTC).isoformat(),
                    status=QuoteStatus.UNSUPPORTED,
                )

            matched_price = data.get("matchedPrice") or data.get("expectedMatchedPrice")
            ref_price = data.get("refPrice")
            price_change = data.get("priceChange") or data.get("expectedPriceChange")
            pct_change = data.get("priceChangePercent") or data.get("expectedPriceChangePercent")
            company_name = data.get("clientNameEn") or data.get("clientName") or data.get("companyNameEn")
            if not company_name:
                catalog = load_company_catalog("configs/companies_vn.yaml")
                for c in catalog:
                    if str(c.get("ticker", "")).upper() == ticker.upper():
                        company_name = c.get("english_name") or c.get("name")
                        break

            return StockQuote(
                ticker=ticker,
                company_name=company_name,
                market=self.market,
                price=float(matched_price) if matched_price is not None else None,
                change=float(price_change) if price_change is not None else None,
                percent_change=float(pct_change) if pct_change is not None else None,
                currency="VND",
                day_high=float(data.get("highest")) if data.get("highest") is not None else None,
                day_low=float(data.get("lowest")) if data.get("lowest") is not None else None,
                day_open=float(data.get("openPrice")) if data.get("openPrice") is not None else None,
                previous_close=float(ref_price) if ref_price is not None else None,
                volume=int(data.get("stockVol") or data.get("nmTotalTradedQty") or 0) or None,
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.REAL_TIME,
                is_delayed=False,
            )
        except Exception as exc:
            logger.error("Failed to fetch quote for %s from SSI: %s", ticker, exc)
            return StockQuote(
                ticker=ticker,
                market=self.market,
                price=None,
                change=None,
                percent_change=None,
                currency="VND",
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.ERROR,
            )
        finally:
            if should_close:
                client.close()


class MockQuoteProvider(BaseQuoteProvider):
    """Deterministic Mock quote provider for offline testing and zero-cost test suite runs."""

    def __init__(self, market: Market = Market.US, mock_data: dict[str, StockQuote] | None = None) -> None:
        """Initialize mock provider with optional canned responses."""
        self._market = market
        self._mock_data: dict[str, StockQuote] = mock_data or {}

    @property
    def market(self) -> Market:
        """Market jurisdiction."""
        return self._market

    def set_mock_quote(self, ticker: str, quote: StockQuote) -> None:
        """Register a canned quote for a ticker."""
        self._mock_data[ticker.upper()] = quote

    def fetch_quote(self, ticker: str) -> StockQuote | None:
        """Return canned quote or generate deterministic synthetic quote."""
        symbol = ticker.upper()
        if symbol in self._mock_data:
            return self._mock_data[symbol]

        is_vn = self._market == Market.VN
        return StockQuote(
            ticker=symbol,
            company_name=f"Mock Corp ({symbol})",
            market=self._market,
            price=45000.0 if is_vn else 150.0,
            change=500.0 if is_vn else 2.5,
            percent_change=1.12 if is_vn else 1.69,
            currency="VND" if is_vn else "USD",
            day_high=45500.0 if is_vn else 152.0,
            day_low=44500.0 if is_vn else 148.5,
            day_open=44800.0 if is_vn else 149.0,
            previous_close=44500.0 if is_vn else 147.5,
            volume=1000000,
            as_of=datetime.now(UTC).isoformat(),
            status=QuoteStatus.REAL_TIME,
            is_delayed=False,
        )


class QuoteProviderFactory:
    """Factory creating and routing market quote providers based on ticker jurisdiction."""

    def __init__(
        self,
        us_provider: BaseQuoteProvider | None = None,
        vn_provider: BaseQuoteProvider | None = None,
    ) -> None:
        """Initialize factory with optional custom or mocked providers."""
        self._us_provider = us_provider
        self._vn_provider = vn_provider
        self._vn_tickers: set | None = None

    def _get_vn_tickers(self) -> set:
        """Lazily load known Vietnamese company tickers from config."""
        if self._vn_tickers is None:
            catalog = load_company_catalog("configs/companies_vn.yaml")
            self._vn_tickers = {str(c["ticker"]).upper() for c in catalog if "ticker" in c}
        return self._vn_tickers

    def get_provider(self, ticker: str) -> BaseQuoteProvider:
        """Resolve the appropriate provider for a given ticker."""
        symbol = ticker.strip().upper()
        vn_tickers = self._get_vn_tickers()

        if symbol in vn_tickers:
            if self._vn_provider is None:
                self._vn_provider = SSIQuoteProvider()
            return self._vn_provider

        # Default to US provider for US and international ADRs
        if self._us_provider is None:
            self._us_provider = FinnhubQuoteProvider()
        return self._us_provider
