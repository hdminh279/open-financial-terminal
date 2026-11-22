"""Market Quote Service and Persistent Watchlist Repository Engine.

Combines Token Bucket rate limiting, in-memory TTL caching, and MinIO S3 Lakehouse
persistence (sec-silver/user_settings/watchlist.json) with fallback in-memory caching.
"""

import concurrent.futures
import json
import logging
import os
import time
from datetime import UTC, datetime

from src.core.financial.company_registry import load_company_catalog
from src.data.ingestion.rate_limiter import RateLimiter
from src.data.market.contracts import (
    Market,
    QuoteStatus,
    StockQuote,
    WatchlistConfig,
    WatchlistItem,
    WatchlistResponse,
)
from src.data.market.providers import QuoteProviderFactory
from src.data.storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)


class QuoteService:
    """Service providing cached, rate-limited market quotes across multi-market providers."""

    def __init__(
        self,
        factory: QuoteProviderFactory | None = None,
        ttl_seconds: float = 300.0,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Initialize QuoteService.

        Args:
            factory: Optional QuoteProviderFactory. Defaults to standard production factory.
            ttl_seconds: Time-to-live for cached quotes in seconds (default 300s).
            rate_limiter: Optional Token Bucket RateLimiter (default 5 req/sec burst for responsiveness).
        """
        self.factory = factory or QuoteProviderFactory()
        self.ttl_seconds = ttl_seconds
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_second=5.0)
        self._cache: dict[str, tuple[StockQuote, float]] = {}

    def get_quote(self, ticker: str, force_refresh: bool = False) -> StockQuote:
        """Retrieve quote with in-memory TTL caching."""
        symbol = ticker.strip().upper()
        now = time.time()

        if not force_refresh and symbol in self._cache:
            quote, expires_at = self._cache[symbol]
            if now < expires_at:
                logger.debug("Quote cache hit for %s (expires in %.1fs)", symbol, expires_at - now)
                return quote

        # Enforce rate limiter before upstream HTTP call
        self.rate_limiter.acquire()

        provider = self.factory.get_provider(symbol)
        quote = provider.fetch_quote(symbol)

        if quote is None:
            quote = StockQuote(
                ticker=symbol,
                market=provider.market,
                price=None,
                change=None,
                percent_change=None,
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.UNSUPPORTED,
            )

        self._cache[symbol] = (quote, now + self.ttl_seconds)
        return quote

    def get_quotes_batch(self, tickers: list[str], force_refresh: bool = False) -> list[StockQuote]:
        """Retrieve quotes for a list of tickers in parallel."""
        if not tickers:
            return []

        workers = min(len(tickers), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {executor.submit(self.get_quote, t, force_refresh): t for t in tickers}
            results: dict[str, StockQuote] = {}
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    results[t] = future.result()
                except Exception as exc:
                    logger.error("Failed to fetch quote for %s in batch: %s", t, exc)
                    results[t] = StockQuote(
                        ticker=t,
                        market=Market.US,
                        price=None,
                        as_of=datetime.now(UTC).isoformat(),
                        status=QuoteStatus.ERROR,
                    )

            # Preserve the original order of requested tickers
            return [results[t] for t in tickers if t in results]


def _load_default_watchlist_tickers() -> tuple[tuple[str, Market], ...]:
    """Build default multi-market watchlist from all configured SEC issuers plus VN benchmark tickers."""
    seen: set[str] = set()
    entries: list[tuple[str, Market]] = []

    for company in load_company_catalog("configs/companies.yaml", include_vn=False):
        ticker = str(company.get("ticker", "")).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            entries.append((ticker, Market.US))

    for vn_ticker in ("FPT", "VNM", "HPG"):
        if vn_ticker not in seen:
            seen.add(vn_ticker)
            entries.append((vn_ticker, Market.VN))

    if entries:
        return tuple(entries)
    return (
        ("AAPL", Market.US),
        ("MSFT", Market.US),
        ("GOOGL", Market.US),
        ("NVDA", Market.US),
        ("FPT", Market.VN),
        ("VNM", Market.VN),
        ("HPG", Market.VN),
    )


class WatchlistRepository:
    """Repository managing persistent user watchlists in MinIO S3 with local memory fallback."""

    BUCKET_NAME = "sec-silver"
    STORAGE_KEY = "user_settings/watchlist.json"
    MAX_ITEMS = 200

    DEFAULT_TICKERS: tuple[tuple[str, Market], ...] = _load_default_watchlist_tickers()

    def __init__(
        self,
        s3_storage: S3Storage | None = None,
        quote_service: QuoteService | None = None,
    ) -> None:
        """Initialize repository.

        Args:
            s3_storage: Optional S3Storage adapter.
            quote_service: Optional QuoteService for quote enrichment.
        """
        self._s3 = s3_storage
        self._s3_enabled = (
            True if s3_storage is not None else os.getenv("ENABLE_S3_WATCHLIST", "true").lower() == "true"
        )
        self._memory_cache: WatchlistConfig | None = None
        self.quote_service = quote_service or QuoteService()

    def _init_s3(self) -> S3Storage | None:
        """Lazy-initialize S3Storage adapter."""
        if self._s3 is not None:
            return self._s3
        if not self._s3_enabled:
            return None
        try:
            self._s3 = S3Storage()
            return self._s3
        except Exception as exc:
            logger.warning("MinIO S3Storage initialization failed (%s); operating in memory-only mode.", exc)
            self._s3 = None
            return None

    def _create_default_config(self, user_id: str = "default_user") -> WatchlistConfig:
        """Create a default populated watchlist configuration."""
        now = datetime.now(UTC).isoformat()
        items = [WatchlistItem(ticker=ticker, market=market, added_at=now) for ticker, market in self.DEFAULT_TICKERS]
        return WatchlistConfig(
            version="1.0",
            user_id=user_id,
            items=items,
            updated_at=now,
        )

    def _load_from_s3(self) -> WatchlistConfig | None:
        """Attempt to load watchlist JSON from MinIO S3 Lakehouse."""
        s3 = self._init_s3()
        if not s3:
            return None

        try:
            if not s3.object_exists(self.BUCKET_NAME, self.STORAGE_KEY):
                logger.info("Watchlist not found in S3 (%s/%s); creating default.", self.BUCKET_NAME, self.STORAGE_KEY)
                default_cfg = self._create_default_config()
                self._save_to_s3(default_cfg)
                return default_cfg

            raw_bytes = s3.download_bytes(self.BUCKET_NAME, self.STORAGE_KEY)
            data = json.loads(raw_bytes.decode("utf-8"))
            return WatchlistConfig(**data)
        except Exception as exc:
            logger.error("Failed to load watchlist from MinIO S3: %s", exc)
            return None

    def _save_to_s3(self, config: WatchlistConfig) -> bool:
        """Persist watchlist JSON to MinIO S3 Lakehouse."""
        s3 = self._init_s3()
        if not s3:
            return False

        try:
            s3.ensure_buckets_exist([self.BUCKET_NAME])
            payload = config.model_dump_json(indent=2).encode("utf-8")
            s3.upload_bytes(
                bucket=self.BUCKET_NAME,
                key=self.STORAGE_KEY,
                data=payload,
                content_type="application/json",
                metadata={"user_id": config.user_id, "updated_at": config.updated_at},
            )
            return True
        except Exception as exc:
            logger.error("Failed to save watchlist to MinIO S3: %s", exc)
            return False

    def get_watchlist(self, user_id: str = "default_user") -> WatchlistConfig:
        """Retrieve user watchlist configuration."""
        if self._memory_cache is not None:
            return self._memory_cache

        persisted = self._load_from_s3()
        if persisted is not None:
            self._memory_cache = persisted
            return persisted

        default_cfg = self._create_default_config(user_id=user_id)
        self._memory_cache = default_cfg
        return default_cfg

    def add_ticker(
        self,
        ticker: str,
        market: Market | None = None,
        user_id: str = "default_user",
    ) -> WatchlistConfig:
        """Add a ticker symbol to the user's watchlist."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Ticker symbol cannot be empty")

        config = self.get_watchlist(user_id=user_id)

        # Check if already present
        if any(item.ticker == symbol for item in config.items):
            logger.info("Ticker %s is already present in watchlist", symbol)
            return config

        if len(config.items) >= self.MAX_ITEMS:
            raise ValueError(f"Watchlist capacity limit reached (maximum {self.MAX_ITEMS} tickers)")

        # Auto-detect market if not specified
        if market is None:
            vn_catalog = load_company_catalog("configs/companies_vn.yaml")
            vn_tickers = {str(c["ticker"]).upper() for c in vn_catalog if "ticker" in c}
            market = Market.VN if symbol in vn_tickers else Market.US

        now = datetime.now(UTC).isoformat()
        new_item = WatchlistItem(ticker=symbol, market=market, added_at=now)
        config.items.append(new_item)
        config.updated_at = now

        self._memory_cache = config
        self._save_to_s3(config)
        return config

    def remove_ticker(self, ticker: str, user_id: str = "default_user") -> WatchlistConfig:
        """Remove a ticker symbol from the user's watchlist."""
        symbol = ticker.strip().upper()
        config = self.get_watchlist(user_id=user_id)

        initial_len = len(config.items)
        config.items = [item for item in config.items if item.ticker != symbol]

        if len(config.items) != initial_len:
            now = datetime.now(UTC).isoformat()
            config.updated_at = now
            self._memory_cache = config
            self._save_to_s3(config)

        return config

    def reset_to_default(self, user_id: str = "default_user") -> WatchlistConfig:
        """Reset watchlist to original default basket."""
        default_cfg = self._create_default_config(user_id=user_id)
        self._memory_cache = default_cfg
        self._save_to_s3(default_cfg)
        return default_cfg

    def get_enriched_watchlist(
        self,
        user_id: str = "default_user",
        force_refresh: bool = False,
    ) -> WatchlistResponse:
        """Retrieve watchlist enriched with real-time market quotes."""
        config = self.get_watchlist(user_id=user_id)
        ticker_list = [item.ticker for item in config.items]
        quotes = self.quote_service.get_quotes_batch(ticker_list, force_refresh=force_refresh)
        return WatchlistResponse(
            items=quotes,
            total=len(quotes),
            updated_at=datetime.now(UTC).isoformat(),
        )
