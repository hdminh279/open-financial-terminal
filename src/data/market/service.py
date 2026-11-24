"""Market Quote Service and Persistent Watchlist Repository Engine.

Combines Token Bucket rate limiting, in-memory TTL caching, and MinIO S3 app-state
persistence (cfi-app-state/watchlist/{user_id}.json) with fallback in-memory caching.
"""

import concurrent.futures
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime

from src.api.security import validate_user_id
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

    DEFAULT_FINNHUB_RPS = 0.9
    DEFAULT_FINNHUB_BURST = 5.0
    DEFAULT_VN_RPS = 5.0
    DEFAULT_STALE_TTL_SECONDS = 1800.0
    DEFAULT_MAX_CACHE_SIZE = 500

    def __init__(
        self,
        factory: QuoteProviderFactory | None = None,
        ttl_seconds: float = 300.0,
        rate_limiter: RateLimiter | None = None,
        finnhub_rate_limiter: RateLimiter | None = None,
        stale_ttl_seconds: float | None = None,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        """Initialize QuoteService.

        Args:
            factory: Optional QuoteProviderFactory. Defaults to standard production factory.
            ttl_seconds: Fresh TTL for cached quotes in seconds (default 300s).
            rate_limiter: Optional shared Token Bucket RateLimiter for VN/custom callers.
            finnhub_rate_limiter: Optional RateLimiter enforcing Finnhub free tier (<= 0.9 req/s).
            stale_ttl_seconds: Extended stale-while-revalidate window in seconds (default 1800s).
            max_cache_size: Maximum number of cached tickers kept in the LRU cache (default 500).
        """
        self.factory = factory or QuoteProviderFactory()
        self.ttl_seconds = ttl_seconds
        self.stale_ttl_seconds = (
            float(stale_ttl_seconds)
            if stale_ttl_seconds is not None
            else max(float(ttl_seconds) * 6.0, self.DEFAULT_STALE_TTL_SECONDS)
        )
        self.max_cache_size = max(1, int(max_cache_size))
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_second=self.DEFAULT_VN_RPS)
        self.finnhub_rate_limiter = (
            finnhub_rate_limiter
            or rate_limiter
            or RateLimiter(requests_per_second=self.DEFAULT_FINNHUB_RPS, capacity=self.DEFAULT_FINNHUB_BURST)
        )
        self._cache: OrderedDict[str, tuple[StockQuote, float, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _stale_fallback(
        cached_entry: tuple[StockQuote, float, float] | None,
        now: float,
    ) -> StockQuote | None:
        """Return a delayed stale-while-revalidate quote if a valid cached price is within stale TTL."""
        if cached_entry is None:
            return None
        stale_quote, _fresh_until, stale_until = cached_entry
        if stale_quote.price is not None and now < stale_until:
            return stale_quote.model_copy(update={"is_delayed": True, "status": QuoteStatus.DELAYED})
        return None

    def get_quote(self, ticker: str, force_refresh: bool = False) -> StockQuote:
        """Retrieve quote with bounded LRU TTL caching, non-blocking rate limiting, and SWR fallback."""
        symbol = ticker.strip().upper()
        now = time.time()

        cached_entry: tuple[StockQuote, float, float] | None = None
        with self._cache_lock:
            cached_entry = self._cache.get(symbol)
            if cached_entry is not None:
                self._cache.move_to_end(symbol)
            if not force_refresh and cached_entry is not None:
                quote, fresh_until, _stale_until = cached_entry
                if now < fresh_until:
                    logger.debug("Quote cache hit for %s (expires in %.1fs)", symbol, fresh_until - now)
                    return quote

        provider = self.factory.get_provider(symbol)
        limiter = self.finnhub_rate_limiter if provider.market == Market.US else self.rate_limiter
        if not limiter.try_acquire():
            stale = self._stale_fallback(cached_entry, now)
            if stale is not None:
                logger.info("Serving stale-while-revalidate quote for %s (rate limiter exhausted)", symbol)
                return stale
            logger.warning("Quote rate limiter exhausted for %s; returning non-blocking error", symbol)
            return StockQuote(
                ticker=symbol,
                market=provider.market,
                price=None,
                change=None,
                percent_change=None,
                as_of=datetime.now(UTC).isoformat(),
                status=QuoteStatus.ERROR,
            )

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

        # Never cache transient errors / HTTP 429 (price=None, status=ERROR) for 300s
        if quote.status == QuoteStatus.ERROR:
            stale = self._stale_fallback(cached_entry, now)
            if stale is not None:
                logger.info("Serving stale-while-revalidate quote for %s after upstream error", symbol)
                return stale
            return quote

        with self._cache_lock:
            self._cache[symbol] = (quote, now + self.ttl_seconds, now + self.stale_ttl_seconds)
            self._cache.move_to_end(symbol)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
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
    """Repository managing per-user persistent watchlists in MinIO S3 with memory fallback."""

    BUCKET_NAME = "cfi-app-state"
    KEY_PREFIX = "watchlist"
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
        self._memory_cache: dict[str, WatchlistConfig] = {}
        self._lock = threading.Lock()
        self.quote_service = quote_service or QuoteService()

    @classmethod
    def _storage_key(cls, user_id: str) -> str:
        uid = validate_user_id(user_id)
        return f"{cls.KEY_PREFIX}/{uid}.json"

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
        uid = validate_user_id(user_id)
        now = datetime.now(UTC).isoformat()
        items = [WatchlistItem(ticker=ticker, market=market, added_at=now) for ticker, market in self.DEFAULT_TICKERS]
        return WatchlistConfig(
            version="1.0",
            user_id=uid,
            items=items,
            updated_at=now,
        )

    def _load_from_s3(self, user_id: str = "default_user") -> WatchlistConfig | None:
        """Attempt to load user-scoped watchlist JSON from MinIO S3 app-state bucket."""
        s3 = self._init_s3()
        if not s3:
            return None

        key = self._storage_key(user_id)
        try:
            if not s3.object_exists(self.BUCKET_NAME, key):
                logger.info("Watchlist not found in S3 (%s/%s); creating default.", self.BUCKET_NAME, key)
                default_cfg = self._create_default_config(user_id=user_id)
                self._save_to_s3(default_cfg)
                return default_cfg

            raw_bytes = s3.download_bytes(self.BUCKET_NAME, key)
            data = json.loads(raw_bytes.decode("utf-8"))
            return WatchlistConfig(**data)
        except Exception as exc:
            logger.error("Failed to load watchlist from MinIO S3: %s", exc)
            return None

    def _save_to_s3(self, config: WatchlistConfig) -> bool:
        """Persist user-scoped watchlist JSON to MinIO S3 app-state bucket."""
        s3 = self._init_s3()
        if not s3:
            return False

        key = self._storage_key(config.user_id)
        try:
            s3.ensure_buckets_exist([self.BUCKET_NAME])
            payload = config.model_dump_json(indent=2).encode("utf-8")
            s3.upload_bytes(
                bucket=self.BUCKET_NAME,
                key=key,
                data=payload,
                content_type="application/json",
                metadata={"user_id": config.user_id, "updated_at": config.updated_at},
            )
            return True
        except Exception as exc:
            logger.error("Failed to save watchlist to MinIO S3: %s", exc)
            return False

    def get_watchlist(self, user_id: str = "default_user") -> WatchlistConfig:
        """Retrieve user-scoped watchlist configuration."""
        uid = validate_user_id(user_id)
        with self._lock:
            if uid in self._memory_cache:
                return self._memory_cache[uid]

        persisted = self._load_from_s3(user_id=uid)
        if persisted is not None:
            with self._lock:
                self._memory_cache[uid] = persisted
            return persisted

        default_cfg = self._create_default_config(user_id=uid)
        with self._lock:
            self._memory_cache[uid] = default_cfg
        return default_cfg

    def add_ticker(
        self,
        ticker: str,
        market: Market | None = None,
        user_id: str = "default_user",
    ) -> WatchlistConfig:
        """Add a ticker symbol to the user's watchlist."""
        uid = validate_user_id(user_id)
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Ticker symbol cannot be empty")

        config = self.get_watchlist(user_id=uid)

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
        config.items.insert(0, new_item)
        config.updated_at = now

        with self._lock:
            self._memory_cache[uid] = config
        self._save_to_s3(config)
        return config

    def remove_ticker(self, ticker: str, user_id: str = "default_user") -> WatchlistConfig:
        """Remove a ticker symbol from the user's watchlist."""
        uid = validate_user_id(user_id)
        symbol = ticker.strip().upper()
        config = self.get_watchlist(user_id=uid)

        initial_len = len(config.items)
        config.items = [item for item in config.items if item.ticker != symbol]

        if len(config.items) != initial_len:
            now = datetime.now(UTC).isoformat()
            config.updated_at = now
            with self._lock:
                self._memory_cache[uid] = config
            self._save_to_s3(config)

        return config

    def reset_to_default(self, user_id: str = "default_user") -> WatchlistConfig:
        """Reset user watchlist to original default basket."""
        uid = validate_user_id(user_id)
        default_cfg = self._create_default_config(user_id=uid)
        with self._lock:
            self._memory_cache[uid] = default_cfg
        self._save_to_s3(default_cfg)
        return default_cfg

    def get_enriched_watchlist(
        self,
        user_id: str = "default_user",
        force_refresh: bool = False,
        page: int = 1,
        page_size: int | None = None,
        market: Market | None = None,
    ) -> WatchlistResponse:
        """Retrieve paginated user watchlist enriched with real-time market quotes."""
        config = self.get_watchlist(user_id=user_id)
        items = config.items
        if market is not None:
            items = [item for item in items if item.market == market]

        total = len(items)
        if page_size is not None:
            eff_page_size = max(1, int(page_size))
            total_pages = max(1, math.ceil(total / eff_page_size))
            current_page = min(max(1, int(page)), total_pages)
            start = (current_page - 1) * eff_page_size
            page_items = items[start : start + eff_page_size]
        else:
            eff_page_size = max(1, total)
            total_pages = 1
            current_page = 1
            page_items = items

        ticker_list = [item.ticker for item in page_items]
        quotes = self.quote_service.get_quotes_batch(ticker_list, force_refresh=force_refresh)
        return WatchlistResponse(
            items=quotes,
            total=total,
            page=current_page,
            page_size=eff_page_size,
            total_pages=total_pages,
            updated_at=datetime.now(UTC).isoformat(),
        )
