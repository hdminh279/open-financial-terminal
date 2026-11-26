"""Domain contracts and Pydantic data schemas for market quotes and user watchlist."""

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

TICKER_RE = re.compile(r"^[A-Z0-9.-]{1,12}$")


class Market(str, Enum):
    """Supported market jurisdictions."""

    US = "US"
    VN = "VN"


class QuoteStatus(str, Enum):
    """Status classification of the returned stock quote."""

    REAL_TIME = "REAL_TIME"
    DELAYED = "DELAYED"
    CLOSED = "CLOSED"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class StockQuote(BaseModel):
    """Normalized real-time or end-of-day market quote for a single entity."""

    ticker: str = Field(..., description="Canonical uppercase ticker symbol")
    company_name: str | None = Field(None, description="Human-readable company name")
    market: Market = Field(..., description="Market jurisdiction: US or VN")
    price: float | None = Field(None, description="Latest traded or matched price. None if unavailable.")
    change: float | None = Field(None, description="Absolute price difference against reference or prior close")
    percent_change: float | None = Field(None, description="Percentage change (-100 to +infinity)")
    currency: str = Field("USD", description="Currency code (USD or VND)")
    day_high: float | None = Field(None, description="Intraday high price")
    day_low: float | None = Field(None, description="Intraday low price")
    day_open: float | None = Field(None, description="Market session open price")
    previous_close: float | None = Field(None, description="Prior trading session close price")
    volume: int | None = Field(None, description="Accumulated trading session volume")
    as_of: str = Field(..., description="ISO 8601 UTC timestamp of quote snapshot")
    status: QuoteStatus = Field(QuoteStatus.REAL_TIME, description="Operational status of the quote")
    is_delayed: bool = Field(False, description="Flag indicating delayed market data")

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        """Strip whitespace and enforce uppercase canonical ticker."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty")
        return v.strip().upper()

    @field_validator("company_name", mode="before")
    @classmethod
    def sanitize_company_name(cls, v: str | None) -> str | None:
        """Strip parenthesized text and parentheses per UI standards."""
        if v and isinstance(v, str):
            cleaned = re.sub(r"\s*\(.*?\)", "", v).strip()
            return cleaned if cleaned else v
        return v


class WatchlistItem(BaseModel):
    """Single item entry within a user's persistent watchlist."""

    ticker: str = Field(..., description="Canonical ticker symbol")
    market: Market = Field(..., description="Market classification (US or VN)")
    added_at: str = Field(..., description="ISO 8601 UTC timestamp when ticker was added")

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        """Strip whitespace, uppercase, and enforce strict ticker character bounds."""
        symbol = (v or "").strip().upper()
        if not symbol:
            raise ValueError("Ticker symbol cannot be empty")
        if not TICKER_RE.match(symbol):
            raise ValueError("Ticker symbol must be 1-12 alphanumeric, dot, or hyphen characters")
        return symbol


class WatchlistConfig(BaseModel):
    """Persistent watchlist schema stored in MinIO (cfi-app-state/watchlist/{user_id}.json)."""

    version: str = Field("1.0", description="Configuration schema version")
    user_id: str = Field("default_user", description="Owner identifier (defaults to single user)")
    items: list[WatchlistItem] = Field(default_factory=list, description="Ordered list of watched tickers")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of last modification")


class WatchlistResponse(BaseModel):
    """Enriched watchlist response payload containing paginated live quotes for the client."""

    items: list[StockQuote] = Field(default_factory=list, description="Live quote snapshots for watched tickers")
    total: int = Field(0, description="Total count of items in watchlist (after optional market filter)")
    page: int = Field(1, ge=1, description="Current 1-based page number")
    page_size: int = Field(20, ge=1, description="Maximum items per page")
    total_pages: int = Field(1, ge=1, description="Total number of pages")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of generation")


class AddWatchlistRequest(BaseModel):
    """Request payload to add a ticker to the active watchlist."""

    ticker: str = Field(..., description="Ticker symbol to add (e.g. AAPL, FPT)")
    market: Market | None = Field(None, description="Optional market hint (auto-detected if omitted)")

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        """Strip whitespace, uppercase, and enforce strict ticker character bounds."""
        symbol = (v or "").strip().upper()
        if not symbol:
            raise ValueError("Ticker symbol cannot be empty")
        if not TICKER_RE.match(symbol):
            raise ValueError("Ticker symbol must be 1-12 alphanumeric, dot, or hyphen characters")
        return symbol
