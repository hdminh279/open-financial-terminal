"""Domain contracts and Pydantic data schemas for market quotes and user watchlist."""

import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


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
    company_name: Optional[str] = Field(None, description="Human-readable company name")
    market: Market = Field(..., description="Market jurisdiction: US or VN")
    price: Optional[float] = Field(None, description="Latest traded or matched price. None if unavailable.")
    change: Optional[float] = Field(None, description="Absolute price difference against reference or prior close")
    percent_change: Optional[float] = Field(None, description="Percentage change (-100 to +infinity)")
    currency: str = Field("USD", description="Currency code (USD or VND)")
    day_high: Optional[float] = Field(None, description="Intraday high price")
    day_low: Optional[float] = Field(None, description="Intraday low price")
    day_open: Optional[float] = Field(None, description="Market session open price")
    previous_close: Optional[float] = Field(None, description="Prior trading session close price")
    volume: Optional[int] = Field(None, description="Accumulated trading session volume")
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
    def sanitize_company_name(cls, v: Optional[str]) -> Optional[str]:
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
        """Strip whitespace and enforce uppercase canonical ticker."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty")
        return v.strip().upper()


class WatchlistConfig(BaseModel):
    """Persistent watchlist schema stored in Lakehouse (sec-silver/user_settings/watchlist.json)."""

    version: str = Field("1.0", description="Configuration schema version")
    user_id: str = Field("default_user", description="Owner identifier (defaults to single user)")
    items: List[WatchlistItem] = Field(default_factory=list, description="Ordered list of watched tickers")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of last modification")


class WatchlistResponse(BaseModel):
    """Enriched watchlist response payload containing live quotes for the client."""

    items: List[StockQuote] = Field(default_factory=list, description="Live quote snapshots for watched tickers")
    total: int = Field(0, description="Total count of items in watchlist")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of generation")


class AddWatchlistRequest(BaseModel):
    """Request payload to add a ticker to the active watchlist."""

    ticker: str = Field(..., description="Ticker symbol to add (e.g. AAPL, FPT)")
    market: Optional[Market] = Field(None, description="Optional market hint (auto-detected if omitted)")

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        """Strip whitespace and enforce uppercase canonical ticker."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty")
        return v.strip().upper()
