"""FastAPI route handlers for dynamic real-time user watchlist and stock quotes."""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.security import get_current_user_id
from src.data.market.contracts import (
    AddWatchlistRequest,
    Market,
    StockQuote,
    WatchlistResponse,
)
from src.data.market.service import QuoteService, WatchlistRepository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Watchlist & Quotes"])

# Global repository and quote service instances (singleton per process)
_watchlist_repo: WatchlistRepository | None = None
_quote_service: QuoteService | None = None
_WATCHLIST_LOCK = threading.Lock()


def get_watchlist_repository() -> WatchlistRepository:
    """Dependency injector for WatchlistRepository."""
    global _watchlist_repo
    if _watchlist_repo is None:
        with _WATCHLIST_LOCK:
            if _watchlist_repo is None:
                _watchlist_repo = WatchlistRepository()
    return _watchlist_repo


def get_quote_service() -> QuoteService:
    """Dependency injector for QuoteService."""
    global _quote_service
    if _quote_service is None:
        with _WATCHLIST_LOCK:
            if _quote_service is None:
                _quote_service = QuoteService()
    return _quote_service


@router.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="Get user watchlist with paginated real-time stock quotes",
)
def get_user_watchlist(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of tickers per page"),
    market: Market | None = Query(None, description="Optional market filter (US or VN)"),
    refresh: bool = Query(False, description="Force refresh bypassing TTL cache"),
    user_id: str = Depends(get_current_user_id),
) -> WatchlistResponse:
    """Retrieve paginated user-scoped watchlist enriched with live or cached quotes."""
    repo = get_watchlist_repository()
    return repo.get_enriched_watchlist(
        user_id=user_id,
        force_refresh=refresh,
        page=page,
        page_size=page_size,
        market=market,
    )


@router.post(
    "/watchlist",
    response_model=WatchlistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a stock ticker to user watchlist",
)
def add_ticker_to_watchlist(
    request: AddWatchlistRequest,
    user_id: str = Depends(get_current_user_id),
) -> WatchlistResponse:
    """Add a new ticker to user's persistent watchlist and return first page of refreshed quotes."""
    repo = get_watchlist_repository()
    try:
        repo.add_ticker(ticker=request.ticker, market=request.market, user_id=user_id)
        return repo.get_enriched_watchlist(user_id=user_id, page=1, page_size=10)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.delete(
    "/watchlist/{ticker}",
    response_model=WatchlistResponse,
    summary="Remove a stock ticker from user watchlist",
)
def remove_ticker_from_watchlist(
    ticker: str,
    user_id: str = Depends(get_current_user_id),
) -> WatchlistResponse:
    """Remove a ticker from the persistent watchlist and return first page of refreshed list."""
    repo = get_watchlist_repository()
    repo.remove_ticker(ticker=ticker, user_id=user_id)
    return repo.get_enriched_watchlist(user_id=user_id, page=1, page_size=10)


@router.post(
    "/watchlist/reset",
    response_model=WatchlistResponse,
    summary="Reset user watchlist to default basket",
)
def reset_watchlist(
    user_id: str = Depends(get_current_user_id),
) -> WatchlistResponse:
    """Reset watchlist back to default multi-market basket (100 SEC issuers plus 50 VN issuers)."""
    repo = get_watchlist_repository()
    repo.reset_to_default(user_id=user_id)
    return repo.get_enriched_watchlist(user_id=user_id, page=1, page_size=10)


@router.get(
    "/quotes",
    response_model=list[StockQuote],
    summary="Fetch live quotes for arbitrary tickers",
)
def get_stock_quotes(
    tickers: str = Query(..., description="Comma-separated ticker list, e.g. AAPL,FPT,MSFT"),
    refresh: bool = Query(False, description="Force refresh bypassing TTL cache"),
) -> list[StockQuote]:
    """Retrieve real-time quotes for an ad-hoc list of stock tickers."""
    quote_svc = get_quote_service()
    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid ticker symbols provided",
        )
    return quote_svc.get_quotes_batch(symbols, force_refresh=refresh)
