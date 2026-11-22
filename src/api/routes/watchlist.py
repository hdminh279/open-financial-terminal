"""FastAPI route handlers for dynamic real-time user watchlist and stock quotes."""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from src.data.market.contracts import (
    AddWatchlistRequest,
    StockQuote,
    WatchlistResponse,
)
from src.data.market.service import QuoteService, WatchlistRepository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Watchlist & Quotes"])

# Global repository and quote service instances (singleton per process)
_watchlist_repo: WatchlistRepository | None = None
_quote_service: QuoteService | None = None


def get_watchlist_repository() -> WatchlistRepository:
    """Dependency injector for WatchlistRepository."""
    global _watchlist_repo
    if _watchlist_repo is None:
        _watchlist_repo = WatchlistRepository()
    return _watchlist_repo


def get_quote_service() -> QuoteService:
    """Dependency injector for QuoteService."""
    global _quote_service
    if _quote_service is None:
        _quote_service = QuoteService()
    return _quote_service


@router.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="Get user watchlist with real-time stock quotes",
)
def get_user_watchlist(
    refresh: bool = Query(False, description="Force refresh bypassing TTL cache"),
) -> WatchlistResponse:
    """Retrieve user watchlist enriched with live or cached quotes."""
    repo = get_watchlist_repository()
    return repo.get_enriched_watchlist(force_refresh=refresh)


@router.post(
    "/watchlist",
    response_model=WatchlistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a stock ticker to user watchlist",
)
def add_ticker_to_watchlist(request: AddWatchlistRequest) -> WatchlistResponse:
    """Add a new ticker to user's persistent watchlist and return refreshed quotes."""
    repo = get_watchlist_repository()
    try:
        repo.add_ticker(ticker=request.ticker, market=request.market)
        return repo.get_enriched_watchlist()
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
def remove_ticker_from_watchlist(ticker: str) -> WatchlistResponse:
    """Remove a ticker from the persistent watchlist and return refreshed list."""
    repo = get_watchlist_repository()
    repo.remove_ticker(ticker=ticker)
    return repo.get_enriched_watchlist()


@router.post(
    "/watchlist/reset",
    response_model=WatchlistResponse,
    summary="Reset user watchlist to default basket",
)
def reset_watchlist() -> WatchlistResponse:
    """Reset watchlist back to default multi-market basket (AAPL, MSFT, GOOGL, NVDA, FPT, VNM, HPG)."""
    repo = get_watchlist_repository()
    repo.reset_to_default()
    return repo.get_enriched_watchlist()


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
