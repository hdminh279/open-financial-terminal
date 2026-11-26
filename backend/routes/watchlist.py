"""FastAPI route handlers for dynamic real-time user watchlist and stock quotes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.deps import get_quote_service, get_watchlist_repository
from backend.security import get_current_user_id
from backend.services.market.contracts import (
    TICKER_RE,
    AddWatchlistRequest,
    Market,
    StockQuote,
    WatchlistResponse,
)
from backend.services.market.service import QuoteService, WatchlistRepository

MAX_QUOTES_BATCH = 25

router = APIRouter(tags=["Watchlist & Quotes"])


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
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> WatchlistResponse:
    """Retrieve paginated user-scoped watchlist enriched with live or cached quotes."""
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
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> WatchlistResponse:
    """Add a new ticker to user's persistent watchlist and return first page of refreshed quotes."""
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
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> WatchlistResponse:
    """Remove a ticker from the persistent watchlist and return first page of refreshed list."""
    repo.remove_ticker(ticker=ticker, user_id=user_id)
    return repo.get_enriched_watchlist(user_id=user_id, page=1, page_size=10)


@router.post(
    "/watchlist/reset",
    response_model=WatchlistResponse,
    summary="Reset user watchlist to default basket",
)
def reset_watchlist(
    user_id: str = Depends(get_current_user_id),
    repo: WatchlistRepository = Depends(get_watchlist_repository),
) -> WatchlistResponse:
    """Reset watchlist back to default multi-market basket (100 SEC issuers plus 50 VN issuers)."""
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
    quote_svc: QuoteService = Depends(get_quote_service),
) -> list[StockQuote]:
    """Retrieve real-time quotes for an ad-hoc list of stock tickers (max 25 symbols)."""
    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid ticker symbols provided",
        )
    if len(symbols) > MAX_QUOTES_BATCH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {MAX_QUOTES_BATCH} tickers may be queried per request.",
        )
    invalid = [s for s in symbols if not TICKER_RE.match(s)]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ticker symbol(s): {', '.join(invalid[:5])}",
        )
    return quote_svc.get_quotes_batch(symbols, force_refresh=refresh)
