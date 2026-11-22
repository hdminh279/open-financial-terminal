"""Live Web Search Tool using DuckDuckGo Search (DDGS).

Provides real-time web search capabilities for breaking news, recent events,
and post-filing developments without requiring an external API key ($0 cost).
"""

import html
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None  # Handled gracefully at runtime


class WebSearchResultItem(BaseModel):
    """Normalized single web search result item."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Webpage or news article title")
    url: str = Field(..., description="Direct destination URL")
    snippet: str = Field(..., description="Cleaned, truncated excerpt or summary")
    date: str | None = Field(default=None, description="Publication date if available")


class WebSearchResponse(BaseModel):
    """Enveloped response returned by WebSearchTool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original search query executed")
    total_found: int = Field(default=0, description="Number of results retrieved")
    results: list[WebSearchResultItem] = Field(
        default_factory=list,
        description="List of search result items",
    )
    error: str | None = Field(
        default=None,
        description="Runtime failure detail if search backend is unavailable or failed",
    )


class WebSearchTool:
    """Production-grade Web Search Tool wrapping DuckDuckGo Search with TTL cache and RSS fallback."""

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 2,
        cache_ttl_seconds: int = 300,
        enable_rss_fallback: bool = False,
        max_cache_size: int = 256,
    ) -> None:
        """Initialize the WebSearchTool.

        Args:
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retry attempts on network error.
            cache_ttl_seconds: In-memory TTL cache duration for successful results.
            enable_rss_fallback: Fallback to Google News RSS when DuckDuckGo errors/rate-limits.
            max_cache_size: Maximum number of cached search responses retained in memory.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_seconds = max(0, cache_ttl_seconds)
        self.enable_rss_fallback = enable_rss_fallback
        self.max_cache_size = max(1, int(max_cache_size))
        self._cache: OrderedDict[tuple[str, int], tuple[float, WebSearchResponse]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _clean_snippet(self, raw_text: str | None, max_length: int = 350) -> str:
        """Sanitize raw snippet text by stripping HTML tags and normalizing spaces.

        Args:
            raw_text: Raw snippet string from search engine.
            max_length: Maximum allowed character length.

        Returns:
            Sanitized, readable summary string.
        """
        if not raw_text:
            return ""

        # Unescape HTML entities (&amp;, &quot;, &#39;, etc.)
        cleaned = html.unescape(raw_text)

        # Remove any HTML tags if present
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # Normalize multiple spaces and newlines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) > max_length:
            return cleaned[:max_length].rstrip() + "..."
        return cleaned

    def _search_google_news_rss(self, query_str: str, max_results: int) -> list[WebSearchResultItem]:
        """Fallback live news lookup via Google News RSS (zero-cost, no API key)."""
        has_vi = bool(re.search(r"[\u00C0-\u1EF9]", query_str))
        params = (
            {"q": query_str, "hl": "vi", "gl": "VN", "ceid": "VN:vi"}
            if has_vi
            else {"q": query_str, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        resp = httpx.get(
            "https://news.google.com/rss/search",
            params=params,
            timeout=self.timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items: list[WebSearchResultItem] = []
        for item_el in root.findall(".//item")[:max_results]:
            title = (item_el.findtext("title") or "").strip()
            url = (item_el.findtext("link") or "").strip()
            desc = self._clean_snippet(item_el.findtext("description") or "")
            pub_date = (item_el.findtext("pubDate") or "").strip() or None
            if title and url:
                items.append(
                    WebSearchResultItem(
                        title=title,
                        url=url,
                        snippet=desc or title,
                        date=pub_date,
                    )
                )
        return items

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Execute web search query via DuckDuckGo Search with backoff, cache, and RSS fallback.

        Args:
            query: User search query string.
            max_results: Maximum number of search results to return (1-10).

        Returns:
            WebSearchResponse containing sanitized search result items.
        """
        query_str = (query or "").strip()
        if not query_str:
            logger.warning("Empty search query passed to WebSearchTool.search")
            return WebSearchResponse(query="", total_found=0, results=[])

        clamped_max = max(1, min(max_results, 10))
        cache_key = (query_str.lower(), clamped_max)
        if self.cache_ttl_seconds > 0:
            with self._cache_lock:
                entry = self._cache.get(cache_key)
                if entry is not None:
                    cached_at, cached_resp = entry
                    if (time.monotonic() - cached_at) <= self.cache_ttl_seconds:
                        self._cache.move_to_end(cache_key)
                        return cached_resp
                    del self._cache[cache_key]

        if DDGS is None and not self.enable_rss_fallback:
            logger.error("DuckDuckGo Search package (ddgs/duckduckgo_search) is not installed")
            return WebSearchResponse(
                query=query_str,
                total_found=0,
                results=[],
                error="DuckDuckGo Search package is not installed",
            )

        results: list[WebSearchResultItem] = []
        last_error: Exception | None = None

        if DDGS is not None:
            for attempt in range(1, self.max_retries + 1):
                try:
                    with DDGS(timeout=self.timeout) as ddgs:
                        # Support positional query for both ddgs (query=...) and duckduckgo_search (keywords=...)
                        try:
                            raw_items = ddgs.text(query_str, max_results=clamped_max)
                        except TypeError:
                            raw_items = ddgs.text(keywords=query_str, max_results=clamped_max)  # type: ignore

                        if raw_items:
                            for item in raw_items:
                                title = (item.get("title") or "").strip()
                                url = (item.get("href") or "").strip()
                                raw_body = item.get("body") or ""

                                if not url or not title:
                                    continue

                                snippet = self._clean_snippet(raw_body)
                                results.append(
                                    WebSearchResultItem(
                                        title=title,
                                        url=url,
                                        snippet=snippet,
                                        date=item.get("date"),
                                    )
                                )
                    last_error = None
                    break  # Successful fetch, exit retry loop
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Web search attempt %d/%d failed for query '%s': %s",
                        attempt,
                        self.max_retries,
                        query_str,
                        exc,
                    )
                    if attempt < self.max_retries:
                        time.sleep(min(1.5 * attempt, 4.0))
        else:
            last_error = RuntimeError("DuckDuckGo Search package is not installed")

        if not results and self.enable_rss_fallback:
            try:
                results = self._search_google_news_rss(query_str, clamped_max)
                if results:
                    last_error = None
            except Exception as rss_exc:
                logger.warning("Google News RSS fallback failed for '%s': %s", query_str, rss_exc)
                if last_error is None:
                    last_error = rss_exc

        if last_error and not results:
            logger.error(
                "All %d web search attempts failed for query '%s': %s",
                self.max_retries,
                query_str,
                last_error,
            )
            return WebSearchResponse(
                query=query_str,
                total_found=0,
                results=[],
                error=str(last_error),
            )

        resp = WebSearchResponse(
            query=query_str,
            total_found=len(results),
            results=results,
        )
        if results and self.cache_ttl_seconds > 0:
            with self._cache_lock:
                self._cache[cache_key] = (time.monotonic(), resp)
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.max_cache_size:
                    self._cache.popitem(last=False)
        return resp
