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

    MAX_RSS_RESPONSE_BYTES = 2 * 1024 * 1024
    _UNSAFE_XML_RE = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

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
        if len(resp.content) > self.MAX_RSS_RESPONSE_BYTES:
            logger.warning("Google News RSS payload exceeded %d bytes; aborting parse.", self.MAX_RSS_RESPONSE_BYTES)
            return []
        xml_text = resp.text
        if self._UNSAFE_XML_RE.search(xml_text):
            logger.warning("Rejected Google News RSS payload containing DTD/ENTITY declarations.")
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse Google News RSS XML: %s", exc)
            return []
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

    _NEWS_QUERY_RE = re.compile(
        r"\b(?:tin\s+tức|bản\s+tin|mới\s+nhất|hôm\s+nay|gần\s+đây|news|latest|breaking|today|recent|headlines?)\b",
        re.IGNORECASE,
    )
    _NEWS_FILLER_RE = re.compile(
        r"\b(?:cho\s+tôi|cho\s+mình|hãy|xem|tìm|cập\s+nhật|tin\s+tức|bản\s+tin|mới\s+nhất|ngày\s+hôm\s+nay|hôm\s+nay|gần\s+đây|về|của"
        r"|please|give\s+me|find|latest|breaking|today'?s|today|recent|news|about|on|for)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_news_query(cls, query_str: str) -> bool:
        """Return True when the search query targets current news or headlines."""
        return bool(cls._NEWS_QUERY_RE.search(query_str))

    @classmethod
    def _normalize_news_query(cls, query_str: str) -> str:
        """Strip conversational filler so news search engines match topic keywords."""
        cleaned = cls._NEWS_FILLER_RE.sub(" ", query_str)
        cleaned = re.sub(r"[,.;:!?]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if len(cleaned) >= 3 else query_str

    @staticmethod
    def _merge_unique_results(
        primary: list[WebSearchResultItem],
        secondary: list[WebSearchResultItem],
        max_results: int,
    ) -> list[WebSearchResultItem]:
        """Combine result lists while deduplicating by URL and normalized title."""
        merged: list[WebSearchResultItem] = []
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        for item in [*primary, *secondary]:
            norm_title = item.title.strip().lower()
            if item.url in seen_urls or norm_title in seen_titles:
                continue
            seen_urls.add(item.url)
            seen_titles.add(norm_title)
            merged.append(item)
            if len(merged) >= max_results:
                break
        return merged

    def _search_ddgs_news(self, query_str: str, max_results: int) -> list[WebSearchResultItem]:
        """Execute DuckDuckGo News search returning dated news articles."""
        if DDGS is None:
            return []
        with DDGS(timeout=self.timeout) as ddgs:
            if not hasattr(ddgs, "news"):
                return []
            try:
                raw_items = ddgs.news(query_str, max_results=max_results)
            except TypeError:
                raw_items = ddgs.news(keywords=query_str, max_results=max_results)  # type: ignore

            items: list[WebSearchResultItem] = []
            for item in raw_items or []:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                url = (item.get("url") or item.get("href") or "").strip()
                raw_body = item.get("body") or item.get("excerpt") or ""
                if not url or not title:
                    continue
                items.append(
                    WebSearchResultItem(
                        title=title,
                        url=url,
                        snippet=self._clean_snippet(raw_body) or title,
                        date=item.get("date"),
                    )
                )
            return items

    def _search_ddgs_text(self, query_str: str, max_results: int) -> list[WebSearchResultItem]:
        """Execute standard DuckDuckGo web search."""
        if DDGS is None:
            raise RuntimeError("DuckDuckGo Search package is not installed")
        with DDGS(timeout=self.timeout) as ddgs:
            try:
                raw_items = ddgs.text(query_str, max_results=max_results)
            except TypeError:
                raw_items = ddgs.text(keywords=query_str, max_results=max_results)  # type: ignore

            items: list[WebSearchResultItem] = []
            for item in raw_items or []:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                url = (item.get("href") or "").strip()
                raw_body = item.get("body") or ""
                if not url or not title:
                    continue
                items.append(
                    WebSearchResultItem(
                        title=title,
                        url=url,
                        snippet=self._clean_snippet(raw_body),
                        date=item.get("date"),
                    )
                )
            return items

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Execute web search query via DuckDuckGo Search with backoff, news routing, and RSS."""
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
        is_news_mode = self.enable_rss_fallback and self._is_news_query(query_str)

        if is_news_mode:
            news_query = self._normalize_news_query(query_str)
            rss_items: list[WebSearchResultItem] = []
            ddgs_news_items: list[WebSearchResultItem] = []
            try:
                rss_items = self._search_google_news_rss(news_query, clamped_max)
            except Exception as rss_exc:
                last_error = rss_exc
                logger.warning("Google News RSS search failed for '%s': %s", news_query, rss_exc)
            try:
                ddgs_news_items = self._search_ddgs_news(news_query, clamped_max)
            except Exception as news_exc:
                if last_error is None:
                    last_error = news_exc
                logger.warning("DDGS news search failed for '%s': %s", news_query, news_exc)

            results = self._merge_unique_results(rss_items, ddgs_news_items, clamped_max)
            if results:
                last_error = None

        if not results and DDGS is not None:
            for attempt in range(1, self.max_retries + 1):
                try:
                    results = self._search_ddgs_text(query_str, clamped_max)
                    last_error = None
                    break
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
        elif not results and DDGS is None:
            last_error = last_error or RuntimeError("DuckDuckGo Search package is not installed")

        if not results and self.enable_rss_fallback and not is_news_mode:
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
