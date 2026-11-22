"""Live Web Search Tool using DuckDuckGo Search (DDGS).

Provides real-time web search capabilities for breaking news, recent events,
and post-filing developments without requiring an external API key ($0 cost).
"""

import html
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Resilient import supporting both new ddgs and legacy duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        DDGS = None  # Handled gracefully at runtime


class WebSearchResultItem(BaseModel):
    """Normalized single web search result item."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Webpage or news article title")
    url: str = Field(..., description="Direct destination URL")
    snippet: str = Field(..., description="Cleaned, truncated excerpt or summary")
    date: Optional[str] = Field(default=None, description="Publication date if available")


class WebSearchResponse(BaseModel):
    """Enveloped response returned by WebSearchTool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original search query executed")
    total_found: int = Field(default=0, description="Number of results retrieved")
    results: List[WebSearchResultItem] = Field(
        default_factory=list,
        description="List of search result items",
    )


class WebSearchTool:
    """Production-grade Web Search Tool wrapping DuckDuckGo Search."""

    def __init__(self, timeout: int = 10, max_retries: int = 2) -> None:
        """Initialize the WebSearchTool.

        Args:
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retry attempts on network error.
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def _clean_snippet(self, raw_text: Optional[str], max_length: int = 350) -> str:
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

    def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Execute web search query via DuckDuckGo Search.

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

        if DDGS is None:
            logger.error("DuckDuckGo Search package (ddgs/duckduckgo_search) is not installed")
            return WebSearchResponse(query=query_str, total_found=0, results=[])

        clamped_max = max(1, min(max_results, 10))

        results: List[WebSearchResultItem] = []
        last_error: Optional[Exception] = None

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

        if last_error and not results:
            logger.error(
                "All %d web search attempts failed for query '%s': %s",
                self.max_retries,
                query_str,
                last_error,
            )

        return WebSearchResponse(
            query=query_str,
            total_found=len(results),
            results=results,
        )
