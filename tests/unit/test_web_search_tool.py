"""Unit tests for WebSearchTool (DuckDuckGo Search Adapter).

Tests offline with mocked DDGS client to ensure 100% deterministic, zero-cost,
sub-second execution and robust error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ai.agent.tools.web_search_tool import (
    WebSearchResponse,
    WebSearchResultItem,
    WebSearchTool,
)


@pytest.fixture
def web_tool() -> WebSearchTool:
    """Fixture providing a WebSearchTool instance with fast timeout."""
    return WebSearchTool(timeout=2, max_retries=2)


def test_clean_snippet(web_tool: WebSearchTool) -> None:
    """Verify HTML tag stripping, entity unescaping, and length truncation."""
    raw_html = "<p>Apple Inc. announced &amp; released <b>new AI features</b> &quot;Apple Intelligence&quot;.</p>"
    cleaned = web_tool._clean_snippet(raw_html, max_length=100)

    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&quot;" not in cleaned
    assert 'Apple Inc. announced & released new AI features "Apple Intelligence".' in cleaned


def test_clean_snippet_truncation(web_tool: WebSearchTool) -> None:
    """Verify snippet length truncation with ellipsis."""
    long_text = "word " * 100
    cleaned = web_tool._clean_snippet(long_text, max_length=50)
    assert len(cleaned) <= 53  # 50 chars + "..."
    assert cleaned.endswith("...")


def test_search_empty_query(web_tool: WebSearchTool) -> None:
    """Verify empty query returns empty response without invoking DDGS."""
    res = web_tool.search("")
    assert res.query == ""
    assert res.total_found == 0
    assert res.results == []


@patch("src.ai.agent.tools.web_search_tool.DDGS")
def test_search_success(mock_ddgs_cls: MagicMock, web_tool: WebSearchTool) -> None:
    """Verify search successfully parses DDGS text outputs into structured models."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_ddgs_instance.text.return_value = [
        {
            "title": "FPT Partners with NVIDIA",
            "href": "https://example.com/fpt-nvidia",
            "body": "FPT Corporation announced a comprehensive partnership with NVIDIA to build an AI factory.",
            "date": "2024-04-23",
        },
        {
            "title": "Apple Antitrust Case Update",
            "href": "https://example.com/apple-doj",
            "body": "The US Department of Justice filed a landmark antitrust lawsuit against Apple.",
            "date": "2024-03-21",
        },
    ]

    response = web_tool.search("FPT NVIDIA AI factory", max_results=5)

    assert isinstance(response, WebSearchResponse)
    assert response.query == "FPT NVIDIA AI factory"
    assert response.total_found == 2
    assert len(response.results) == 2

    first = response.results[0]
    assert isinstance(first, WebSearchResultItem)
    assert first.title == "FPT Partners with NVIDIA"
    assert first.url == "https://example.com/fpt-nvidia"
    assert "AI factory" in first.snippet
    assert first.date == "2024-04-23"


@patch("src.ai.agent.tools.web_search_tool.DDGS")
def test_search_empty_results(mock_ddgs_cls: MagicMock, web_tool: WebSearchTool) -> None:
    """Verify search handles empty results from DDGS gracefully."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.return_value = []

    response = web_tool.search("xyznonexistentterm12345")
    assert response.total_found == 0
    assert response.results == []


@patch("src.ai.agent.tools.web_search_tool.DDGS")
def test_search_exception_handled_gracefully(mock_ddgs_cls: MagicMock, web_tool: WebSearchTool) -> None:
    """Verify network exceptions or rate limits are caught and return empty results without crashing."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.side_effect = Exception("Rate limit exceeded 429")

    response = web_tool.search("Apple latest news")
    assert response.total_found == 0
    assert response.results == []


def test_search_when_ddgs_uninstalled(monkeypatch: pytest.MonkeyPatch, web_tool: WebSearchTool) -> None:
    """Verify graceful failure when DDGS library is absent."""
    import src.ai.agent.tools.web_search_tool as wst

    monkeypatch.setattr(wst, "DDGS", None)

    response = web_tool.search("Apple stock")
    assert response.total_found == 0
    assert response.results == []


def test_normalize_news_query_strips_conversational_filler() -> None:
    """Verify Vietnamese and English conversational filler is stripped for news search."""
    assert (
        WebSearchTool._normalize_news_query("Cho tôi tin tức tài chính mới nhất của việt nam ngày hôm nay")
        == "tài chính việt nam"
    )
    assert WebSearchTool._normalize_news_query("Latest breaking news about NVIDIA today") == "NVIDIA"


@patch("src.ai.agent.tools.web_search_tool.DDGS")
def test_news_query_prefers_rss_and_ddgs_news_over_text(mock_ddgs_cls: MagicMock) -> None:
    """Verify news-like queries prioritize dated RSS and ddgs.news() articles over generic ddgs.text()."""
    tool = WebSearchTool(timeout=2, max_retries=1, enable_rss_fallback=True)
    mock_ddgs_instance = MagicMock()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.news.return_value = [
        {
            "title": "VN-Index tăng điểm phiên chiều",
            "url": "https://vnexpress.net/vn-index-tang-diem",
            "body": "Thị trường chứng khoán Việt Nam khởi sắc.",
            "date": "2026-11-23T08:00:00+00:00",
        }
    ]
    with patch.object(
        tool,
        "_search_google_news_rss",
        return_value=[
            WebSearchResultItem(
                title="Trái phiếu chính phủ huy động vốn dài hạn",
                url="https://news.google.com/rss/articles/sample1",
                snippet="Diễn biến thị trường trái phiếu chính phủ.",
                date="Mon, 23 Nov 2026 09:25:34 GMT",
            )
        ],
    ):
        resp = tool.search("Cho tôi tin tức tài chính mới nhất của việt nam ngày hôm nay", max_results=5)

    assert resp.total_found == 2
    assert resp.results[0].url == "https://news.google.com/rss/articles/sample1"
    assert resp.results[1].url == "https://vnexpress.net/vn-index-tang-diem"
    mock_ddgs_instance.text.assert_not_called()
