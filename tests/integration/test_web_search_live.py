"""Live Integration test for WebSearchTool using actual DuckDuckGo Search.

Marked with @pytest.mark.live so it is skipped during regular offline CI/unit runs.
Run explicitly with: uv run pytest tests/integration/test_web_search_live.py -m live
"""

import pytest

from src.ai.agent.tools.web_search_tool import WebSearchTool


@pytest.mark.live
def test_live_duckduckgo_search_corporate_query() -> None:
    """Verify live DDGS search returns actual results with URLs and titles."""
    tool = WebSearchTool(timeout=10, max_retries=2)
    response = tool.search("Apple Inc SEC 10-K filing", max_results=3)

    assert response.total_found > 0
    assert len(response.results) > 0

    first = response.results[0]
    assert first.title != ""
    assert first.url.startswith("http")
    assert len(first.snippet) > 0
