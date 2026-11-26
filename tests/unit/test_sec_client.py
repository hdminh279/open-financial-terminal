import httpx
import pytest

from backend.core.rate_limiter import RateLimiter
from pipelines.ingestion.sec_client import SECClient


def test_invalid_user_agent():
    """Verify that empty or malformed User-Agent without email raises ValueError."""
    with pytest.raises(ValueError, match="Invalid SEC User-Agent"):
        SECClient(user_agent="")
    with pytest.raises(ValueError, match="Invalid SEC User-Agent"):
        SECClient(user_agent="MyBotNoEmail")


def test_valid_user_agent_and_headers():
    """Verify that headers are properly configured on initialization."""
    valid_ua = "CorporateFilingIntelligence admin@example.com"
    client = SECClient(user_agent=valid_ua)
    assert client.client.headers["User-Agent"] == valid_ua
    assert "gzip" in client.client.headers["Accept-Encoding"]
    client.close()


def test_rate_limiter_called_on_request(mocker):
    """Verify that rate_limiter.acquire() is invoked before every HTTP request."""
    mock_limiter = mocker.MagicMock(spec=RateLimiter)
    client = SECClient(
        user_agent="TestApp test@example.com",
        rate_limiter=mock_limiter,
    )

    mock_response = httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", "https://api.test"))
    mocker.patch.object(client.client, "get", return_value=mock_response)

    result = client.get_json("https://api.test")
    assert result == {"status": "ok"}
    mock_limiter.acquire.assert_called_once()
    client.close()


def test_retry_on_429_then_success(mocker):
    """Verify exponential backoff retries when encountering HTTP 429 and succeeds subsequently."""
    mock_limiter = mocker.MagicMock(spec=RateLimiter)
    client = SECClient(
        user_agent="TestApp test@example.com",
        rate_limiter=mock_limiter,
    )

    # First attempt: 429 Too Many Requests; Second attempt: 200 OK
    resp_429 = httpx.Response(429, request=httpx.Request("GET", "https://api.test"))
    resp_200 = httpx.Response(200, json={"data": "success"}, request=httpx.Request("GET", "https://api.test"))

    mock_get = mocker.patch.object(client.client, "get", side_effect=[resp_429, resp_200])

    data = client.get_json("https://api.test")
    assert data == {"data": "success"}
    assert mock_get.call_count == 2
    assert mock_limiter.acquire.call_count == 2
    client.close()


def test_non_retryable_404_fails_fast(mocker):
    """Verify that non-retryable 404 client error raises immediately without retrying."""
    client = SECClient(user_agent="TestApp test@example.com")
    resp_404 = httpx.Response(404, request=httpx.Request("GET", "https://api.test"))

    mock_get = mocker.patch.object(client.client, "get", return_value=resp_404)

    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://api.test")

    # Must fail on the first attempt without retrying
    assert mock_get.call_count == 1
    client.close()
