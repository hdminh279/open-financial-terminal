import logging
from typing import Any, Dict, Optional
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from src.data.ingestion.rate_limiter import RateLimiter
logger = logging.getLogger(__name__)

def _is_retryable_error(exception: BaseException) -> bool:
    """Determine whether an error is transient and safe to retry."""
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on rate limiting (429) and transient server errors (5xx)
        return exception.response.status_code in {429, 500, 502, 503, 504}
    return False


class SECClient:
    """Resilient HTTP client for SEC EDGAR APIs with mandatory headers and rate limiting."""
    def __init__(
        self,
        user_agent: str,
        rate_limiter: Optional[RateLimiter] = None,
        timeout_seconds: float = 15.0,
    ):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "Invalid SEC User-Agent. SEC requires format: '<App_Name> <admin_contact@domain.com>'"
            )
        self.user_agent = user_agent.strip()
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_second=10.0)
        self.client = httpx.Client(
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=timeout_seconds,
        )

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(5),
        wait = wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        reraise=True,
    )

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Send a rate_limited GET request with exponential backoff on transient errors."""
        self.rate_limiter.acquire()
        logger.debug("Requesting URL: %s with params: %s", url, params)

        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convenience wrapper that returns the parsed JSON payload."""
        response = self.get(url, params=params)
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self.client.close()

    def __enter__(self) -> "SECClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


