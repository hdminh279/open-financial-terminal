import time

import pytest

from src.data.ingestion.rate_limiter import RateLimiter


def test_invalid_rate():
    """Verify that zero or negative rates raise ValueError."""
    with pytest.raises(ValueError, match="requests_per_second must be strictly positive"):
        RateLimiter(requests_per_second=0)
    with pytest.raises(ValueError, match="requests_per_second must be strictly positive"):
        RateLimiter(requests_per_second=-1.5)


def test_rate_limiter_burst():
    """Verify that initial available tokens can be consumed immediately with near-zero latency."""
    limiter = RateLimiter(requests_per_second=5.0)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"Burst should be instantaneous, but took {elapsed:.4f}s"


def test_rate_limiter_throttling():
    """Verify that requests exceeding the bucket capacity are properly throttled."""
    rate = 10.0
    limiter = RateLimiter(requests_per_second=rate)

    start = time.monotonic()
    # Consume 15 tokens: first 10 are immediate, remaining 5 require 0.1s each
    for _ in range(15):
        limiter.acquire()
    elapsed = time.monotonic() - start

    # Expected: 5 throttled requests require at least 5 * 0.1s = 0.5s (allow small tolerance 0.4s)
    assert elapsed >= 0.4, f"Rate limiter failed to throttle requests, took only {elapsed:.4f}s"


def test_rate_limiter_try_acquire_non_blocking():
    """Verify try_acquire consumes tokens immediately and returns False without sleeping when exhausted."""
    limiter = RateLimiter(requests_per_second=10.0, capacity=2.0)

    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True

    start = time.monotonic()
    assert limiter.try_acquire() is False
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"try_acquire must not block, but took {elapsed:.4f}s"

    # Wait for one token refill (0.1s at 10 req/s) and verify acquisition succeeds again
    time.sleep(0.12)
    assert limiter.try_acquire() is True
