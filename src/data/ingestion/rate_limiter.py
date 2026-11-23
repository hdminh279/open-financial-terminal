import threading
import time


class RateLimiter:
    """Thread-safe Token Bucket rate limiter to respect upstream API request limits."""

    def __init__(self, requests_per_second: float = 10.0, capacity: float | None = None):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be strictly positive")

        self.rate = float(requests_per_second)
        # Capacity must be >= 1.0 so sub-1.0 req/s rates (e.g. 0.9 req/s) can accumulate a token
        self.capacity = float(capacity) if capacity is not None else max(1.0, self.rate)
        if self.capacity < 1.0:
            raise ValueError("capacity must be at least 1.0")
        self.tokens = self.capacity
        self.last_check = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Blocks the calling thread until at least one token is available"""
        with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_check
                self.last_check = now

                # Refill tokens based on elapsed time
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                # Caculate remaning sleep time required to accumulate one toknes
                sleep_needed = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_needed)
