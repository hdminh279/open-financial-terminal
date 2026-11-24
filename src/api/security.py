"""API security middleware: IP Token Bucket rate limiting, CORS config, signed cookies, and security headers."""

import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.financial.response_templates import detect_query_language

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_override",
        re.compile(
            r"(?:\[(?:system|instruction|admin|developer)\s*override\]"
            r"|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|filters?|prompts?)"
            r"|disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|rules?)"
            r"|bỏ\s+qua\s+(?:mọi|tất\s+cả|toàn\s+bộ)\s+(?:quy\s+tắc|bộ\s+lọc|chỉ\s+dẫn|nguyên\s+tắc))",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"(?:(?:reveal|print|show|output|dump|leak|expose|xuất|in\s+ra|tiết\s+lộ)"
            r".{0,40}(?:system\s*prompt|hidden\s*prompt|\.env\b|api[_\s]?key|secret[_\s]?key|environment\s*variables?))",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak",
        re.compile(r"\b(?:jailbreak|dan\s+mode|developer\s+mode\s+enabled)\b", re.IGNORECASE),
    ),
]


def sanitize_query_input(query: str) -> str:
    """Strip unsafe null bytes and ASCII control characters from user input."""
    return _CONTROL_CHARS_RE.sub("", query or "").strip()


def detect_prompt_injection(query: str) -> str | None:
    """Return rule category if query matches deterministic prompt-injection/exfiltration patterns."""
    cleaned = sanitize_query_input(query)
    if not cleaned:
        return None
    for rule_name, pattern in _INJECTION_RULES:
        if pattern.search(cleaned):
            return rule_name
    return None


def format_injection_refusal(query: str) -> str:
    """Return a polite localized refusal message for blocked prompt injection attempts."""
    if detect_query_language(query) == "vi":
        return (
            "Yêu cầu này vi phạm chính sách bảo mật vận hành của hệ thống. "
            "Tôi không thể bỏ qua quy tắc kiểm chứng, tiết lộ system prompt hay truy xuất cấu hình/API key, "
            "và chỉ hỗ trợ phân tích số liệu tài chính đã xác thực."
        )
    return (
        "This request violates system security guardrails. "
        "I cannot override grounding rules, disclose system prompts, or expose configuration/API keys."
    )


@dataclass
class _Bucket:
    tokens: float
    last_refill: float
    last_seen: float


# shortcut: in-memory per-process bucket; move to Redis/KeyDB before multi-worker scaling
class IPTokenBucketRateLimiter:
    """Thread-safe, non-blocking per-IP Token Bucket rate limiter."""

    def __init__(
        self,
        rate_per_second: float,
        capacity: float,
        stale_seconds: float = 600.0,
        max_buckets: int = 10_000,
    ) -> None:
        if rate_per_second <= 0 or capacity <= 0:
            raise ValueError("rate_per_second and capacity must be strictly positive")
        self.rate = float(rate_per_second)
        self.capacity = float(capacity)
        self.stale_seconds = float(stale_seconds)
        self.max_buckets = int(max_buckets)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, client_ip: str) -> tuple[bool, float]:
        """Consume one token for client_ip if available.

        Returns:
            Tuple of (allowed, retry_after_seconds).
        """
        now = time.monotonic()
        with self._lock:
            if len(self._buckets) >= self.max_buckets and client_ip not in self._buckets:
                self._evict_stale(now)

            bucket = self._buckets.get(client_ip)
            if bucket is None:
                self._buckets[client_ip] = _Bucket(
                    tokens=self.capacity - 1.0,
                    last_refill=now,
                    last_seen=now,
                )
                return True, 0.0

            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
            bucket.last_refill = now
            bucket.last_seen = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            retry_after = (1.0 - bucket.tokens) / self.rate
            return False, max(0.1, round(retry_after, 2))

    def reset(self) -> None:
        """Clear all tracked IP buckets (useful in unit tests)."""
        with self._lock:
            self._buckets.clear()

    def _evict_stale(self, now: float) -> None:
        """Remove inactive IP buckets to bound memory usage."""
        cutoff = now - self.stale_seconds
        stale_keys = [ip for ip, b in self._buckets.items() if b.last_seen < cutoff]
        for key in stale_keys:
            del self._buckets[key]
        if len(self._buckets) >= self.max_buckets:
            # Drop the oldest-seen entry if all active buckets hit capacity
            oldest_ip = min(self._buckets, key=lambda ip: self._buckets[ip].last_seen)
            del self._buckets[oldest_ip]


def is_production_env() -> bool:
    """Return True when APP_ENV indicates a production deployment."""
    return os.getenv("APP_ENV", "local").lower() in ("prod", "production")


def _trust_proxy_headers() -> bool:
    """Return True only when explicitly configured behind a trusted reverse proxy."""
    return os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in ("1", "true", "yes")


def _parse_valid_ip(raw: str | None) -> str | None:
    """Validate and canonicalize an IPv4/IPv6 string; return None on malformed input."""
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def extract_client_ip(request: Request) -> str:
    """Resolve client IP safely, trusting forwarded headers only when TRUST_PROXY_HEADERS=true."""
    if _trust_proxy_headers():
        cf_ip = _parse_valid_ip(request.headers.get("cf-connecting-ip"))
        if cf_ip:
            return cf_ip
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Prefer rightmost valid IP appended by the trusted edge proxy to resist header spoofing
            for part in reversed(xff.split(",")):
                parsed = _parse_valid_ip(part)
                if parsed:
                    return parsed

    host = request.client.host if request.client else None
    if not host:
        return "unknown"
    return _parse_valid_ip(host) or host.strip()


def get_allowed_origins() -> list[str]:
    """Load strict CORS origin whitelist from ALLOWED_ORIGINS environment variable."""
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]

    if is_production_env():
        return []
    return ["http://localhost:8000", "http://127.0.0.1:8000"]


def _build_default_limiters() -> tuple[IPTokenBucketRateLimiter, IPTokenBucketRateLimiter]:
    query_per_min = float(os.getenv("QUERY_RATE_LIMIT_PER_MIN", "15"))
    query_burst = float(os.getenv("QUERY_RATE_BURST", "5"))
    api_per_min = float(os.getenv("API_RATE_LIMIT_PER_MIN", "60"))
    api_burst = float(os.getenv("API_RATE_BURST", "20"))
    return (
        IPTokenBucketRateLimiter(rate_per_second=query_per_min / 60.0, capacity=query_burst),
        IPTokenBucketRateLimiter(rate_per_second=api_per_min / 60.0, capacity=api_burst),
    )


_query_limiter, _general_limiter = _build_default_limiters()


def reset_rate_limiters() -> None:
    """Reset default rate limiters for test isolation."""
    _query_limiter.reset()
    _general_limiter.reset()


# ---------------------------------------------------------------------------
# Signed HttpOnly Cookie User Isolation
# ---------------------------------------------------------------------------

USER_COOKIE_NAME = "cfi_uid"
USER_COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
_EPHEMERAL_COOKIE_SECRET = secrets.token_hex(32)
_UUID_V4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _get_cookie_secret() -> bytes:
    """Resolve HMAC signing secret from env; fail fast in production if unset."""
    secret = os.getenv("SESSION_COOKIE_SECRET", "").strip()
    if secret:
        return secret.encode("utf-8")
    if is_production_env():
        raise ValueError("SESSION_COOKIE_SECRET is required when APP_ENV is production.")
    return _EPHEMERAL_COOKIE_SECRET.encode("utf-8")


def validate_security_config() -> None:
    """Validate mandatory production security settings at application startup."""
    _get_cookie_secret()


def validate_user_id(user_id: str) -> str:
    """Validate user_id format to prevent S3 key path traversal."""
    if not user_id or not _SAFE_USER_ID_RE.match(user_id):
        raise ValueError("Invalid user_id: must be 1-64 alphanumeric, hyphen, or underscore characters.")
    return user_id


def sign_user_id(user_id: str) -> str:
    """Sign a user_id using HMAC-SHA256 for tamper-proof cookie storage."""
    uid = validate_user_id(user_id)
    sig = hmac.new(_get_cookie_secret(), uid.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{uid}.{sig}"


def verify_signed_user_id(cookie_value: str | None) -> str | None:
    """Verify HMAC signature and UUIDv4 format of cfi_uid cookie; return user_id or None."""
    if not cookie_value or "." not in cookie_value:
        return None
    uid, sig = cookie_value.rsplit(".", 1)
    if not _UUID_V4_RE.match(uid) or len(sig) != 64:
        return None
    expected_sig = hmac.new(_get_cookie_secret(), uid.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    return uid


def get_current_user_id(request: Request, response: Response) -> str:
    """FastAPI dependency resolving or issuing an isolated HMAC-signed HttpOnly user_id cookie."""
    existing = verify_signed_user_id(request.cookies.get(USER_COOKIE_NAME))
    if existing is not None:
        return existing

    new_uid = str(uuid.uuid4())
    signed_val = sign_user_id(new_uid)
    is_https = request.headers.get("x-forwarded-proto", "").lower() == "https"
    is_prod = os.getenv("APP_ENV", "local").lower() in ("prod", "production")
    response.set_cookie(
        key=USER_COOKIE_NAME,
        value=signed_val,
        max_age=USER_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_https or is_prod,
        path="/",
    )
    return new_uid


class SecurityHardeningMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP rate limits on /api/* and attach HTTP security headers."""

    _CSP = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    )

    async def dispatch(self, request: Request, call_next):
        """Rate-limit API traffic and attach defensive response headers."""
        path = request.url.path

        if path.startswith("/api/") and path != "/api/health":
            client_ip = extract_client_ip(request)
            limiter = _query_limiter if path in ("/api/query", "/api/query/stream") else _general_limiter
            allowed, retry_after = limiter.allow(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "TooManyRequests",
                        "detail": "Rate limit exceeded. Please retry later.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(int(max(1, round(retry_after))))},
                )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = self._CSP

        is_https = request.headers.get("x-forwarded-proto", "").lower() == "https"
        if is_https or os.getenv("APP_ENV", "local").lower() in ("prod", "production"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
