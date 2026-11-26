"""Pure-Python core security primitives: input sanitization, prompt-injection detection, and user ID validation.

Kept free of FastAPI/Starlette dependencies so AI and service layers can enforce guardrails
without coupling to the HTTP transport layer.
"""

import re

from backend.core.financial.response_templates import detect_query_language

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

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


def validate_user_id(user_id: str) -> str:
    """Validate user_id format to prevent S3 key path traversal."""
    if not user_id or not _SAFE_USER_ID_RE.match(user_id):
        raise ValueError("Invalid user_id: must be 1-64 alphanumeric, hyphen, or underscore characters.")
    return user_id
