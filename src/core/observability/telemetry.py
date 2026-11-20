"""Structured JSON telemetry logging and query sanitization engine.

Provides append-only structured run logging to logs/run_telemetry.jsonl with
multiprocess file locking, in-memory thread synchronization, sensitive query redaction,
and fail-safe error isolation.
"""

from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Sensitive pattern regexes for query redaction
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_API_KEY_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*|ghp_[A-Za-z0-9]{20,})\b", re.IGNORECASE)
_AUTH_SECRET_PATTERN = re.compile(r"\b(password|passwd|secret|token|api_key|apikey)\s*[:=]\s*([^\s\[\]]+)", re.IGNORECASE)


def sanitize_query(query: str) -> str:
    """Sanitize and redact sensitive credentials or PII from raw query string.

    Args:
        query: Raw incoming query string.

    Returns:
        Redacted query safe for telemetry logging.
    """
    if not query:
        return ""

    sanitized = query.strip()
    sanitized = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
    sanitized = _API_KEY_PATTERN.sub("[REDACTED_KEY]", sanitized)
    sanitized = _AUTH_SECRET_PATTERN.sub(r"\1=[REDACTED_SECRET]", sanitized)

    if len(sanitized) > 1000:
        sanitized = sanitized[:1000] + "... [TRUNCATED]"

    return sanitized


class RunTelemetryRecord(BaseModel):
    """Schema contract for single-line JSONL telemetry records."""

    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of execution completion",
    )
    request_id: str = Field(..., description="Unique correlation identifier for the request turn")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier")
    query: str = Field(..., description="Sanitized natural language query text")
    intent: str = Field(..., description="Classified agent intent")
    tickers: List[str] = Field(default_factory=list, description="Target company tickers")
    metrics: List[str] = Field(default_factory=list, description="Target financial metrics")
    periods: List[str] = Field(default_factory=list, description="Target fiscal periods/years")
    tools_called: List[str] = Field(default_factory=list, description="Names of executed tools")
    tool_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution status per tool (e.g. {'financial_sql_tool': 'success'})",
    )
    latency_ms: float = Field(..., description="Total end-to-end execution latency in milliseconds")
    error_code: Optional[str] = Field(
        default=None,
        description="Standardized error code if query encountered failures",
    )


class TelemetryLogger:
    """Thread-safe and process-safe telemetry logger writing to JSON Lines storage."""

    _lock = threading.Lock()

    def __init__(self, log_path: Optional[str] = None) -> None:
        """Initialize logger with configured destination path.

        Args:
            log_path: Path to telemetry file. Defaults to logs/run_telemetry.jsonl.
        """
        env_path = os.getenv("RUN_TELEMETRY_LOG_PATH")
        self.log_path = Path(log_path or env_path or "logs/run_telemetry.jsonl")

    def log_run(
        self,
        request_id: str,
        query: str,
        intent: str,
        latency_ms: float,
        session_id: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        tool_status: Optional[Dict[str, str]] = None,
        error_code: Optional[str] = None,
    ) -> bool:
        """Serialize telemetry record and write append-only with concurrency protection.

        Never raises exceptions to callers; logs internal warnings on write failure.

        Returns:
            True if log written successfully, False otherwise.
        """
        try:
            sanitized_q = sanitize_query(query)
            record = RunTelemetryRecord(
                request_id=request_id,
                session_id=session_id,
                query=sanitized_q,
                intent=intent,
                tickers=tickers or [],
                metrics=metrics or [],
                periods=periods or [],
                tools_called=tools_called or [],
                tool_status=tool_status or {},
                latency_ms=round(latency_ms, 2),
                error_code=error_code,
            )

            # Ensure parent logs directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            line = record.model_dump_json() + "\n"

            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(line)
                        f.flush()
                    finally:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass

            return True

        except Exception as exc:
            logger.warning("Failed to record run telemetry to %s: %s", self.log_path, exc)
            return False


# Shared default singleton
_DEFAULT_LOGGER: Optional[TelemetryLogger] = None


def get_telemetry_logger() -> TelemetryLogger:
    """Retrieve or lazily initialize shared TelemetryLogger instance."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = TelemetryLogger()
    return _DEFAULT_LOGGER


def log_run_telemetry(**kwargs: Any) -> bool:
    """Convenience helper to record run telemetry via default logger."""
    return get_telemetry_logger().log_run(**kwargs)
