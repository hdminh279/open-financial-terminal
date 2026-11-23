"""Structured JSON telemetry logging and query sanitization engine.

Provides append-only structured run logging to logs/run_telemetry.jsonl with
multiprocess file locking, in-memory thread synchronization, sensitive query redaction,
and fail-safe error isolation.
"""

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import statistics
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Sensitive pattern regexes for query redaction
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*|ghp_[A-Za-z0-9]{20,})\b", re.IGNORECASE
)
_AUTH_SECRET_PATTERN = re.compile(
    r"\b(password|passwd|secret|token|api_key|apikey)\s*[:=]\s*([^\s\[\]]+)", re.IGNORECASE
)

# Verified DeepSeek pricing (USD per 1M tokens) from docs/research/DEEPSEEK_API_COMPATIBILITY.md
_DEEPSEEK_PRICING_PER_1M: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("pro", (0.022, 0.66, 1.98)),
    ("flash", (0.003, 0.15, 0.60)),
)


def estimate_deepseek_cost_usd(
    model: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int | None = None,
) -> float:
    """Estimate USD inference cost from verified DeepSeek token usage metrics."""
    model_lower = (model or "deepseek-flash").lower()
    hit_rate, miss_rate, out_rate = _DEEPSEEK_PRICING_PER_1M[1][1]
    for marker, rates in _DEEPSEEK_PRICING_PER_1M:
        if marker in model_lower:
            hit_rate, miss_rate, out_rate = rates
            break

    p_tokens = max(0, int(prompt_tokens))
    c_tokens = max(0, int(completion_tokens))
    hits = max(0, min(p_tokens, int(cache_hit_tokens)))
    misses = max(0, p_tokens - hits) if cache_miss_tokens is None else max(0, int(cache_miss_tokens))

    cost = ((hits * hit_rate) + (misses * miss_rate) + (c_tokens * out_rate)) / 1_000_000.0
    return round(cost, 8)


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
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp of execution completion",
    )
    request_id: str = Field(..., description="Unique correlation identifier for the request turn")
    session_id: str | None = Field(default=None, description="Conversation session identifier")
    query: str = Field(..., description="Sanitized natural language query text")
    intent: str = Field(..., description="Classified agent intent")
    tickers: list[str] = Field(default_factory=list, description="Target company tickers")
    metrics: list[str] = Field(default_factory=list, description="Target financial metrics")
    periods: list[str] = Field(default_factory=list, description="Target fiscal periods/years")
    tools_called: list[str] = Field(default_factory=list, description="Names of executed tools")
    tool_status: dict[str, str] = Field(
        default_factory=dict,
        description="Execution status per tool (e.g. {'financial_sql_tool': 'success'})",
    )
    latency_ms: float = Field(..., description="Total end-to-end execution latency in milliseconds")
    prompt_tokens: int = Field(default=0, description="Total LLM prompt tokens consumed")
    completion_tokens: int = Field(default=0, description="Total LLM completion tokens generated")
    cache_hit_tokens: int = Field(default=0, description="Prompt tokens served from DeepSeek context cache")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated USD inference cost for this request")
    error_code: str | None = Field(
        default=None,
        description="Standardized error code if query encountered failures",
    )


class TelemetryLogger:
    """Thread-safe and process-safe telemetry logger writing to JSON Lines storage."""

    _lock = threading.Lock()

    def __init__(self, log_path: str | None = None) -> None:
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
        session_id: str | None = None,
        tickers: list[str] | None = None,
        metrics: list[str] | None = None,
        periods: list[str] | None = None,
        tools_called: list[str] | None = None,
        tool_status: dict[str, str] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_hit_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        error_code: str | None = None,
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
                prompt_tokens=max(0, int(prompt_tokens)),
                completion_tokens=max(0, int(completion_tokens)),
                cache_hit_tokens=max(0, int(cache_hit_tokens)),
                estimated_cost_usd=round(max(0.0, float(estimated_cost_usd)), 8),
                error_code=error_code,
            )

            # Ensure parent logs directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            line = record.model_dump_json() + "\n"

            with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(line)
                    f.flush()
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return True

        except Exception as exc:
            logger.warning("Failed to record run telemetry to %s: %s", self.log_path, exc)
            return False


# Shared default singleton
_DEFAULT_LOGGER: TelemetryLogger | None = None


def get_telemetry_logger() -> TelemetryLogger:
    """Retrieve or lazily initialize shared TelemetryLogger instance."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = TelemetryLogger()
    return _DEFAULT_LOGGER


def log_run_telemetry(**kwargs: Any) -> bool:
    """Convenience helper to record run telemetry via default logger."""
    return get_telemetry_logger().log_run(**kwargs)


def summarize_traffic_telemetry(
    run_log_path: str | Path | None = None,
    traffic_log_path: str | Path | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """Compute anonymized traffic metrics (DAR, unique sessions, top tickers, P50/P95) from run telemetry.

    Never exposes raw queries or PII; session IDs are hashed before counting.
    """
    src_path = Path(run_log_path or os.getenv("RUN_TELEMETRY_LOG_PATH") or "logs/run_telemetry.jsonl")
    dst_path = Path(traffic_log_path or os.getenv("TRAFFIC_TELEMETRY_LOG_PATH") or "logs/traffic_telemetry.jsonl")
    today_prefix = datetime.now(UTC).strftime("%Y-%m-%d")

    total_requests = 0
    requests_today = 0
    error_count = 0
    abstention_count = 0
    sessions: set[str] = set()
    intent_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    latencies: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cache_hit_tokens = 0
    total_estimated_cost_usd = 0.0

    if src_path.exists():
        with open(src_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_requests += 1
                ts = str(rec.get("timestamp") or "")
                if ts.startswith(today_prefix):
                    requests_today += 1

                sid = rec.get("session_id")
                if isinstance(sid, str) and sid:
                    sessions.add(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:12])

                intent = str(rec.get("intent") or "UNKNOWN")
                intent_counts[intent] += 1

                for t in rec.get("tickers") or []:
                    if isinstance(t, str) and t:
                        ticker_counts[t.upper()] += 1

                for tool in rec.get("tools_called") or []:
                    if isinstance(tool, str) and tool:
                        tool_counts[tool] += 1

                err = rec.get("error_code")
                if err:
                    if err == "INTERNAL_SERVER_ERROR":
                        error_count += 1
                    else:
                        abstention_count += 1

                lat = rec.get("latency_ms")
                if isinstance(lat, int | float) and lat >= 0:
                    latencies.append(float(lat))

                pt = rec.get("prompt_tokens")
                if isinstance(pt, int | float) and pt > 0:
                    total_prompt_tokens += int(pt)
                ct = rec.get("completion_tokens")
                if isinstance(ct, int | float) and ct > 0:
                    total_completion_tokens += int(ct)
                cht = rec.get("cache_hit_tokens")
                if isinstance(cht, int | float) and cht > 0:
                    total_cache_hit_tokens += int(cht)
                cost = rec.get("estimated_cost_usd")
                if isinstance(cost, int | float) and cost > 0:
                    total_estimated_cost_usd += float(cost)

    latencies.sort()
    p50 = round(statistics.median(latencies), 2) if latencies else 0.0
    p95 = round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2) if latencies else 0.0
    cache_hit_rate_pct = (
        round((total_cache_hit_tokens / total_prompt_tokens) * 100, 2) if total_prompt_tokens > 0 else 0.0
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_requests": total_requests,
        "requests_today_utc": requests_today,
        "unique_sessions": len(sessions),
        "error_rate_pct": round((error_count / total_requests) * 100, 2) if total_requests else 0.0,
        "abstention_rate_pct": round((abstention_count / total_requests) * 100, 2) if total_requests else 0.0,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cache_hit_tokens": total_cache_hit_tokens,
        "cache_hit_rate_pct": cache_hit_rate_pct,
        "total_estimated_cost_usd": round(total_estimated_cost_usd, 6),
        "avg_cost_per_request_usd": round(total_estimated_cost_usd / total_requests, 6) if total_requests else 0.0,
        "intent_breakdown": dict(intent_counts),
        "top_tickers": [{"ticker": k, "count": v} for k, v in ticker_counts.most_common(10)],
        "top_tools": [{"tool": k, "count": v} for k, v in tool_counts.most_common(10)],
    }

    if persist:
        with contextlib.suppress(OSError):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "a", encoding="utf-8") as out:
                out.write(json.dumps(summary) + "\n")

    return summary
