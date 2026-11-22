"""High-level Facade for Corporate Filing Intelligence Research Agent.

Encapsulates the compiled LangGraph StateGraph, provides a clean query() interface,
converts raw state outputs into strongly-typed AgentResponse models, and records
end-to-end telemetry.
"""

import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from src.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    ActivityTimelineStep,
    AgentRequest,
    AgentResponse,
    ChartData,
    Citation,
    FinancialQueryPlan,
    IntentType,
    ReportSection,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.graph import build_agent_graph
from src.ai.agent.state import AgentState
from src.api.security import detect_prompt_injection, format_injection_refusal, sanitize_query_input


def _normalize_cache_key(query: str) -> str:
    """Normalize a standalone query string into a canonical cache key."""
    return re.sub(r"\s+", " ", query.strip().lower())


class TTLQueryCache:
    """Thread-safe, bounded LRU in-memory TTL cache for standalone AgentResponse objects."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 256) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, tuple[float, AgentResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> AgentResponse | None:
        """Retrieve a non-expired cached response or None on miss."""
        if self.ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, response = entry
            if now >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return response

    def set(self, key: str, response: AgentResponse) -> None:
        """Store a response in the bounded LRU cache with TTL."""
        if self.ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._store[key] = (expires_at, response)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Return current number of entries stored in cache."""
        with self._lock:
            return len(self._store)


CANONICAL_TIMELINE_STEPS = [
    "Understanding question",
    "Resolving metrics",
    "Querying DuckDB",
    "Searching SEC filings",
    "Validating evidence",
    "Preparing visualization",
]


def canonicalize_timeline(
    raw_events: list[dict[str, Any]],
    intent: IntentType,
    has_charts: bool,
) -> list[ActivityTimelineStep]:
    """Sort and reconcile concurrent timeline events into a deterministic sequence."""
    event_by_step: dict[str, dict[str, Any]] = {e["step"]: e for e in raw_events if "step" in e}

    ordered_steps: list[ActivityTimelineStep] = []
    steps_to_process = list(CANONICAL_TIMELINE_STEPS)
    if "Searching live web" in event_by_step:
        sec_idx = steps_to_process.index("Searching SEC filings")
        steps_to_process.insert(sec_idx + 1, "Searching live web")

    for step_name in steps_to_process:
        if step_name in event_by_step:
            ev = event_by_step[step_name]
            ordered_steps.append(
                ActivityTimelineStep(
                    step=ev["step"],
                    status=ev.get("status", "completed"),
                    description=ev.get("description"),
                    duration_ms=ev.get("duration_ms"),
                )
            )
        else:
            status = "skipped"
            description = "Not required for this inquiry"

            if step_name == "Querying DuckDB" and intent == IntentType.TEXTUAL_RESEARCH:
                description = "Skipped (pure textual research)"
            elif step_name == "Searching SEC filings" and intent == IntentType.FINANCIAL_METRIC:
                description = "Skipped (structured metric query)"
            elif step_name == "Searching live web":
                description = "Skipped (historical filing inquiry)"
            elif step_name == "Preparing visualization" and not has_charts:
                description = "Skipped (no visual trend requested)"

            ordered_steps.append(
                ActivityTimelineStep(
                    step=step_name,
                    status=status,
                    description=description,
                    duration_ms=0.0,
                )
            )

    return ordered_steps


class ResearchAgent:
    """Enterprise Research Agent orchestrating multi-tool SEC filing intelligence."""

    def __init__(
        self,
        compiled_graph: Any | None = None,
        cache_ttl_seconds: float | None = None,
        cache_max_size: int = 256,
    ) -> None:
        """Initialize agent with compiled LangGraph application and optional TTL query cache."""
        self._graph = compiled_graph or build_agent_graph()
        ttl = (
            float(os.getenv("QUERY_CACHE_TTL_SECONDS", "300"))
            if cache_ttl_seconds is None
            else float(cache_ttl_seconds)
        )
        self._query_cache = TTLQueryCache(ttl_seconds=ttl, max_size=cache_max_size)

    @property
    def graph(self) -> Any:
        """Access the underlying compiled LangGraph app."""
        return self._graph

    @property
    def query_cache(self) -> TTLQueryCache:
        """Access the in-memory TTL query cache."""
        return self._query_cache

    @staticmethod
    def _is_cacheable(
        response: AgentResponse,
        conversation_history: list[dict[str, Any]] | None,
    ) -> bool:
        """Allow caching only for successful standalone non-live-news responses."""
        if conversation_history or not response.answer:
            return False
        if response.abstention and response.abstention.reason == AbstentionReason.TOOL_ERROR:
            return False
        for tr in response.tool_results:
            if tr.tool_name == "web_search_tool":
                return False
            if tr.status == ToolExecutionStatus.TOOL_FAILED:
                return False
        return True

    def query(
        self,
        request: AgentRequest | str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Execute end-to-end query workflow across LangGraph state graph.

        Args:
            request: Natural language query string or validated AgentRequest.
            conversation_history: Optional list of previous chat turns for contextual resolution.

        Returns:
            Validated AgentResponse envelope with grounded answers, citations, and telemetry.
        """
        start_time = time.perf_counter()

        raw_query = request.query if isinstance(request, AgentRequest) else str(request)
        query_text = sanitize_query_input(raw_query)

        # Pre-LLM deterministic prompt-injection & secret-exfiltration guard ($0 LLM cost)
        injection_rule = detect_prompt_injection(query_text)
        if injection_rule is not None:
            guard_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            refusal_msg = format_injection_refusal(query_text)
            guard_timeline = canonicalize_timeline(
                raw_events=[
                    {
                        "step": "Understanding question",
                        "status": "completed",
                        "description": f"Blocked pre-LLM by security guard ({injection_rule})",
                        "duration_ms": guard_latency_ms,
                    },
                    {
                        "step": "Validating evidence",
                        "status": "completed",
                        "description": "Pre-LLM security abstention enforced",
                        "duration_ms": 0.0,
                    },
                ],
                intent=IntentType.OUT_OF_SCOPE,
                has_charts=False,
            )
            return AgentResponse(
                query=query_text,
                intent=IntentType.OUT_OF_SCOPE,
                answer=refusal_msg,
                citations=[],
                charts=[],
                report_sections=[],
                abstention=AbstentionInfo(
                    is_abstaining=True,
                    is_partial=False,
                    reason=AbstentionReason.UNSUPPORTED_TOPIC,
                    explanation=refusal_msg,
                    unsupported_aspects=[f"security:{injection_rule}"],
                ),
                tool_results=[],
                execution_time_ms=guard_latency_ms,
                query_plan=None,
                activity_timeline=guard_timeline,
            )

        cache_key = _normalize_cache_key(query_text) if not conversation_history else ""

        if cache_key:
            cached_resp = self._query_cache.get(cache_key)
            if cached_resp is not None:
                cache_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return cached_resp.model_copy(update={"execution_time_ms": cache_latency_ms})

        # Initialize input state
        initial_state: AgentState = {
            "query": query_text,
            "conversation_history": conversation_history or [],
            "tool_results": [],
            "evidence_items": [],
            "citations": [],
            "charts": [],
            "report_sections": [],
        }

        # Invoke the compiled LangGraph workflow
        final_state: dict[str, Any] = self._graph.invoke(initial_state)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Unpack intent
        raw_intent = final_state.get("intent", IntentType.OUT_OF_SCOPE.value)
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            intent = IntentType.OUT_OF_SCOPE

        # Unpack citations
        citations = [Citation(**c) if isinstance(c, dict) else c for c in final_state.get("citations", [])]

        # Unpack charts
        charts = [ChartData(**ch) if isinstance(ch, dict) else ch for ch in final_state.get("charts", [])]

        # Unpack report sections
        report_sections = [
            ReportSection(**sec) if isinstance(sec, dict) else sec for sec in final_state.get("report_sections", [])
        ]

        # Unpack abstention info
        raw_abs = final_state.get("abstention_info")
        abstention = AbstentionInfo(**raw_abs) if raw_abs else None

        # Unpack tool results
        tool_results = [
            ToolExecutionResult(**tr) if isinstance(tr, dict) else tr for tr in final_state.get("tool_results", [])
        ]

        # Unpack query plan
        raw_plan = final_state.get("query_plan")
        query_plan = FinancialQueryPlan(**raw_plan) if raw_plan else None

        answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""

        # Unpack and canonicalize activity timeline
        raw_timeline = final_state.get("activity_timeline", [])
        ordered_timeline = canonicalize_timeline(
            raw_events=raw_timeline,
            intent=intent,
            has_charts=len(charts) > 0,
        )

        response = AgentResponse(
            query=query_text,
            intent=intent,
            answer=answer,
            citations=citations,
            charts=charts,
            report_sections=report_sections,
            abstention=abstention,
            tool_results=tool_results,
            execution_time_ms=total_latency_ms,
            query_plan=query_plan,
            activity_timeline=ordered_timeline,
        )

        if cache_key and self._is_cacheable(response, conversation_history):
            self._query_cache.set(cache_key, response)

        return response
