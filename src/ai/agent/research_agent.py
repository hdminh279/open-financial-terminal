"""High-level Facade for Corporate Filing Intelligence Research Agent.

Encapsulates the compiled LangGraph StateGraph, provides a clean query() interface,
converts raw state outputs into strongly-typed AgentResponse models, and records
end-to-end telemetry.
"""

import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from src.ai.agent.cache import TTLQueryCache, normalize_cache_key
from src.ai.agent.citation import build_synthesis_messages
from src.ai.agent.contracts import (
    AbstentionInfo,
    AbstentionReason,
    ActivityTimelineStep,
    AgentRequest,
    AgentResponse,
    AgentTokenUsage,
    ChartData,
    Citation,
    FinancialQueryPlan,
    IntentType,
    ReportSection,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from src.ai.agent.graph import build_agent_graph
from src.ai.agent.nodes.citation_node import citation_node
from src.ai.agent.state import AgentState
from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.factory import LLMFactory
from src.api.security import detect_prompt_injection, format_injection_refusal, sanitize_query_input
from src.core.observability.telemetry import estimate_deepseek_cost_usd

logger = logging.getLogger(__name__)

_normalize_cache_key = normalize_cache_key


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

    def _build_response_from_state(
        self,
        query_text: str,
        final_state: dict[str, Any],
        total_latency_ms: float,
    ) -> AgentResponse:
        """Convert final AgentState dictionary into a validated AgentResponse envelope."""
        raw_intent = final_state.get("intent", IntentType.OUT_OF_SCOPE.value)
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            intent = IntentType.OUT_OF_SCOPE

        citations = [Citation(**c) if isinstance(c, dict) else c for c in final_state.get("citations", [])]
        charts = [ChartData(**ch) if isinstance(ch, dict) else ch for ch in final_state.get("charts", [])]
        report_sections = [
            ReportSection(**sec) if isinstance(sec, dict) else sec for sec in final_state.get("report_sections", [])
        ]
        raw_abs = final_state.get("abstention_info")
        abstention = AbstentionInfo(**raw_abs) if raw_abs else None
        tool_results = [
            ToolExecutionResult(**tr) if isinstance(tr, dict) else tr for tr in final_state.get("tool_results", [])
        ]
        raw_plan = final_state.get("query_plan")
        query_plan = FinancialQueryPlan(**raw_plan) if raw_plan else None
        answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""

        raw_timeline = final_state.get("activity_timeline", [])
        ordered_timeline = canonicalize_timeline(
            raw_events=raw_timeline,
            intent=intent,
            has_charts=len(charts) > 0,
        )

        token_usage: AgentTokenUsage | None = None
        usage_entries = final_state.get("llm_usage") or []
        if usage_entries:
            agg_prompt = 0
            agg_completion = 0
            agg_cache_hit = 0
            agg_total = 0
            agg_cost = 0.0
            for entry in usage_entries:
                if not isinstance(entry, dict):
                    continue
                pt = int(entry.get("prompt_tokens") or 0)
                ct = int(entry.get("completion_tokens") or 0)
                cht = int(entry.get("cache_hit_tokens") or 0)
                cmt = entry.get("cache_miss_tokens")
                tt = int(entry.get("total_tokens") or (pt + ct))
                agg_prompt += pt
                agg_completion += ct
                agg_cache_hit += cht
                agg_total += tt
                agg_cost += estimate_deepseek_cost_usd(
                    model=entry.get("model"),
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cache_hit_tokens=cht,
                    cache_miss_tokens=int(cmt) if cmt is not None else None,
                )
            token_usage = AgentTokenUsage(
                prompt_tokens=agg_prompt,
                completion_tokens=agg_completion,
                cache_hit_tokens=agg_cache_hit,
                total_tokens=agg_total,
                estimated_cost_usd=round(agg_cost, 8),
            )

        return AgentResponse(
            query=query_text,
            intent=intent,
            answer=answer,
            citations=citations,
            charts=charts,
            report_sections=report_sections,
            abstention=abstention,
            tool_results=tool_results,
            execution_time_ms=total_latency_ms,
            token_usage=token_usage,
            query_plan=query_plan,
            activity_timeline=ordered_timeline,
        )

    @staticmethod
    def _build_injection_response(query_text: str, injection_rule: str, guard_latency_ms: float) -> AgentResponse:
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

    def query(
        self,
        request: AgentRequest | str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Execute end-to-end query workflow across LangGraph state graph."""
        start_time = time.perf_counter()

        raw_query = request.query if isinstance(request, AgentRequest) else str(request)
        query_text = sanitize_query_input(raw_query)

        # Pre-LLM deterministic prompt-injection & secret-exfiltration guard ($0 LLM cost)
        injection_rule = detect_prompt_injection(query_text)
        if injection_rule is not None:
            guard_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_injection_response(query_text, injection_rule, guard_latency_ms)

        cache_key = _normalize_cache_key(query_text) if not conversation_history else ""

        if cache_key:
            cached_resp = self._query_cache.get(cache_key)
            if cached_resp is not None:
                cache_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return cached_resp.model_copy(update={"execution_time_ms": cache_latency_ms})

        initial_state: AgentState = {
            "query": query_text,
            "conversation_history": conversation_history or [],
            "tool_results": [],
            "evidence_items": [],
            "citations": [],
            "charts": [],
            "report_sections": [],
            "llm_usage": [],
            "stream_mode": False,
        }

        final_state: dict[str, Any] = self._graph.invoke(initial_state)
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        response = self._build_response_from_state(query_text, final_state, total_latency_ms)

        if cache_key and self._is_cacheable(response, conversation_history):
            self._query_cache.set(cache_key, response)

        return response

    def stream_query(
        self,
        request: AgentRequest | str,
        conversation_history: list[dict[str, Any]] | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream LangGraph node progress events, synthesis token deltas, and final AgentResponse."""
        start_time = time.perf_counter()
        raw_query = request.query if isinstance(request, AgentRequest) else str(request)
        query_text = sanitize_query_input(raw_query)

        injection_rule = detect_prompt_injection(query_text)
        if injection_rule is not None:
            guard_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            resp = self._build_injection_response(query_text, injection_rule, guard_latency_ms)
            yield {"event": "status", "data": {"step": "Validating evidence", "status": "completed"}}
            yield {"event": "token", "data": {"delta": resp.answer}}
            yield {"event": "done", "data": resp}
            return

        cache_key = _normalize_cache_key(query_text) if not conversation_history else ""
        if cache_key:
            cached_resp = self._query_cache.get(cache_key)
            if cached_resp is not None:
                cache_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                resp = cached_resp.model_copy(update={"execution_time_ms": cache_latency_ms})
                yield {"event": "status", "data": {"step": "Understanding question", "status": "completed"}}
                yield {"event": "token", "data": {"delta": resp.answer}}
                yield {"event": "done", "data": resp}
                return

        merged_state: dict[str, Any] = {
            "query": query_text,
            "conversation_history": conversation_history or [],
            "tool_results": [],
            "evidence_items": [],
            "citations": [],
            "charts": [],
            "report_sections": [],
            "llm_usage": [],
            "stream_mode": True,
        }
        reducer_keys = {"tool_results", "evidence_items", "activity_timeline", "llm_usage"}

        yield {"event": "status", "data": {"step": "Understanding question", "status": "in_progress"}}

        for update in self._graph.stream(merged_state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for _node_name, node_payload in update.items():
                if not isinstance(node_payload, dict):
                    continue
                for k, v in node_payload.items():
                    if k in reducer_keys:
                        merged_state.setdefault(k, [])
                        if isinstance(v, list):
                            merged_state[k].extend(v)
                    else:
                        merged_state[k] = v

                for step_ev in node_payload.get("activity_timeline") or []:
                    if isinstance(step_ev, dict):
                        yield {"event": "status", "data": step_ev}

        is_abstaining = bool(merged_state.get("is_abstaining", False))
        is_partial = bool(merged_state.get("is_partial", False))

        if is_abstaining and not is_partial:
            final_text = merged_state.get("final_answer") or merged_state.get("draft_answer") or ""
            if final_text:
                yield {"event": "token", "data": {"delta": final_text}}
        else:
            yield {"event": "status", "data": {"step": "Synthesizing answer", "status": "in_progress"}}
            adapter = llm_adapter or LLMFactory.get_adapter(profile="chat")
            messages = build_synthesis_messages(merged_state)
            draft_parts: list[str] = []
            prefix_buf = ""
            prefix_flushed = False
            tag_prefix = "[INSUFFICIENT_EVIDENCE]"
            stream_model = getattr(adapter, "model", "deepseek-flash")

            try:
                for chunk in adapter.stream(messages=messages, temperature=0.0):
                    if chunk.content_delta:
                        draft_parts.append(chunk.content_delta)
                        if not prefix_flushed:
                            prefix_buf += chunk.content_delta
                            if len(prefix_buf) >= len(tag_prefix) or (
                                not tag_prefix.startswith(prefix_buf.lstrip()) and "\n" in prefix_buf
                            ):
                                cleaned_prefix = prefix_buf.lstrip()
                                if cleaned_prefix.startswith(tag_prefix):
                                    cleaned_prefix = cleaned_prefix[len(tag_prefix) :].lstrip()
                                if cleaned_prefix:
                                    yield {"event": "token", "data": {"delta": cleaned_prefix}}
                                prefix_flushed = True
                        else:
                            yield {"event": "token", "data": {"delta": chunk.content_delta}}

                    if chunk.usage is not None:
                        merged_state.setdefault("llm_usage", []).append(
                            chunk.usage.model_dump() | {"model": stream_model}
                        )

                if not prefix_flushed and prefix_buf:
                    cleaned_prefix = prefix_buf.lstrip()
                    if cleaned_prefix.startswith(tag_prefix):
                        cleaned_prefix = cleaned_prefix[len(tag_prefix) :].lstrip()
                    if cleaned_prefix:
                        yield {"event": "token", "data": {"delta": cleaned_prefix}}

                merged_state["draft_answer"] = "".join(draft_parts)
            except Exception as exc:
                logger.error("Streaming synthesis failed: %s", exc, exc_info=True)
                err_msg = f"Error synthesizing response: {exc}."
                merged_state["draft_answer"] = err_msg
                yield {"event": "token", "data": {"delta": err_msg}}

            cit_update = citation_node({**merged_state, "stream_mode": False})
            for k, v in cit_update.items():
                if k in reducer_keys:
                    merged_state.setdefault(k, [])
                    if isinstance(v, list):
                        merged_state[k].extend(v)
                else:
                    merged_state[k] = v

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        response = self._build_response_from_state(query_text, merged_state, total_latency_ms)

        if cache_key and self._is_cacheable(response, conversation_history):
            self._query_cache.set(cache_key, response)

        yield {"event": "done", "data": response}
