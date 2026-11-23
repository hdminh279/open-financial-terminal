"""Intent Router and Autonomous Tool Planning Node for LangGraph.

Empowers the LLM to autonomously evaluate user inquiries, classify intent
(including DEEP_RESEARCH_REPORT), select required tools (SQL, RAG, or both),
and synthesize specialized query parameters without hardcoded regex heuristics.
"""

import json
import logging
import re
from typing import Any

from src.ai.agent.contracts import (
    ComparisonType,
    ExecutionStrategy,
    FinancialQueryPlan,
    IntentType,
    RAGQueryParams,
    RoutingDecision,
    SQLQueryParams,
    WebSearchQueryParams,
)
from src.ai.agent.state import AgentState
from src.ai.llm.base import BaseLLMAdapter
from src.ai.llm.contracts import ChatMessage
from src.ai.llm.factory import LLMFactory
from src.core.financial.company_registry import get_supported_tickers, load_company_catalog
from src.core.financial.metric_registry import MetricRegistry
from src.core.financial.response_templates import get_routing_markers
from src.core.financial.year_utils import parse_fiscal_years

logger = logging.getLogger(__name__)


def is_realtime_news_query(query: str) -> bool:
    """Return True when the query contains high-precision live-news phrases."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    markers = get_routing_markers("high_precision_realtime_news")
    return any(marker in q_lower for marker in markers)


def has_anaphora_reference(query: str) -> bool:
    """Return True when the query contains conversational follow-up references."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    markers = get_routing_markers("anaphora_references")
    return any(marker in q_lower for marker in markers)


def should_attach_history(query: str) -> bool:
    """Determine whether prior conversation context should be injected into the planner prompt.

    Standalone live-news inquiries without conversational anaphora are isolated from
    prior history so previously discussed tickers cannot leak into current news searches.
    """
    return not (is_realtime_news_query(query) and not has_anaphora_reference(query))


def _is_ticker_mentioned_in_query(ticker: str, query_lower: str, catalog: list[dict[str, Any]]) -> bool:
    """Check whether a specific ticker or its company name appears in the current query."""
    t_upper = ticker.strip().upper()
    if re.search(rf"\b{re.escape(t_upper.lower())}\b", query_lower):
        return True
    stop_words = {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "company",
        "group",
        "holdings",
        "plc",
        "bank",
    }
    for comp in catalog:
        if str(comp.get("ticker", "")).upper() != t_upper:
            continue
        name = str(comp.get("name", "")).lower()
        aliases = [str(a).lower() for a in (comp.get("aliases") or []) if isinstance(a, str)]
        candidates = [name] + aliases
        name_tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]+", name)
            if len(tok) >= 3 and tok not in stop_words
        ]
        candidates.extend(name_tokens)
        for term in candidates:
            if term and re.search(rf"\b{re.escape(term)}\b", query_lower):
                return True
    return False


def _query_mentions_sql_metric(query: str, metric_registry: MetricRegistry) -> bool:
    """Return True when the current query explicitly references a SQL Gold financial metric."""
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return False
    for syn, canonical in metric_registry._synonym_index.items():
        if len(syn) < 3:
            continue
        if not metric_registry.is_sql_metric(canonical):
            continue
        if re.search(rf"\b{re.escape(syn)}\b", q_lower):
            return True
    return False


PLANNER_SYSTEM_PROMPT = """You are the master financial query planner and tool selector for Corporate Filing Intelligence.
Your job is to analyze user inquiries across international corporations (US SEC EDGAR 10-K/10-Q/20-F) and Vietnamese enterprises (VN30 BCTC under VAS), determine intent, select the required tools, extract parameters, and plan the execution flow.

### System Universe & Capabilities:
- Corporate Universe: Global corporations (e.g. Apple (AAPL), Microsoft (MSFT), Alphabet/Google (GOOGL), NVIDIA (NVDA)) and Vietnamese Blue-chips (e.g. FPT, Vinamilk (VNM), Hoa Phat (HPG), Vietcombank (VCB), etc.).
- Multi-Layer Intelligence Architecture:
  1. Historical Financial Accounting (DuckDB SQL Marts)
  2. Textual Filing Disclosures & MD&A (Qdrant Vector DB)
  3. Real-Time Web Search & Breaking News (DuckDuckGo Live Search)

### Available Tools:
1. 'financial_sql_tool':
   - Queries exact numerical accounting figures and exact period dates (start_date, end_date) from balance sheets, income statements, cash flows, and financial ratios.
   - Supported metrics: revenues, net_income, operating_income, gross_profit, eps_diluted, net_interest_income,
     total_operating_income, operating_cash_flow, cash_and_cash_equivalents, current_assets, current_liabilities,
     total_assets, total_liabilities, stockholders_equity_parent, stockholders_equity_total,
     operating_margin, net_margin, gross_margin, current_ratio, debt_to_equity, revenue_growth_yoy.
   - NOTE: financial_sql_tool stores CONSOLIDATED corporate metrics. It does NOT store product-line segments (like iPhone, Mac, Azure).
   - NOTE (Banks / VAS-BANK): Commercial banks (e.g. VCB, BID, CTG, TCB, MBB, ACB, HDB, TPB, VIB, STB, SHB, SSB, LPB) do NOT report corporate 'revenues'; query 'total_operating_income' and 'net_interest_income' instead (or alongside 'revenues').
2. 'hybrid_retriever':
   - Searches narrative disclosures in filed reports: business overview (Item 1), risk factors (Item 1A), legal proceedings (Item 3),
     Management's Discussion & Analysis / MD&A (Item 7), segment/product-line revenue tables (Item 7 & Note disclosures in Item 8),
     and strategic initiatives (e.g. AI, cloud, product sales).
3. 'web_search_tool':
   - Searches real-time internet information, breaking news, recent events, stock market movements today, leadership changes,
     partnerships, macro developments, or events occurring after the latest periodic filing date.

### Intent & Tool Planning Rules:
1. 'FINANCIAL_METRIC': Pure numerical lookup of consolidated accounting metrics or financial ratios.
   - Tools: ["financial_sql_tool"]
   - Strategy: "parallel"
2. 'TEXTUAL_RESEARCH': Qualitative research into strategy, risk factors, legal matters, product categories, or management commentary.
   - Tools: ["hybrid_retriever"]
   - Strategy: "parallel"
   - CRITICAL: Whenever a user asks what a supported company disclosed in its filings (10-K, 10-Q, 20-F) — even about unusual, historical, or potentially absent topics — ALWAYS classify as 'TEXTUAL_RESEARCH' with ["hybrid_retriever"] so the retrieval and abstention pipeline verifies the filings directly rather than guessing 'OUT_OF_SCOPE' upfront.
3. 'MIXED_RESEARCH': Inquiries requiring BOTH concrete metrics and qualitative context/explanations, OR queries about specific product/service segments (e.g. iPhone, Azure, Cloud), OR inquiries combining filed data with recent news.
   - Tools: Relevant combination, e.g. ["financial_sql_tool", "hybrid_retriever"], ["financial_sql_tool", "web_search_tool"], or all three.
   - Strategy: "parallel"
4. 'DEEP_RESEARCH_REPORT': Comprehensive requests asking for a full report, multi-section equity analysis, deep dive, or comparative overview with charts.
   - Tools: ["financial_sql_tool", "hybrid_retriever"] (and "web_search_tool" if latest updates are relevant)
   - Strategy: "parallel"
5. Live News & Recent Events:
   - Pure Live News (general/macro economic/market news like 'Give me today's latest news', 'Tin tức kinh tế nổi bật hôm nay là gì?', or company news like 'Latest news about FPT today', 'Breaking news about NVDA today' in any language):
     * NEVER classify live news inquiries as 'OUT_OF_SCOPE'; always set "intent": "TEXTUAL_RESEARCH", "is_realtime_news": true.
     * Tools: ["web_search_tool"] ONLY (do NOT add "financial_sql_tool" or "hybrid_retriever" unless the current query also asks for financial metrics or filing disclosures).
     * Set "sql_params": null and "rag_params": null. Formulate "web_params" with search_query.
   - Hybrid Financial Metrics + Live News (e.g. 'FPT 2024 revenue and latest news about FPT today'):
     * Set "is_realtime_news": true, "intent": "MIXED_RESEARCH".
     * Tools: ["financial_sql_tool", "web_search_tool"] (parallel fan-out). Include "hybrid_retriever" when filing narratives or management commentary are also requested.
   - Pure Filing Research (10-K/10-Q/20-F): Do NOT add "web_search_tool" unless the user explicitly asks for recent post-filing updates.
   - Commodity & Disambiguated Web Queries: When searching commodity prices (e.g. 'giá khí gas thế giới'), formulate web_params.search_query specifically for the commodity (e.g. 'natural gas futures price today -crypto -token') to exclude unrelated crypto tokens.
6. 'OUT_OF_SCOPE': Strictly reserved for completely unrelated non-corporate trivia (weather, general programming syntax, personal chit-chat) OR speculative future stock price prediction / investment recommendations (e.g. 'Should I buy stock X?', 'Will price rise tomorrow?').
   - Tools: []
   - Strategy: "direct_abstain"
   - Do NOT classify a filing-disclosure question about a supported company as 'OUT_OF_SCOPE' just because the topic sounds unlikely; route to 'TEXTUAL_RESEARCH' so 'hybrid_retriever' verifies the corpus.

### Parameter Planning Guidelines:
- Product & Segment Inquiries: When the user asks about specific product lines or business segments (such as 'iPhone', 'iPad', 'Mac', 'Services', 'Wearables', 'Azure', 'Xbox', 'Surface', 'YouTube', 'Google Cloud', 'Google Search'):
  * Always select 'hybrid_retriever' (or 'MIXED_RESEARCH' if net income or total revenue is also requested).
  * Formulate "search_query" in rag_params targeting segment disclosures, e.g.: "{ticker} net sales by product category {product}". Set "item": "Item 7" in rag_params rather than repeating Item keywords in query text.
- Fiscal Calendar & Period End-Date Inquiries: When a user asks when a company's fiscal year ends, whether its fiscal calendar is shifted, or asks for revenue alongside fiscal period dates, ALWAYS include 'financial_sql_tool' (it stores exact start_date and end_date for every period).
- Contextual Isolation vs Follow-ups (CRITICAL):
  * Set "is_followup": false whenever the current query is self-contained (names its own target company/companies like 'Compare FPT and VNM', OR asks general breaking news like 'Give me today's latest news'). When "is_followup" is false, NEVER inherit tickers, metrics, or years from Prior Conversation Context.
  * Set "is_followup": true ONLY when the current query is an ellipsis or anaphoric follow-up relying on prior turns (e.g. 'what about Microsoft?', 'draw a chart for this', 'how was its revenue?'). Only then carry over the relevant ticker, metric, or period from previous turns.
- Multi-Year Comparisons & Trends: If the user inquiry mentions multiple periods, asks for a trend, specifies a range, or asks to compare against prior years (e.g. '2024 vs 3 prior years', 'so sánh với 3 năm trước đó'), populate the full expanded array in "fiscal_years" (e.g. [2021, 2022, 2023, 2024]) OR set "fiscal_years": [] to fetch all available periods. NEVER restrict "fiscal_years" to a single year like [2024] when comparing across multiple years.
- Metric Inference for Follow-up / Conversational queries: If a question specifies a company without an explicit metric name, infer the primary financial metric: 'net_income' (if discussing profit) or 'revenues' (default), and set "fiscal_years": []. Never leave sql_params as null for supported companies in FINANCIAL_METRIC intent.
- Canonical Metric Mapping:
  * Profit, net earnings, net income -> 'net_income'
  * Revenue, sales, top line -> 'revenues'
  * Operating income, operating profit -> 'operating_income'
  * Gross profit -> 'gross_profit'
  * Net margin, net profit margin -> 'net_margin'
  * Gross margin -> 'gross_margin'

- Chart Preference Determination:
  * "line": User requests or implies a line chart, trend, growth trajectory, or evolution over time (e.g. 'Show AAPL revenue trend over time', 'plot line of operating margins').
  * "bar": User requests a bar chart or cross-sectional peer comparison (e.g. 'Compare AAPL and MSFT revenue as a bar chart').
  * "scatter": User explicitly requests a scatter plot.
  * "area": User explicitly requests an area chart.
  * "none": Qualitative questions, general explanations, or when no chart is requested or needed (e.g. 'What is Apple's business model?', 'Who is CEO?').
  Default to "none" if ambiguous, unstated, or irrelevant.

Return strictly valid JSON with this exact schema:
{
  "intent": "FINANCIAL_METRIC" | "TEXTUAL_RESEARCH" | "MIXED_RESEARCH" | "DEEP_RESEARCH_REPORT" | "OUT_OF_SCOPE",
  "is_followup": true | false,
  "is_realtime_news": true | false,
  "chart_preference": "line" | "bar" | "scatter" | "area" | "none",
  "target_tools": ["tool_name"],
  "execution_strategy": "parallel" | "sequential" | "direct_abstain",
  "tickers": ["AAPL", "MSFT"] | [],
  "fiscal_years": [YYYY, ...] | [],
  "fiscal_periods": ["FY", "Q1"] | [],
  "sql_params": {
    "tickers": ["AAPL"],
    "metrics": ["revenues"],
    "fiscal_years": [YYYY],
    "fiscal_periods": ["FY"]
  } | null,
  "rag_params": {
    "search_query": "Optimized semantic query string for retrieval",
    "ticker": "AAPL",
    "fiscal_year": YYYY,
    "item": "Item 7"
  } | null,
  "web_params": {
    "search_query": "Optimized web search query for breaking news or recent events",
    "max_results": 5
  } | null,
  "reasoning": "Polite, direct user-facing explanation in the same language as the user query (e.g. Vietnamese if query is in Vietnamese) explaining why this request is out of scope or how it is being resolved."
}"""


class IntentRouter:
    """Autonomous LLM-driven query understanding and tool orchestration router."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        metric_registry: MetricRegistry | None = None,
    ) -> None:
        """Initialize router with optional injected LLM adapter and MetricRegistry."""
        self._llm_adapter = llm_adapter
        self._metric_registry = metric_registry

    @property
    def llm_adapter(self) -> BaseLLMAdapter:
        """Lazy initialization of DeepSeek chat profile adapter."""
        if self._llm_adapter is None:
            self._llm_adapter = LLMFactory.get_adapter(profile="chat")
        return self._llm_adapter

    @property
    def metric_registry(self) -> MetricRegistry:
        """Lazy initialization of MetricRegistry."""
        if self._metric_registry is None:
            self._metric_registry = MetricRegistry()
        return self._metric_registry

    def _supervise_and_validate(self, raw_data: dict, original_query: str) -> RoutingDecision:
        """Supervise LLM planning output, enforce enterprise contracts, and normalize entities.

        Args:
            raw_data: Parsed JSON dictionary from LLM response.
            original_query: Original user query string.

        Returns:
            Validated and sanitized RoutingDecision containing FinancialQueryPlan.
        """
        raw_intent = raw_data.get("intent", "OUT_OF_SCOPE")
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            intent = IntentType.OUT_OF_SCOPE

        # 1. Ticker boundary validation (batch-first with dynamic registry lookup)
        supported_tickers = get_supported_tickers()
        raw_tickers_input = raw_data.get("tickers") or raw_data.get("ticker")
        if isinstance(raw_tickers_input, str):
            raw_tickers = [raw_tickers_input.strip().upper()]
        elif isinstance(raw_tickers_input, list):
            raw_tickers = [t.strip().upper() for t in raw_tickers_input if isinstance(t, str)]
        else:
            raw_tickers = []

        valid_tickers = [t for t in raw_tickers if t in supported_tickers]
        invalid_tickers = [t for t in raw_tickers if t not in supported_tickers]

        # Dynamic VN30 basket expansion when user asks for the entire VN30 index
        q_lower_vn = original_query.lower()
        vn30_markers = get_routing_markers("vn30_basket_markers") or ("vn30",)
        asks_full_vn30 = any(marker in q_lower_vn for marker in vn30_markers)
        if asks_full_vn30:
            vn_tickers = [
                str(c["ticker"]).upper()
                for c in load_company_catalog("configs/companies_vn.yaml")
                if c.get("ticker") and c.get("index_membership", "VN30") == "VN30"
            ]
            for vt in vn_tickers:
                if vt in supported_tickers and vt not in valid_tickers:
                    valid_tickers.append(vt)
            invalid_tickers = [t for t in invalid_tickers if t != "VN30"]
            if intent == IntentType.OUT_OF_SCOPE and valid_tickers:
                intent = IntentType.FINANCIAL_METRIC

        # If tickers were explicitly specified and all are unsupported -> full abstain
        if raw_tickers and not valid_tickers:
            plan = FinancialQueryPlan(
                intent=IntentType.OUT_OF_SCOPE,
                tickers=raw_tickers,
                metrics=[],
                fiscal_years=[],
                fiscal_periods=[],
                requires_sql=False,
                requires_rag=False,
                comparison_type=ComparisonType.NONE,
                chart_preference="none",
            )
            return RoutingDecision(
                intent=IntentType.OUT_OF_SCOPE,
                target_tools=[],
                execution_strategy=ExecutionStrategy.DIRECT_ABSTAIN,
                tickers=raw_tickers,
                chart_preference="none",
                reasoning=f"Company '{', '.join(invalid_tickers)}' is outside the supported universe: {sorted(supported_tickers)}.",
                query_plan=plan,
            )

        # Safety net: if a query asks about filing disclosures for a supported company, never short-circuit as OUT_OF_SCOPE
        if (
            intent == IntentType.OUT_OF_SCOPE
            and valid_tickers
            and not is_realtime_news_query(original_query)
            and re.search(
                r"\b(?:10-k|10-q|20-f|sec\s+filing|item\s+\d+|disclos|filed\s+in|contracts?|thuyết\s+minh|báo\s+cáo)\b",
                original_query,
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:should\s+i\s+buy|should\s+i\s+sell|price\s+target|stock\s+price\s+tomorrow|predict)\b",
                original_query,
                re.IGNORECASE,
            )
        ):
            intent = IntentType.TEXTUAL_RESEARCH

        # 2. Year parsing (Respect LLM's multi-year [] decision instead of single-year regex override)
        sp_raw_init = raw_data.get("sql_params") or {}
        raw_years_input = raw_data.get("fiscal_years") or raw_data.get("fiscal_year")
        if not raw_years_input and sp_raw_init.get("fiscal_years"):
            raw_years_input = sp_raw_init.get("fiscal_years")

        has_explicit_empty_years = ("fiscal_years" in raw_data and raw_data.get("fiscal_years") == []) or (
            "fiscal_years" in sp_raw_init and sp_raw_init.get("fiscal_years") == []
        )

        if isinstance(raw_years_input, int):
            raw_years = [raw_years_input]
        elif isinstance(raw_years_input, list):
            raw_years = [y for y in raw_years_input if isinstance(y, int)]
        else:
            raw_years = []

        if raw_years:
            valid_years = parse_fiscal_years(original_query, explicit_years=raw_years)
        elif has_explicit_empty_years:
            regex_years = parse_fiscal_years(original_query, explicit_years=None)
            valid_years = regex_years if len(regex_years) >= 2 else []
        else:
            valid_years = parse_fiscal_years(original_query, explicit_years=None)

        raw_periods_input = raw_data.get("fiscal_periods") or raw_data.get("fiscal_period")
        if isinstance(raw_periods_input, str):
            periods = [raw_periods_input.strip().upper()]
        elif isinstance(raw_periods_input, list):
            periods = [p.strip().upper() for p in raw_periods_input if isinstance(p, str)]
        else:
            periods = []

        raw_target_tools = raw_data.get("target_tools", [])

        # 3. Metric resolution via MetricRegistry & Layer Partitioning
        candidate_metrics: list[str] = []
        raw_m = raw_data.get("metrics") or raw_data.get("metric")
        if isinstance(raw_m, str):
            candidate_metrics.append(raw_m.strip().lower())
        elif isinstance(raw_m, list):
            candidate_metrics.extend([m.strip().lower() for m in raw_m if isinstance(m, str)])

        sp_raw = raw_data.get("sql_params") or {}
        sp_m = sp_raw.get("metrics") or sp_raw.get("metric")
        if isinstance(sp_m, str):
            candidate_metrics.append(sp_m.strip().lower())
        elif isinstance(sp_m, list):
            candidate_metrics.extend([m.strip().lower() for m in sp_m if isinstance(m, str)])

        seen_cand = set()
        unique_candidates: list[str] = []
        for m in candidate_metrics:
            if m not in seen_cand:
                seen_cand.add(m)
                unique_candidates.append(m)

        resolved_metrics: list[str] = []
        sql_metrics: list[str] = []
        rag_metrics: list[str] = []

        for term in unique_candidates:
            definition = self.metric_registry.resolve(term)
            if definition:
                canonical_key = definition.key
                if canonical_key not in resolved_metrics:
                    resolved_metrics.append(canonical_key)
                if definition.storage_layer == "sql_gold" and canonical_key not in sql_metrics:
                    sql_metrics.append(canonical_key)
                elif definition.storage_layer == "rag_textual" and canonical_key not in rag_metrics:
                    rag_metrics.append(canonical_key)
            else:
                if term not in resolved_metrics:
                    resolved_metrics.append(term)
                if self.metric_registry.is_sql_metric(term):
                    if term not in sql_metrics:
                        sql_metrics.append(term)
                elif self.metric_registry.is_rag_metric(term):
                    if term not in rag_metrics:
                        rag_metrics.append(term)
                else:
                    if "financial_sql_tool" in raw_target_tools and term not in sql_metrics:
                        sql_metrics.append(term)

        # Fallback metric inference for financial questions without explicit metric key
        if intent == IntentType.FINANCIAL_METRIC and not resolved_metrics:
            q_lower = original_query.lower()
            if any(k in q_lower for k in ["net income", "profit"]):
                inferred = "net_income"
            elif any(k in q_lower for k in ["revenue", "sales"]):
                inferred = "revenues"
            elif any(k in q_lower for k in ["operating income", "operating"]):
                inferred = "operating_income"
            else:
                inferred = "revenues"
            resolved_metrics.append(inferred)
            sql_metrics.append(inferred)

        # 4. Tool Requirement Flags, Contextual Isolation & Intent Auto-promotion
        is_followup = bool(raw_data.get("is_followup", False))
        is_realtime_news = bool(raw_data.get("is_realtime_news", False)) or is_realtime_news_query(original_query)
        mentions_sql_metric = _query_mentions_sql_metric(original_query, self.metric_registry)

        requires_sql = bool(sql_metrics) or "financial_sql_tool" in raw_target_tools
        requires_web = (
            "web_search_tool" in raw_target_tools
            or (raw_data.get("web_params") is not None)
            or (is_realtime_news and intent != IntentType.OUT_OF_SCOPE)
        )

        # Prevent pure live-news queries from implicitly forcing RAG or leaking historical SQL dumps
        if is_realtime_news and intent != IntentType.OUT_OF_SCOPE:
            if not is_followup:
                catalog = load_company_catalog()
                q_lower_iso = original_query.lower()
                explicit_tickers = [t for t in valid_tickers if _is_ticker_mentioned_in_query(t, q_lower_iso, catalog)]
                # Strip tickers leaked from prior turns when not mentioned in the current query
                if valid_tickers and len(explicit_tickers) < len(valid_tickers):
                    valid_tickers = explicit_tickers

            if not mentions_sql_metric and not is_followup:
                requires_sql = False
                sql_metrics = []

            filing_disclosure_markers = get_routing_markers("filing_disclosure_markers")
            asks_filing_narrative = bool(rag_metrics) or any(
                m in original_query.lower() for m in filing_disclosure_markers
            )
            if not asks_filing_narrative:
                requires_rag = False
            else:
                requires_rag = bool(rag_metrics) or "hybrid_retriever" in raw_target_tools
        else:
            is_web_only_textual = (
                requires_web
                and not requires_sql
                and "hybrid_retriever" not in raw_target_tools
                and not rag_metrics
                and raw_data.get("rag_params") is None
            )
            requires_rag = (
                bool(rag_metrics)
                or "hybrid_retriever" in raw_target_tools
                or (
                    not is_web_only_textual
                    and intent
                    in (
                        IntentType.TEXTUAL_RESEARCH,
                        IntentType.MIXED_RESEARCH,
                        IntentType.DEEP_RESEARCH_REPORT,
                    )
                )
            )

        if intent == IntentType.OUT_OF_SCOPE:
            # Guard: High-precision live news queries should route to web_search_tool
            # (and cleanly abstain if web fails/returns empty) rather than skipping web search
            if is_realtime_news_query(original_query) and not raw_tickers:
                intent = IntentType.TEXTUAL_RESEARCH
                requires_sql = False
                requires_rag = False
                requires_web = True
            else:
                requires_sql = False
                requires_rag = False
                requires_web = False

        # Auto-promote to MIXED_RESEARCH if multiple complementary layers are demanded,
        # or demote back to TEXTUAL_RESEARCH if SQL was stripped for pure live-news queries
        if (requires_sql and (requires_rag or requires_web)) and intent != IntentType.DEEP_RESEARCH_REPORT:
            intent = IntentType.MIXED_RESEARCH
        elif (
            (requires_rag or requires_web)
            and not requires_sql
            and intent
            in (
                IntentType.FINANCIAL_METRIC,
                IntentType.MIXED_RESEARCH,
            )
        ):
            intent = IntentType.TEXTUAL_RESEARCH

        # Determine target tools and execution strategy
        if intent == IntentType.OUT_OF_SCOPE:
            target_tools = []
            strategy = ExecutionStrategy.DIRECT_ABSTAIN
        else:
            target_tools = []
            if requires_sql:
                target_tools.append("financial_sql_tool")
            if requires_rag:
                target_tools.append("hybrid_retriever")
            if requires_web:
                target_tools.append("web_search_tool")
            strategy = ExecutionStrategy.PARALLEL if target_tools else ExecutionStrategy.DIRECT_ABSTAIN

        # 5. Semantic Comparison Classification
        q_lower = original_query.lower()
        has_trend_keywords = any(k in q_lower for k in ["compare", "comparison", "trend", "chart", "over time"])
        is_multi_company = len(valid_tickers) > 1
        is_multi_year = len(valid_years) > 1 or (not valid_years and has_trend_keywords)
        is_multi_metric = len(resolved_metrics) > 1

        if is_multi_company and (is_multi_metric or is_multi_year):
            comparison_type = ComparisonType.COMPREHENSIVE
        elif is_multi_company:
            comparison_type = ComparisonType.CROSS_COMPANY
        elif is_multi_year or len(periods) > 1:
            comparison_type = ComparisonType.TIME_SERIES
        elif is_multi_metric:
            comparison_type = ComparisonType.MULTI_METRIC
        else:
            comparison_type = ComparisonType.NONE

        # 6. Chart Preference Resolution & FinancialQueryPlan Synthesis
        raw_pref = raw_data.get("chart_preference")
        allowed_prefs = {"line", "bar", "scatter", "area", "none"}
        if isinstance(raw_pref, str) and raw_pref.strip().lower() in allowed_prefs:
            chart_preference = raw_pref.strip().lower()
        else:
            if raw_pref is not None:
                logger.warning("Unrecognized chart_preference '%s' from LLM. Falling back to 'none'.", raw_pref)
            chart_preference = "none"

        query_plan = FinancialQueryPlan(
            intent=intent,
            tickers=valid_tickers,
            metrics=resolved_metrics,
            fiscal_years=valid_years,
            fiscal_periods=periods,
            requires_sql=requires_sql,
            requires_rag=requires_rag,
            requires_web=requires_web,
            comparison_type=comparison_type,
            sql_metrics=sql_metrics,
            rag_metrics=rag_metrics,
            chart_preference=chart_preference,
        )

        if intent == IntentType.OUT_OF_SCOPE:
            return RoutingDecision(
                intent=IntentType.OUT_OF_SCOPE,
                target_tools=[],
                execution_strategy=ExecutionStrategy.DIRECT_ABSTAIN,
                tickers=valid_tickers or raw_tickers,
                fiscal_years=valid_years or raw_years,
                fiscal_periods=periods,
                chart_preference="none",
                reasoning=raw_data.get("reasoning", "Query identified as out-of-scope."),
                query_plan=query_plan,
            )

        # 7. Parse SQL Params if required
        sql_params: SQLQueryParams | None = None
        if requires_sql and valid_tickers:
            metrics_for_sql = sql_metrics or (["revenues"] if intent == IntentType.FINANCIAL_METRIC else [])
            if asks_full_vn30 and "revenues" in metrics_for_sql and "total_operating_income" not in metrics_for_sql:
                metrics_for_sql = list(metrics_for_sql) + ["total_operating_income"]
                if "total_operating_income" not in resolved_metrics:
                    resolved_metrics.append("total_operating_income")
            if metrics_for_sql:
                sql_params = SQLQueryParams(
                    tickers=valid_tickers,
                    metrics=metrics_for_sql,
                    fiscal_years=valid_years,
                    fiscal_periods=periods,
                )

        # 8. Parse RAG Params if required
        rag_params: RAGQueryParams | None = None
        if requires_rag:
            rp = raw_data.get("rag_params") or {}
            # Multi-Entity and Multi-Year Handling:
            # When comparing multiple companies or years, do not restrict RAG to a single entity or single year
            primary_ticker = valid_tickers[0] if len(valid_tickers) == 1 else None

            base_query = rp.get("search_query") or original_query
            if rag_metrics and not rp.get("search_query"):
                rag_terms = " ".join(rag_metrics)
                base_query = f"{primary_ticker or ''} {rag_terms} {original_query}".strip()

            rag_params = RAGQueryParams(
                search_query=base_query,
                ticker=rp.get("ticker") if len(valid_tickers) == 1 else None,
                fiscal_year=rp.get("fiscal_year") if len(valid_years) == 1 else None,
                item=rp.get("item"),
            )

        # 9. Parse Web Search Params if required
        web_params: WebSearchQueryParams | None = None
        if requires_web:
            wp = raw_data.get("web_params") or {}
            web_query = wp.get("search_query") or original_query
            max_res = wp.get("max_results") or 5
            web_params = WebSearchQueryParams(
                search_query=web_query,
                max_results=max_res,
            )

        return RoutingDecision(
            intent=intent,
            target_tools=target_tools,
            execution_strategy=strategy,
            sql_params=sql_params,
            rag_params=rag_params,
            web_params=web_params,
            tickers=valid_tickers,
            fiscal_years=valid_years,
            fiscal_periods=periods,
            chart_preference=chart_preference,
            is_followup=is_followup,
            is_realtime_news=is_realtime_news,
            reasoning=raw_data.get("reasoning", "Autonomous LLM tool planning complete."),
            query_plan=query_plan,
        )

    def route(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Autonomously route user query by empowering the LLM planner with schema supervision.

        Args:
            query: Raw user inquiry string.
            conversation_history: Optional list of previous chat turns for contextual continuity.

        Returns:
            Guaranteed validated RoutingDecision with tool-specific parameters.
        """
        user_prompt = query
        if conversation_history and should_attach_history(query):
            history_lines = ["\n\n### Prior Conversation Context:"]
            for msg in conversation_history[-4:]:
                r = msg.get("role", "user")
                c = msg.get("content", "")
                c_short = c[:160].replace("\n", " ").strip()
                history_lines.append(f"- {r.capitalize()}: {c_short}")
            user_prompt += "\n".join(history_lines)

        messages = [
            ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_adapter.generate(
                messages=messages,
                temperature=0.0,
                json_mode=True,
            )
            data = json.loads(response.content)
            decision = self._supervise_and_validate(data, query)
            if response.usage is not None:
                usage_payload = response.usage.model_dump() | {"model": response.model}
                return decision.model_copy(update={"token_usage": usage_payload})
            return decision

        except Exception as exc:
            logger.warning("LLM Tool Planning encountered error: %s. Emitting safe fallback.", exc)
            fallback_plan = FinancialQueryPlan(
                intent=IntentType.OUT_OF_SCOPE,
                tickers=[],
                metrics=[],
                fiscal_years=[],
                fiscal_periods=[],
                requires_sql=False,
                requires_rag=False,
                comparison_type=ComparisonType.NONE,
                chart_preference="none",
            )
            return RoutingDecision(
                intent=IntentType.OUT_OF_SCOPE,
                target_tools=[],
                execution_strategy=ExecutionStrategy.DIRECT_ABSTAIN,
                chart_preference="none",
                reasoning=f"Planner error fallback: {exc}",
                query_plan=fallback_plan,
            )


# ---------------------------------------------------------------------------
# LangGraph Node and Conditional Edge Functions
# ---------------------------------------------------------------------------


def router_node(state: AgentState, router: IntentRouter | None = None) -> dict[str, Any]:
    """LangGraph node executing autonomous intent planning and parameter synthesis.

    Args:
        state: Current AgentState.
        router: Injected or default IntentRouter instance.

    Returns:
        State dictionary update for LangGraph.
    """
    query = state.get("query", "").strip()
    history = state.get("conversation_history") or []
    active_router = router or IntentRouter()
    decision = active_router.route(query, conversation_history=history)

    # 1. Understanding question step
    step_understanding = {
        "step": "Understanding question",
        "status": "completed",
        "description": f"Classified intent as {decision.intent.value} ({decision.execution_strategy.value})",
        "duration_ms": None,
    }

    # 2. Resolving metrics step
    metrics_desc = (
        f"Resolved tickers: {decision.tickers or ['none']}, "
        f"metrics: {decision.query_plan.metrics if decision.query_plan else []}"
    )
    step_metrics = {
        "step": "Resolving metrics",
        "status": "completed" if decision.intent != IntentType.OUT_OF_SCOPE else "skipped",
        "description": metrics_desc if decision.intent != IntentType.OUT_OF_SCOPE else "Skipped for out-of-scope query",
        "duration_ms": None,
    }

    state_update: dict[str, Any] = {
        "intent": decision.intent.value,
        "routing_decision": decision.model_dump(),
        "query_plan": decision.query_plan.model_dump() if decision.query_plan else None,
        "sql_params": decision.sql_params.model_dump() if decision.sql_params else None,
        "rag_params": decision.rag_params.model_dump() if decision.rag_params else None,
        "web_params": decision.web_params.model_dump() if decision.web_params else None,
        "execution_strategy": decision.execution_strategy.value,
        "activity_timeline": [step_understanding, step_metrics],
    }
    if decision.token_usage:
        state_update["llm_usage"] = [decision.token_usage]
    return state_update


def route_next_step(state: AgentState) -> list[str] | str:
    """LangGraph conditional edge function determining fan-out or direct routing.

    Args:
        state: Current AgentState with populated intent and query_plan.

    Returns:
        Single node name or list of node names for parallel fan-out.
    """
    intent = state.get("intent")

    if intent == IntentType.OUT_OF_SCOPE.value:
        return "abstention_node"

    routing_decision = state.get("routing_decision") or {}
    target_tools = routing_decision.get("target_tools") or []

    query_plan = state.get("query_plan") or {}
    requires_sql = query_plan.get("requires_sql", False) or ("financial_sql_tool" in target_tools)
    requires_rag = query_plan.get("requires_rag", False) or ("hybrid_retriever" in target_tools)
    requires_web = query_plan.get("requires_web", False) or ("web_search_tool" in target_tools)

    selected_nodes: list[str] = []
    if requires_sql:
        selected_nodes.append("sql_node")
    if requires_rag:
        selected_nodes.append("rag_node")
    if requires_web:
        selected_nodes.append("web_node")

    if len(selected_nodes) > 1:
        return selected_nodes
    if len(selected_nodes) == 1:
        return selected_nodes[0]

    if intent == IntentType.FINANCIAL_METRIC.value:
        return "sql_node"

    if intent == IntentType.TEXTUAL_RESEARCH.value:
        return "rag_node"

    if intent in (IntentType.MIXED_RESEARCH.value, IntentType.DEEP_RESEARCH_REPORT.value):
        # Parallel fan-out in LangGraph
        return ["sql_node", "rag_node"]

    return "abstention_node"
