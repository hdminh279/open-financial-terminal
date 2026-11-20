"""Domain models and contracts for the Financial Research Agent.

Defines strongly-typed Pydantic contracts for intent routing decisions,
tool execution results, verifiable evidence items, citations, abstention metadata,
structured chart data, report sections, and unified agent request/response envelopes.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class IntentType(str, Enum):
    """Authoritative classification of user query intent."""

    FINANCIAL_METRIC = "FINANCIAL_METRIC"      # Structured numerical metrics via SQL Tool
    TEXTUAL_RESEARCH = "TEXTUAL_RESEARCH"      # Qualitative textual research via RAG
    MIXED_RESEARCH = "MIXED_RESEARCH"          # Hybrid requiring both SQL and RAG
    DEEP_RESEARCH_REPORT = "DEEP_RESEARCH_REPORT"  # Comprehensive multi-section report with charts
    OUT_OF_SCOPE = "OUT_OF_SCOPE"              # Unsupported company, general knowledge, or speculative prediction


class ComparisonType(str, Enum):
    """Classification of analytical comparison in the user query."""

    NONE = "none"                      # Single company, single metric, single period
    TIME_SERIES = "time_series"        # Multi-year or multi-period trend
    CROSS_COMPANY = "cross_company"    # Peer comparison across multiple companies
    MULTI_METRIC = "multi_metric"      # Multiple metrics for single entity
    COMPREHENSIVE = "comprehensive"    # Multi-company and multi-year/multi-metric


class ChartType(str, Enum):
    """Supported chart visualization types."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    AREA = "area"
    NONE = "none"


class FinancialQueryPlan(BaseModel):
    """Structured execution plan synthesized by Intent Router with Metric Registry routing."""

    model_config = ConfigDict(frozen=True)

    intent: IntentType = Field(..., description="High-level query intent")
    tickers: List[str] = Field(default_factory=list, description="Target companies")
    metrics: List[str] = Field(default_factory=list, description="Resolved canonical metric keys")
    fiscal_years: List[int] = Field(default_factory=list, description="Target fiscal years")
    fiscal_periods: List[str] = Field(default_factory=list, description="Target fiscal periods")
    requires_sql: bool = Field(default=False, description="Flag indicating if SQL Gold layer is needed")
    requires_rag: bool = Field(default=False, description="Flag indicating if RAG Textual layer is needed")
    comparison_type: ComparisonType = Field(default=ComparisonType.NONE, description="Analytical comparison type")
    sql_metrics: List[str] = Field(default_factory=list, description="Canonical metrics routed to SQL Gold")
    rag_metrics: List[str] = Field(default_factory=list, description="Canonical or segment metrics routed to RAG")
    chart_preference: Literal["line", "bar", "scatter", "area", "none"] = Field(
        default="none",
        description="LLM-selected visualization style: 'line', 'bar', 'scatter', 'area', or 'none'",
    )


class ChartData(BaseModel):
    """Structured data contract for UI visualization and Markdown charts."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Descriptive title of the chart")
    chart_type: ChartType = Field(..., description="Visualization format: bar or line")
    x_labels: List[str] = Field(..., description="Ordered category labels (e.g. ['FY2022', 'FY2023', 'FY2024', 'FY2025'])")
    datasets: List[Dict[str, Any]] = Field(
        ...,
        description="Dataset series, e.g. [{'label': 'Revenues ($B)', 'values': [394.33, 383.29, 391.04]}]",
    )
    unit: str = Field(default="USD", description="Unit of measurement (USD, Percent, Ratio)")
    mermaid_code: Optional[str] = Field(
        default=None,
        description="Renderable Mermaid xychart-beta block for native Markdown rendering",
    )
    ascii_chart: Optional[str] = Field(
        default=None,
        description="Terminal-friendly ASCII bar representation",
    )
    plotly_figure_json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Spec-driven Plotly figure JSON (data, layout, config) for modern workstation rendering",
    )


class ReportSection(BaseModel):
    """Individual chapter in an equity research report."""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(..., description="Unique section identifier (e.g. 'sec_financial_overview')")
    title: str = Field(..., description="Section heading")
    content: str = Field(..., description="Factual analysis with citation markers [1], [2]")
    citations: List[int] = Field(default_factory=list, description="Citation IDs referenced in this section")
    chart: Optional[ChartData] = Field(default=None, description="Optional embedded visualization")


class ToolExecutionStatus(str, Enum):
    """Execution status for agent tools."""

    SUCCESS = "success"              # Tool executed cleanly and retrieved records
    NO_DATA_FOUND = "no_data_found"  # Tool executed cleanly but returned zero records
    TOOL_FAILED = "tool_failed"      # System exception, network error, or runtime crash


class EvidenceSourceType(str, Enum):
    """Origin of collected evidence."""

    SQL = "sql"
    RAG = "rag"


class ExecutionStrategy(str, Enum):
    """Tool invocation orchestration strategy decided by LLM."""

    PARALLEL = "parallel"          # Execute tools concurrently (e.g. independent SQL and RAG)
    SEQUENTIAL = "sequential"      # Execute sequentially with dependency
    DIRECT_ABSTAIN = "direct_abstain"  # Immediate abstention without calling any tool


class AbstentionReason(str, Enum):
    """Granular categorization of why an answer or portion was abstained."""

    UNSUPPORTED_COMPANY = "unsupported_company"      # Outside AAPL, MSFT, GOOGL universe
    OUT_OF_RANGE_YEAR = "out_of_range_year"          # Outside verified available filing periods in lakehouse
    SPECULATIVE_PREDICTION = "speculative_prediction"# Stock price target, future forecast
    INVESTMENT_ADVICE = "investment_advice"          # Buy/sell financial recommendations
    UNSUPPORTED_TOPIC = "unsupported_topic"          # Completely unrelated to SEC filings
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # Context does not contain the answer
    TOOL_ERROR = "tool_error"                        # Underlying tool failed execution


class SQLQueryParams(BaseModel):
    """Structured parameters planned by LLM for FinancialSQLTool in batch-first mode."""

    model_config = ConfigDict(frozen=True)

    tickers: List[str] = Field(default_factory=list, description="Target ticker symbols (AAPL, MSFT, GOOGL)")
    metrics: List[str] = Field(default_factory=list, description="Canonical metric names from statements or ratios")
    fiscal_years: List[int] = Field(default_factory=list, description="Fiscal years")
    fiscal_periods: List[str] = Field(default_factory=list, description="Periods (Q1, Q2, Q3, Q4, FY)")


class RAGQueryParams(BaseModel):
    """Structured parameters planned by LLM for HybridRetriever."""

    model_config = ConfigDict(frozen=True)

    search_query: str = Field(
        ...,
        description="Refined semantic search query optimized for vector and BM25 retrieval",
    )
    ticker: Optional[str] = Field(default=None, description="Filter ticker if applicable")
    fiscal_year: Optional[int] = Field(default=None, description="Filter fiscal year if applicable")
    item: Optional[str] = Field(
        default=None,
        description="Filing section filter (e.g. 'Item 1A', 'Item 7', 'Item 1')",
    )


class RoutingDecision(BaseModel):
    """Structured decision produced autonomously by the Intent Router LLM."""

    model_config = ConfigDict(frozen=True)

    intent: IntentType = Field(..., description="Classified intent of the user query")
    target_tools: List[str] = Field(
        default_factory=list,
        description="Names of tools to execute (e.g. ['financial_sql_tool', 'hybrid_retriever'])",
    )
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Execution workflow selected by LLM",
    )
    sql_params: Optional[SQLQueryParams] = Field(
        default=None,
        description="Targeted parameters for SQL Tool if selected",
    )
    rag_params: Optional[RAGQueryParams] = Field(
        default=None,
        description="Targeted parameters for RAG Retriever if selected",
    )
    tickers: List[str] = Field(
        default_factory=list,
        description="Primary normalized ticker symbols (AAPL, MSFT, GOOGL)",
    )
    fiscal_years: List[int] = Field(
        default_factory=list,
        description="Primary fiscal years",
    )
    fiscal_periods: List[str] = Field(
        default_factory=list,
        description="Primary fiscal periods (Q1-Q4, FY)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="LLM explanation of tool selection and workflow planning",
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Any missing parameters needed for complete answering",
    )
    chart_preference: Literal["line", "bar", "scatter", "area", "none"] = Field(
        default="none",
        description="LLM-selected visualization style: 'line', 'bar', 'scatter', 'area', or 'none'",
    )
    query_plan: Optional[FinancialQueryPlan] = Field(
        default=None,
        description="Structured analytical query plan with Metric Registry routing",
    )


class ToolExecutionResult(BaseModel):
    """Standardized output envelope wrapping execution of a single tool."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Canonical tool identifier")
    status: ToolExecutionStatus = Field(..., description="Execution outcome")
    data: Optional[Any] = Field(
        default=None,
        description="Raw output data (e.g., FinancialQueryResponse or RetrievalResponse)",
    )
    execution_time_ms: float = Field(default=0.0, description="Latency in milliseconds")
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed exception message if status == TOOL_FAILED",
    )


class EvidenceItem(BaseModel):
    """Normalized evidence record collected from SQL or RAG for citation generation."""

    model_config = ConfigDict(frozen=True)

    citation_id: int = Field(..., description="Sequential reference identifier [1], [2]")
    source_type: EvidenceSourceType = Field(..., description="Originating engine (sql/rag)")
    ticker: str = Field(..., description="Company ticker symbol")
    form: Optional[str] = Field(default=None, description="SEC filing form (10-K, 10-Q)")
    fiscal_year: Optional[int] = Field(default=None, description="Fiscal year")
    fiscal_period: Optional[str] = Field(default=None, description="Fiscal period (Q1-Q4, FY)")
    item_or_metric: str = Field(
        ...,
        description="Item title (e.g. 'Item 1A. Risk Factors') or canonical metric name",
    )
    content_summary: str = Field(..., description="Brief factual excerpt or formatted value")
    source_url: str = Field(
        ...,
        description="Authoritative clickable SEC EDGAR URL preserved from source",
    )
    accession_number: Optional[str] = Field(
        default=None,
        description="SEC filing accession number",
    )


class Citation(BaseModel):
    """Formatted reference entry suitable for display in response footnotes."""

    model_config = ConfigDict(frozen=True)

    citation_id: int = Field(..., description="Citation marker number (e.g. 1)")
    formatted_reference: str = Field(
        ...,
        description="Human-readable citation string with clickable Markdown link",
    )
    source_url: str = Field(..., description="Underlying destination SEC URL")


class AbstentionInfo(BaseModel):
    """Metadata regarding full or partial query abstention."""

    model_config = ConfigDict(frozen=True)

    is_abstaining: bool = Field(..., description="True if any portion of query was abstained")
    is_partial: bool = Field(
        default=False,
        description="True if some parts were answered while other parts were abstained",
    )
    reason: Optional[AbstentionReason] = Field(
        default=None,
        description="Primary cause of abstention",
    )
    explanation: str = Field(
        default="",
        description="User-facing transparent explanation of the abstention reason",
    )
    unsupported_aspects: List[str] = Field(
        default_factory=list,
        description="Specific unsupported sub-topics or entities",
    )


class ActivityTimelineStep(BaseModel):
    """Normalized step record reflecting real runtime execution in LangGraph pipeline."""

    model_config = ConfigDict(frozen=True)

    step: str = Field(..., description="Canonical step identifier (e.g. 'Understanding question')")
    status: Literal["completed", "failed", "skipped", "in_progress"] = Field(
        default="completed",
        description="Execution status of this specific step",
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed context, row counts, or tool latency",
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Execution latency in milliseconds",
    )


class AgentRequest(BaseModel):
    """Input query request payload for the Research Agent."""

    query: str = Field(..., min_length=1, description="Natural language user question")
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional session context or preferences",
    )


class AgentResponse(BaseModel):
    """Final comprehensive response returned by the Research Agent."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original user question")
    intent: IntentType = Field(..., description="Detected query intent")
    answer: str = Field(..., description="Synthesized answer text with [1] citation markers")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Ordered list of verifiable source citations",
    )
    charts: List[ChartData] = Field(
        default_factory=list,
        description="Generated financial visual charts (Mermaid & structured datasets)",
    )
    report_sections: List[ReportSection] = Field(
        default_factory=list,
        description="Structured report chapters if intent == DEEP_RESEARCH_REPORT",
    )
    abstention: Optional[AbstentionInfo] = Field(
        default=None,
        description="Abstention metadata if applicable",
    )
    tool_results: List[ToolExecutionResult] = Field(
        default_factory=list,
        description="Execution trace and telemetry from involved tools",
    )
    query_plan: Optional[FinancialQueryPlan] = Field(
        default=None,
        description="Execution query plan with tool routing and comparison semantics",
    )
    execution_time_ms: float = Field(default=0.0, description="Total pipeline latency in ms")
    activity_timeline: List[ActivityTimelineStep] = Field(
        default_factory=list,
        description="Ordered sequence of executed pipeline steps with latencies and statuses",
    )
