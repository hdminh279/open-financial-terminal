"""API Contracts and Pydantic Domain Schemas for FastAPI Backend.

Defines request and response payloads for natural language research queries,
direct financial metrics, company metadata, chat session persistence, and health probes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.ai.agent.contracts import (
    AbstentionInfo,
    ActivityTimelineStep,
    ChartData,
    Citation,
    IntentType,
    ReportSection,
    ToolExecutionResult,
)


class QueryRequest(BaseModel):
    """Incoming user query payload for the research agent."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, description="Natural language user question")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation session ID to persist or append to chat history",
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional contextual filters or user preferences",
    )


class QueryResponse(BaseModel):
    """Comprehensive API response envelope returned by the research endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original user query")
    session_id: str = Field(..., description="Active conversation session ID")
    intent: IntentType = Field(..., description="Classified intent from the LangGraph agent")
    answer: str = Field(..., description="Synthesized grounded answer with citation markers")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Clickable citations with verified SEC EDGAR URLs",
    )
    charts: List[ChartData] = Field(
        default_factory=list,
        description="Visual chart data specifications (Bar / Line / Mermaid)",
    )
    report_sections: List[ReportSection] = Field(
        default_factory=list,
        description="Structured chapters if deep equity research report was requested",
    )
    abstention: Optional[AbstentionInfo] = Field(
        default=None,
        description="Abstention metadata if query was out of scope or evidence insufficient",
    )
    tool_results: List[ToolExecutionResult] = Field(
        default_factory=list,
        description="Telemetry and raw results from executed tools",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique correlation identifier matching backend telemetry",
    )
    activity_timeline: List[ActivityTimelineStep] = Field(
        default_factory=list,
        description="Ordered sequence of executed pipeline steps with latencies and statuses",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Total roundtrip execution latency in milliseconds",
    )


class MetricRecordResponse(BaseModel):
    """Standardized single metric observation record."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: int = Field(..., description="Fiscal reporting year")
    fiscal_period: str = Field(..., description="Fiscal period (Q1-Q4, FY)")
    period_type: str = Field(..., description="Duration or instant period type")
    metric_name: str = Field(..., description="Standardized canonical metric name")
    value: Optional[float] = Field(default=None, description="Numeric metric value")
    unit: str = Field(..., description="Unit of measurement (USD, Shares, etc.)")
    filed_date: Optional[str] = Field(default=None, description="Official SEC filing date")
    form: Optional[str] = Field(default=None, description="SEC filing form (10-K or 10-Q)")
    provenance_url: Optional[str] = Field(
        default=None,
        description="Clickable SEC EDGAR archive provenance URL",
    )


class MetricResponse(BaseModel):
    """Response payload for direct financial metric queries in batch-first mode."""

    model_config = ConfigDict(frozen=True)

    tickers: List[str] = Field(default_factory=list, description="Queried company tickers")
    metrics: List[str] = Field(default_factory=list, description="Queried canonical metric names")
    count: int = Field(..., description="Total observation records returned")
    records: List[MetricRecordResponse] = Field(
        default_factory=list,
        description="Chronological metric records",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Database retrieval latency in milliseconds",
    )

    @computed_field
    def ticker(self) -> str:
        """Scalar convenience for single-ticker callers."""
        return self.tickers[0] if self.tickers else ""

    @computed_field
    def metric(self) -> str:
        """Scalar convenience for single-metric callers."""
        return self.metrics[0] if self.metrics else ""


class CompanyItem(BaseModel):
    """Summary of a supported company entity."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Trading ticker symbol")
    name: str = Field(..., description="Official company entity name")
    cik: str = Field(..., description="10-digit SEC Central Index Key")
    supported_years: List[int] = Field(
        default_factory=list,
        description="Available reporting fiscal years in Gold Marts",
    )
    available_forms: List[str] = Field(
        default=["10-K", "10-Q"],
        description="Supported SEC filing form types",
    )


class CompanyListResponse(BaseModel):
    """Collection of all companies available in the intelligence engine."""

    model_config = ConfigDict(frozen=True)

    companies: List[CompanyItem] = Field(
        default_factory=list,
        description="List of configured enterprise entities",
    )


class ChatMessage(BaseModel):
    """Individual conversational turn in a session."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(..., description="Unique message UUID")
    role: Literal["user", "assistant"] = Field(..., description="Author role")
    content: str = Field(..., description="Message text or synthesized response")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional telemetry, intent, citations, or chart payload",
    )


class ChatSession(BaseModel):
    """Full conversational session including complete message history."""

    session_id: str = Field(..., description="Unique session UUID")
    title: str = Field(default="New Research Session", description="Session conversation topic")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation ISO-8601 timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last updated ISO-8601 timestamp",
    )
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Chronological ordered conversation turns",
    )


class ChatSessionSummary(BaseModel):
    """Compact summary of a conversational session for sidebar listing."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Unique session UUID")
    title: str = Field(..., description="Conversation title")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Session last updated timestamp")
    message_count: int = Field(..., description="Number of turns in session")


class ChatSessionListResponse(BaseModel):
    """Listing of all saved conversation sessions."""

    model_config = ConfigDict(frozen=True)

    sessions: List[ChatSessionSummary] = Field(
        default_factory=list,
        description="List of active or archived chat sessions",
    )


class HealthResponse(BaseModel):
    """Readiness and liveness probe response."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status",
    )
    minio_connected: bool = Field(..., description="MinIO S3 storage accessibility")
    qdrant_connected: bool = Field(..., description="Qdrant vector database accessibility")
    duckdb_ready: bool = Field(..., description="DuckDB Gold Parquet lakehouse accessibility")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Probe execution timestamp",
    )
