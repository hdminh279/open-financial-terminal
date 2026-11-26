"""Backend Composition Root and FastAPI dependency injection providers.

Centralizes thread-safe lazy singletons for ResearchAgent, ChatHistoryManager,
DuckDBFinancialRepository, FinancialSQLTool, HybridRetriever, S3Storage,
QdrantStorage, QuoteService, and WatchlistRepository with clean shutdown support.
"""

import threading

from backend.ai.agent.graph import build_agent_graph
from backend.ai.agent.research_agent import ResearchAgent
from backend.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from backend.ai.agent.tools.financial_sql_tool import FinancialSQLTool
from backend.ai.llm.factory import LLMFactory
from backend.ai.retrieval.retriever import HybridRetriever
from backend.ai.vectordb.qdrant_client import QdrantStorage
from backend.core.storage.s3_storage import S3Storage
from backend.services.chat_history import ChatHistoryManager
from backend.services.market.service import QuoteService, WatchlistRepository

_agent_instance: ResearchAgent | None = None
_history_instance: ChatHistoryManager | None = None
_duckdb_repo_instance: DuckDBFinancialRepository | None = None
_sql_tool_instance: FinancialSQLTool | None = None
_s3_instance: S3Storage | None = None
_qdrant_instance: QdrantStorage | None = None
_retriever_instance: HybridRetriever | None = None
_quote_service_instance: QuoteService | None = None
_watchlist_repo_instance: WatchlistRepository | None = None
_DEPS_LOCK = threading.Lock()


def get_duckdb_repository() -> DuckDBFinancialRepository:
    """Retrieve or lazily initialize the singleton DuckDBFinancialRepository."""
    global _duckdb_repo_instance
    if _duckdb_repo_instance is None:
        with _DEPS_LOCK:
            if _duckdb_repo_instance is None:
                _duckdb_repo_instance = DuckDBFinancialRepository()
    return _duckdb_repo_instance


def get_sql_tool() -> FinancialSQLTool:
    """Retrieve or lazily initialize the singleton FinancialSQLTool sharing DuckDBFinancialRepository."""
    global _sql_tool_instance
    if _sql_tool_instance is None:
        repo = get_duckdb_repository()
        with _DEPS_LOCK:
            if _sql_tool_instance is None:
                _sql_tool_instance = FinancialSQLTool(repository=repo)
    return _sql_tool_instance


def get_s3_storage() -> S3Storage | None:
    """Retrieve or lazily initialize the S3Storage adapter."""
    global _s3_instance
    if _s3_instance is None:
        with _DEPS_LOCK:
            if _s3_instance is None:
                try:
                    _s3_instance = S3Storage()
                except Exception:
                    _s3_instance = None
    return _s3_instance


def get_qdrant_storage() -> QdrantStorage | None:
    """Retrieve or lazily initialize the QdrantStorage adapter."""
    global _qdrant_instance
    if _qdrant_instance is None:
        with _DEPS_LOCK:
            if _qdrant_instance is None:
                try:
                    _qdrant_instance = QdrantStorage()
                except Exception:
                    _qdrant_instance = None
    return _qdrant_instance


def get_retriever() -> HybridRetriever:
    """Retrieve or lazily initialize the singleton HybridRetriever sharing QdrantStorage."""
    global _retriever_instance
    if _retriever_instance is None:
        qdrant = get_qdrant_storage()
        with _DEPS_LOCK:
            if _retriever_instance is None:
                _retriever_instance = HybridRetriever(qdrant_storage=qdrant)
    return _retriever_instance


def get_quote_service() -> QuoteService:
    """Retrieve or lazily initialize the singleton QuoteService."""
    global _quote_service_instance
    if _quote_service_instance is None:
        with _DEPS_LOCK:
            if _quote_service_instance is None:
                _quote_service_instance = QuoteService()
    return _quote_service_instance


def get_watchlist_repository() -> WatchlistRepository:
    """Retrieve or lazily initialize the singleton WatchlistRepository sharing S3Storage and QuoteService."""
    global _watchlist_repo_instance
    if _watchlist_repo_instance is None:
        quote_svc = get_quote_service()
        s3 = get_s3_storage()
        with _DEPS_LOCK:
            if _watchlist_repo_instance is None:
                _watchlist_repo_instance = WatchlistRepository(s3_storage=s3, quote_service=quote_svc)
    return _watchlist_repo_instance


def get_chat_history() -> ChatHistoryManager:
    """Retrieve or lazily initialize the singleton ChatHistoryManager sharing S3Storage."""
    global _history_instance
    if _history_instance is None:
        s3 = get_s3_storage()
        with _DEPS_LOCK:
            if _history_instance is None:
                _history_instance = ChatHistoryManager(s3_storage=s3)
    return _history_instance


def get_research_agent() -> ResearchAgent:
    """Retrieve or lazily initialize the singleton ResearchAgent wired with shared tool singletons."""
    global _agent_instance
    if _agent_instance is None:
        with _DEPS_LOCK:
            if _agent_instance is None:
                compiled = build_agent_graph(
                    sql_tool_provider=get_sql_tool,
                    retriever_provider=get_retriever,
                )
                _agent_instance = ResearchAgent(compiled_graph=compiled)
    return _agent_instance


def close_all_services() -> None:
    """Release pooled HTTP clients across market data and LLM adapters on app shutdown."""
    with _DEPS_LOCK:
        if _watchlist_repo_instance is not None:
            _watchlist_repo_instance.close()
        if _quote_service_instance is not None:
            _quote_service_instance.close()
    LLMFactory.close_all()


def override_research_agent(agent: ResearchAgent) -> None:
    """Explicitly override agent instance (useful for unit testing)."""
    global _agent_instance
    _agent_instance = agent


def override_chat_history(manager: ChatHistoryManager) -> None:
    """Explicitly override chat history instance (useful for unit testing)."""
    global _history_instance
    _history_instance = manager


def override_duckdb_repository(repo: DuckDBFinancialRepository) -> None:
    """Explicitly override DuckDB repository instance (useful for unit testing)."""
    global _duckdb_repo_instance, _sql_tool_instance
    _duckdb_repo_instance = repo
    _sql_tool_instance = None
