"""FastAPI dependency injection providers.

Provides singleton instances of ResearchAgent, ChatHistoryManager, DuckDBFinancialRepository,
and S3Storage with thread-safe lazy initialization and clean test-override support.
"""

import threading

from src.ai.agent.research_agent import ResearchAgent
from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.ai.vectordb.qdrant_client import QdrantStorage
from src.api.chat_history import ChatHistoryManager
from src.data.storage.s3_storage import S3Storage

_agent_instance: ResearchAgent | None = None
_history_instance: ChatHistoryManager | None = None
_duckdb_repo_instance: DuckDBFinancialRepository | None = None
_s3_instance: S3Storage | None = None
_qdrant_instance: QdrantStorage | None = None
_DEPS_LOCK = threading.Lock()


def get_research_agent() -> ResearchAgent:
    """Retrieve or lazily initialize the singleton ResearchAgent."""
    global _agent_instance
    if _agent_instance is None:
        with _DEPS_LOCK:
            if _agent_instance is None:
                _agent_instance = ResearchAgent()
    return _agent_instance


def get_chat_history() -> ChatHistoryManager:
    """Retrieve or lazily initialize the singleton ChatHistoryManager."""
    global _history_instance
    if _history_instance is None:
        with _DEPS_LOCK:
            if _history_instance is None:
                _history_instance = ChatHistoryManager()
    return _history_instance


def get_duckdb_repository() -> DuckDBFinancialRepository:
    """Retrieve or lazily initialize the singleton DuckDBFinancialRepository."""
    global _duckdb_repo_instance
    if _duckdb_repo_instance is None:
        with _DEPS_LOCK:
            if _duckdb_repo_instance is None:
                _duckdb_repo_instance = DuckDBFinancialRepository()
    return _duckdb_repo_instance


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
    global _duckdb_repo_instance
    _duckdb_repo_instance = repo
