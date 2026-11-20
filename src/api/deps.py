"""FastAPI dependency injection providers.

Provides singleton instances of ResearchAgent, ChatHistoryManager, DuckDBFinancialRepository,
and S3Storage with lazy initialization and clean test-override support.
"""

from functools import lru_cache
from typing import Optional

from src.ai.agent.research_agent import ResearchAgent
from src.ai.agent.tools.duckdb_repository import DuckDBFinancialRepository
from src.api.chat_history import ChatHistoryManager
from src.data.storage.s3_storage import S3Storage


_agent_instance: Optional[ResearchAgent] = None
_history_instance: Optional[ChatHistoryManager] = None
_duckdb_repo_instance: Optional[DuckDBFinancialRepository] = None
_s3_instance: Optional[S3Storage] = None


def get_research_agent() -> ResearchAgent:
    """Retrieve or lazily initialize the singleton ResearchAgent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ResearchAgent()
    return _agent_instance


def get_chat_history() -> ChatHistoryManager:
    """Retrieve or lazily initialize the singleton ChatHistoryManager."""
    global _history_instance
    if _history_instance is None:
        _history_instance = ChatHistoryManager()
    return _history_instance


def get_duckdb_repository() -> DuckDBFinancialRepository:
    """Retrieve or lazily initialize the singleton DuckDBFinancialRepository."""
    global _duckdb_repo_instance
    if _duckdb_repo_instance is None:
        _duckdb_repo_instance = DuckDBFinancialRepository()
    return _duckdb_repo_instance


def get_s3_storage() -> Optional[S3Storage]:
    """Retrieve or lazily initialize the S3Storage adapter."""
    global _s3_instance
    if _s3_instance is None:
        try:
            _s3_instance = S3Storage()
        except Exception:
            _s3_instance = None
    return _s3_instance


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
