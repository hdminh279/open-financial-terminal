"""Root test configuration and fixture environment setup.

Ensures offline test execution defaults to local Parquet fixtures
without requiring a live MinIO container, and isolates pooled singletons
between test cases.
"""

import os

import pytest

# Force test fixtures for pytest runs
os.environ["USE_TEST_FIXTURES"] = "true"


@pytest.fixture(autouse=True)
def _reset_llm_factory_cache():
    """Ensure LLMFactory pooled adapter cache never leaks mocked adapters across tests."""
    from backend.ai.llm.factory import LLMFactory

    LLMFactory.clear_cache()
    yield
    LLMFactory.clear_cache()
