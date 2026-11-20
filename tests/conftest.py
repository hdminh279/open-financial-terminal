"""Root test configuration and fixture environment setup.

Ensures offline test execution defaults to local Parquet fixtures
without requiring a live MinIO container, while live production/dev
runs connect strictly to the MinIO Lakehouse.
"""

import os
import pytest

# Force test fixtures for pytest runs
os.environ["USE_TEST_FIXTURES"] = "true"
