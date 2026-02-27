"""Unit tests for DB operations.

Uses SQLite in-memory for fast testing.
"""

import pytest

pytestmark = pytest.mark.unit


class TestPipelineRunCRUD:
    """Tests for pipeline run CRUD operations.

    Integration tests with PostgreSQL are in tests/integration/.
    Unit tests here validate query construction with SQLite in-memory.
    """

    @pytest.mark.asyncio
    async def test_placeholder(self):
        """Placeholder — full DB tests require async session fixtures."""
        # TODO: Add in-memory SQLite async session fixture
        pass
