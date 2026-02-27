"""Integration tests for LLM calls.

Tests actual API connectivity and response parsing.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.slow]


class TestLLMIntegration:
    """Tests for LLM connectivity and structured output."""

    def test_placeholder(self):
        """Placeholder — Step 2 implementation."""
        pass
