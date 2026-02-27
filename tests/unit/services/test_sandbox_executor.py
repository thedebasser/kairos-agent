"""Unit tests for sandbox executor."""

import pytest

pytestmark = pytest.mark.unit


class TestSandboxExecutor:
    """Tests for simulation sandbox execution.

    Note: Full sandbox tests require Docker and are in integration tests.
    Unit tests cover input validation and result parsing.
    """

    def test_placeholder(self):
        """Placeholder — sandbox integration tests in tests/integration/."""
        # Sandbox execution requires Docker, tested in integration suite
        pass
