"""Unit tests for LLM context window budget estimation (Finding 3.4)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from kairos.services.llm_routing import _check_context_budget

# Patch targets — the function uses local imports from litellm
_TC = "litellm.token_counter"
_MT = "litellm.get_max_tokens"


class TestContextBudget:
    """Tests for pre-flight context window budget estimation."""

    def test_returns_budget_dict(self):
        """Returns dict with expected keys."""
        msgs = [{"role": "user", "content": "Hello"}]
        with patch(_TC, return_value=10), \
             patch(_MT, return_value=200_000):
            budget = _check_context_budget("anthropic/claude-sonnet-4-20250514", msgs, 16384)

        assert budget["input_tokens"] == 10
        assert budget["context_window"] == 200_000
        assert budget["available_for_output"] == 199_990
        assert budget["utilisation_pct"] == 0  # 10/200000 < 1%

    def test_warns_on_overflow(self, caplog):
        """Logs WARNING when input + max_output exceeds context window."""
        msgs = [{"role": "user", "content": "x" * 100}]
        with patch(_TC, return_value=190_000), \
             patch(_MT, return_value=200_000), \
             caplog.at_level(logging.WARNING):
            budget = _check_context_budget("anthropic/claude-sonnet-4-20250514", msgs, 16384)

        assert budget["utilisation_pct"] == 95
        assert any("overflow" in r.message.lower() for r in caplog.records)

    def test_info_on_high_utilisation(self, caplog):
        """Logs INFO when utilisation > 80% but not overflowing."""
        msgs = [{"role": "user", "content": "x" * 100}]
        with patch(_TC, return_value=170_000), \
             patch(_MT, return_value=200_000), \
             caplog.at_level(logging.INFO):
            budget = _check_context_budget("anthropic/claude-sonnet-4-20250514", msgs, 16384)

        assert budget["utilisation_pct"] == 85
        assert any("high" in r.message.lower() for r in caplog.records)

    def test_handles_token_counter_failure_gracefully(self):
        """Returns zeros if token_counter raises."""
        msgs = [{"role": "user", "content": "test"}]
        with patch(_TC, side_effect=Exception("no tokenizer")):
            budget = _check_context_budget("unknown-model", msgs, 16384)

        assert budget["input_tokens"] == 0
        assert budget["context_window"] == 0

    def test_handles_unknown_model_max_tokens(self):
        """Falls back to 200K context window if get_max_tokens returns None."""
        msgs = [{"role": "user", "content": "test"}]
        with patch(_TC, return_value=1000), \
             patch(_MT, return_value=None):
            budget = _check_context_budget("custom/model", msgs, 16384)

        assert budget["context_window"] == 200_000
        assert budget["available_for_output"] == 199_000

    def test_normal_utilisation_no_warning(self, caplog):
        """No warning when utilisation is low."""
        msgs = [{"role": "user", "content": "test"}]
        with patch(_TC, return_value=1000), \
             patch(_MT, return_value=200_000), \
             caplog.at_level(logging.WARNING):
            budget = _check_context_budget("anthropic/claude-sonnet-4-20250514", msgs, 16384)

        assert budget["utilisation_pct"] == 0
        assert not any("overflow" in r.message.lower() for r in caplog.records)
