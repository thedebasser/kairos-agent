"""Unit tests for LLM thinking content accumulation (Finding 1.3)."""

from __future__ import annotations

from kairos.ai.llm.routing import (
    _accumulate_thinking,
    _thinking_buffer,
    collect_thinking,
)


class TestThinkingAccumulator:
    """Tests for the thinking buffer used to persist LLM reasoning."""

    def setup_method(self):
        """Clear buffer before each test."""
        _thinking_buffer.clear()

    def test_collect_returns_empty_initially(self):
        entries = collect_thinking()
        assert entries == []

    def test_accumulate_adds_entry(self):
        _accumulate_thinking("claude-sonnet", "Let me think about this...")
        entries = collect_thinking()
        assert len(entries) == 1
        assert entries[0]["model"] == "claude-sonnet"
        assert entries[0]["thinking"] == "Let me think about this..."

    def test_collect_clears_buffer(self):
        _accumulate_thinking("claude-sonnet", "Thinking A")
        _accumulate_thinking("claude-sonnet", "Thinking B")
        first = collect_thinking()
        assert len(first) == 2

        second = collect_thinking()
        assert len(second) == 0

    def test_none_thinking_is_ignored(self):
        _accumulate_thinking("claude-sonnet", None)
        entries = collect_thinking()
        assert len(entries) == 0

    def test_empty_string_thinking_is_ignored(self):
        _accumulate_thinking("claude-sonnet", "")
        entries = collect_thinking()
        assert len(entries) == 0

    def test_multiple_models_tracked(self):
        _accumulate_thinking("claude-sonnet", "Think 1")
        _accumulate_thinking("gpt-4o", "Think 2")
        entries = collect_thinking()
        assert len(entries) == 2
        models = {e["model"] for e in entries}
        assert models == {"claude-sonnet", "gpt-4o"}
