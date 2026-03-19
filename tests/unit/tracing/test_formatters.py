"""Unit tests for kairos.cli.ui.formatters."""

from __future__ import annotations

from datetime import datetime, timezone

from kairos.cli.ui.formatters import (
    format_console_event,
    format_cost,
    format_duration,
    format_level,
    format_llm_call_completed,
    format_run_completed,
    format_step_completed,
    format_step_name,
    format_timestamp,
    format_tokens,
)


class TestFormatHelpers:
    """Tests for individual formatting functions."""

    def test_format_timestamp_from_datetime(self):
        ts = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        text = format_timestamp(ts)
        assert "14:30:45" in text.plain

    def test_format_timestamp_from_string(self):
        text = format_timestamp("2024-06-15T14:30:45+00:00")
        assert "14:30:45" in text.plain

    def test_format_level_info(self):
        text = format_level("info")
        assert "INFO" in text.plain

    def test_format_level_error(self):
        text = format_level("error")
        assert "ERROR" in text.plain

    def test_format_step_name(self):
        text = format_step_name("idea_agent")
        assert text.plain == "idea_agent"

    def test_format_step_name_empty(self):
        text = format_step_name("")
        assert text.plain == ""

    def test_format_duration_ms(self):
        assert format_duration(456) == "456ms"

    def test_format_duration_seconds(self):
        assert format_duration(1234) == "1.23s"

    def test_format_cost(self):
        text = format_cost(0.001234)
        assert "$0.001234" in text.plain

    def test_format_tokens(self):
        text = format_tokens(100, 200)
        assert "100 -> 200" in text.plain


class TestFormatEvents:
    """Tests for event formatting functions."""

    def test_format_console_event(self):
        event = {
            "timestamp": "2024-06-15T14:30:45+00:00",
            "level": "info",
            "step_name": "idea_agent",
            "message": "Generating concept",
        }
        text = format_console_event(event)
        assert "idea_agent" in text.plain
        assert "Generating concept" in text.plain

    def test_format_console_event_no_step(self):
        event = {
            "timestamp": "2024-06-15T14:30:45+00:00",
            "level": "warning",
            "step_name": "",
            "message": "General warning",
        }
        text = format_console_event(event)
        assert "General warning" in text.plain

    def test_format_step_completed(self):
        event = {
            "timestamp": "2024-06-15T14:30:45+00:00",
            "step_name": "simulation_agent",
            "status": "success",
            "duration_ms": 2500,
        }
        text = format_step_completed(event)
        assert "simulation_agent" in text.plain
        assert "SUCCESS" in text.plain

    def test_format_run_completed(self):
        event = {
            "status": "success",
            "total_duration_ms": 15000,
            "total_cost_usd": 0.05,
            "total_llm_calls": 5,
        }
        text = format_run_completed(event)
        assert "SUCCESS" in text.plain
        assert "5 LLM calls" in text.plain

    def test_format_llm_call_completed(self):
        event = {
            "timestamp": "2024-06-15T14:30:45+00:00",
            "model_resolved": "deepseek-r1",
            "tokens_in": 100,
            "tokens_out": 200,
            "latency_ms": 500,
            "cost_usd": 0.001,
            "status": "success",
        }
        text = format_llm_call_completed(event)
        assert "deepseek-r1" in text.plain
        assert "100 -> 200" in text.plain
