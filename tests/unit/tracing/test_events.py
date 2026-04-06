"""Unit tests for kairos.ai.tracing.events."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from kairos.ai.tracing.events import (
    EVENT_TYPES,
    ActionTaken,
    ConsoleMessage,
    Decision,
    LLMCallCompleted,
    LLMCallStarted,
    PromptRendered,
    RunCompleted,
    RunStarted,
    StepCompleted,
    StepStarted,
    TraceEvent,
    parse_event,
)


class TestTraceEvent:
    """Base event model tests."""

    def test_auto_generates_id_and_timestamp(self):
        event = RunStarted(run_id="r1", pipeline="physics", pipeline_run_id="r1")
        assert event.event_id
        assert len(event.event_id) == 16
        assert event.timestamp.tzinfo is not None

    def test_frozen_model(self):
        event = RunStarted(run_id="r1", pipeline="physics", pipeline_run_id="r1")
        try:
            event.run_id = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_event_type_discriminator(self):
        event = RunStarted(run_id="r1", pipeline="physics", pipeline_run_id="r1")
        assert event.event_type == "run_started"

    def test_serialisation_round_trip(self):
        event = StepCompleted(
            run_id="r1",
            step_name="idea_agent",
            step_number=1,
            status="success",
            duration_ms=1234,
        )
        data = json.loads(event.model_dump_json())
        assert data["event_type"] == "step_completed"
        assert data["duration_ms"] == 1234
        restored = parse_event(data)
        assert isinstance(restored, StepCompleted)
        assert restored.step_name == "idea_agent"


class TestEventTypes:
    """Test all event type constructors."""

    def test_run_started(self):
        e = RunStarted(run_id="r", pipeline="physics", pipeline_run_id="r")
        assert e.event_type == "run_started"

    def test_run_completed(self):
        e = RunCompleted(
            run_id="r",
            pipeline="physics",
            pipeline_run_id="r",
            status="success",
            total_duration_ms=5000,
            total_cost_usd=0.01,
            total_llm_calls=3,
        )
        assert e.total_cost_usd == 0.01

    def test_step_started(self):
        e = StepStarted(run_id="r", step_name="sim", step_number=2)
        assert e.event_type == "step_started"

    def test_step_completed_with_errors(self):
        e = StepCompleted(
            run_id="r",
            step_name="sim",
            step_number=2,
            status="error",
            errors=["timeout"],
        )
        assert e.errors == ["timeout"]

    def test_llm_call_started(self):
        e = LLMCallStarted(
            run_id="r",
            call_id="c1",
            step_name="idea",
            model_alias="concept-developer",
            model_resolved="deepseek-r1",
        )
        assert e.call_pattern == "direct"

    def test_llm_call_completed(self):
        e = LLMCallCompleted(
            run_id="r",
            call_id="c1",
            step_name="idea",
            model_alias="concept-developer",
            model_resolved="deepseek-r1",
            tokens_in=100,
            tokens_out=200,
            cost_usd=0.001,
            latency_ms=500,
        )
        assert e.tokens_in == 100
        assert e.tokens_out == 200

    def test_prompt_rendered(self):
        e = PromptRendered(
            run_id="r",
            call_id="c1",
            step_name="idea",
            template_name="concept_developer",
            template_version=1,
            template_hash="abc123",
            variables={"archetype": "test"},
        )
        assert e.variables["archetype"] == "test"

    def test_decision(self):
        e = Decision(
            run_id="r",
            step_name="review",
            saw="low quality score",
            decided="retry with different prompt",
            action="increment attempt",
        )
        assert e.event_type == "decision"

    def test_console_message(self):
        e = ConsoleMessage(
            run_id="r",
            level="warning",
            step_name="sim",
            message="Docker slow",
        )
        assert e.level == "warning"


class TestParseEvent:
    """Test the parse_event deserialization function."""

    def test_all_event_types_registered(self):
        assert len(EVENT_TYPES) == 10

    def test_action_taken(self):
        e = ActionTaken(
            run_id="r",
            step_name="sim",
            tool="blender:generate_course",
            input_summary="archetype=s_curve",
            output_summary="domino_course.blend",
            status="success",
            duration_ms=5000,
        )
        assert e.event_type == "action_taken"
        assert e.tool == "blender:generate_course"

    def test_parse_unknown_raises(self):
        try:
            parse_event({"event_type": "unknown_event"})
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "unknown_event" in str(e)

    def test_parse_preserves_timestamps(self):
        ts = datetime.now(timezone.utc).isoformat()
        data = {
            "event_type": "console",
            "event_id": "test123",
            "run_id": "r1",
            "timestamp": ts,
            "level": "info",
            "step_name": "",
            "message": "hello",
        }
        event = parse_event(data)
        assert isinstance(event, ConsoleMessage)
        assert event.message == "hello"
