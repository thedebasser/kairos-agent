"""Unit tests for kairos.ai.tracing.tracer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kairos.ai.tracing.events import TraceEvent
from kairos.ai.tracing.tracer import RunTracer


class _CollectorSink:
    """Simple sink that collects events for assertions."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def on_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestRunTracer:
    """Tests for the RunTracer lifecycle."""

    def test_init_run_creates_directory(self, tmp_path: Path):
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.init_run("test-run", "physics")
        assert (tmp_path / "test-run").is_dir()
        tracer.complete_run("success")

    def test_init_run_emits_run_started(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("r1", "physics")

        started_events = [e for e in sink.events if e.event_type == "run_started"]
        assert len(started_events) == 1
        assert started_events[0].run_id == "r1"
        tracer.complete_run("success")

    def test_complete_run_emits_run_completed(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("r2", "physics")
        tracer.complete_run("success")

        completed = [e for e in sink.events if e.event_type == "run_completed"]
        assert len(completed) == 1
        assert completed[0].status == "success"

    def test_complete_run_writes_summary(self, tmp_path: Path):
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.init_run("r3", "physics")
        tracer.complete_run("success")

        summary = tmp_path / "r3" / "run_summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data["status"] == "success"

    def test_console_message(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("r4", "physics")
        tracer.console("Hello world", step_name="test")

        console_events = [e for e in sink.events if e.event_type == "console"]
        assert len(console_events) == 1
        assert console_events[0].message == "Hello world"
        tracer.complete_run("success")

    def test_run_properties(self, tmp_path: Path):
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.init_run("r5", "physics")
        assert tracer.run_id == "r5"
        assert tracer.run_dir == tmp_path / "r5"
        assert tracer.total_cost_usd == 0.0
        assert tracer.total_llm_calls == 0
        tracer.complete_run("success")

    def test_get_output_dir(self, tmp_path: Path):
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.init_run("r6", "physics")
        output_dir = tracer.get_output_dir(version=1)
        assert output_dir.is_dir()
        assert "v1" in str(output_dir)
        tracer.complete_run("success")


class TestStepSpan:
    """Tests for the StepSpan context manager."""

    def test_step_emits_started_and_completed(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("step-run", "physics")

        with tracer.step("idea_agent", step_number=1) as span:
            span.log("Generating concept")

        types = [e.event_type for e in sink.events]
        assert "step_started" in types
        assert "step_completed" in types
        assert "console" in types
        tracer.complete_run("success")

    def test_step_captures_failure(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("fail-run", "physics")

        try:
            with tracer.step("bad_step", step_number=1) as span:
                raise ValueError("boom")
        except ValueError:
            pass

        completed = [e for e in sink.events if e.event_type == "step_completed"]
        assert len(completed) == 1
        assert completed[0].status == "error"
        assert "boom" in completed[0].errors
        tracer.complete_run("error")

    def test_step_decision_recorded(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("dec-run", "physics")

        with tracer.step("review", step_number=4) as span:
            span.decide(saw="score=0.3", decided="retry", action="loop")

        decisions = [e for e in sink.events if e.event_type == "decision"]
        assert len(decisions) == 1
        assert decisions[0].saw == "score=0.3"

        # Also check file was written
        dec_file = tmp_path / "dec-run" / "steps" / "04_review" / "decisions.jsonl"
        assert dec_file.exists()
        tracer.complete_run("success")


class TestLLMCallSpan:
    """Tests for the LLMCallSpan context manager."""

    def test_llm_call_emits_started_and_completed(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("llm-run", "physics")

        with tracer.step("idea", step_number=1) as step:
            with step.llm_call("concept-dev", "deepseek-r1") as llm:
                llm.complete(tokens_in=100, tokens_out=200, cost_usd=0.001)

        types = [e.event_type for e in sink.events]
        assert "llm_call_started" in types
        assert "llm_call_completed" in types

        completed = [e for e in sink.events if e.event_type == "llm_call_completed"]
        assert completed[0].tokens_in == 100
        assert completed[0].cost_usd == 0.001
        assert tracer.total_llm_calls == 1
        assert tracer.total_cost_usd == 0.001
        tracer.complete_run("success")

    def test_llm_call_captures_error(self, tmp_path: Path):
        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("llm-err", "physics")

        try:
            with tracer.step("idea", step_number=1) as step:
                with step.llm_call("concept-dev", "deepseek-r1") as llm:
                    raise RuntimeError("API timeout")
        except RuntimeError:
            pass

        completed = [e for e in sink.events if e.event_type == "llm_call_completed"]
        assert len(completed) == 1
        assert completed[0].status == "error"
        assert "API timeout" in (completed[0].error or "")
        tracer.complete_run("error")

    def test_llm_call_latency_tracked(self, tmp_path: Path):
        import time

        sink = _CollectorSink()
        tracer = RunTracer(runs_dir=tmp_path)
        tracer.add_sink(sink)
        tracer.init_run("lat-run", "physics")

        with tracer.step("idea", step_number=1) as step:
            with step.llm_call("test", "model") as llm:
                time.sleep(0.05)
                llm.complete()

        completed = [e for e in sink.events if e.event_type == "llm_call_completed"]
        assert completed[0].latency_ms >= 40  # allow for timing jitter
        tracer.complete_run("success")
