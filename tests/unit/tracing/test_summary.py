"""Unit tests for kairos.ai.tracing.summary."""

from __future__ import annotations

import json
from pathlib import Path

from kairos.ai.tracing.events import (
    LLMCallCompleted,
    RunCompleted,
    RunStarted,
    StepCompleted,
)
from kairos.ai.tracing.summary import write_run_summary
from kairos.ai.tracing.writers import JSONLWriter


class TestWriteRunSummary:
    """Tests for run_summary.json generation."""

    def _write_events(self, run_dir: Path, events: list) -> None:
        """Helper to write events.jsonl for testing."""
        writer = JSONLWriter(run_dir / "events.jsonl")
        for event in events:
            writer.write(event.model_dump(mode="json"))
        writer.close()

    def test_generates_summary_file(self, tmp_path: Path):
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        events = [
            RunStarted(run_id="r1", pipeline="physics", pipeline_run_id="r1"),
            StepCompleted(
                run_id="r1", step_name="idea_agent", step_number=1,
                status="success", duration_ms=1000,
            ),
            LLMCallCompleted(
                run_id="r1", call_id="c1", step_name="idea_agent",
                model_alias="test", model_resolved="test-model",
                tokens_in=100, tokens_out=200, cost_usd=0.01, latency_ms=500,
            ),
            StepCompleted(
                run_id="r1", step_name="simulation_agent", step_number=2,
                status="success", duration_ms=2000,
            ),
        ]
        self._write_events(run_dir, events)

        completed = RunCompleted(
            run_id="r1", pipeline="physics", pipeline_run_id="r1",
            status="success", total_duration_ms=3000,
            total_cost_usd=0.01, total_llm_calls=1,
        )
        write_run_summary(run_dir, completed)

        summary_path = run_dir / "run_summary.json"
        assert summary_path.exists()

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data["pipeline_run_id"] == "r1"
        assert data["status"] == "success"
        assert data["total_duration_ms"] == 3000
        assert data["total_cost_usd"] == 0.01
        assert len(data["steps"]) == 2
        assert data["steps"][0]["step"] == "idea_agent"
        assert data["steps"][1]["step"] == "simulation_agent"

    def test_handles_empty_events_file(self, tmp_path: Path):
        run_dir = tmp_path / "empty-run"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        completed = RunCompleted(
            run_id="r2", pipeline="physics", pipeline_run_id="r2",
            status="success", total_duration_ms=100,
        )
        write_run_summary(run_dir, completed)

        data = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert data["steps"] == []

    def test_handles_missing_events_file(self, tmp_path: Path):
        run_dir = tmp_path / "no-events"
        run_dir.mkdir()

        completed = RunCompleted(
            run_id="r3", pipeline="physics", pipeline_run_id="r3",
            status="error", errors=["crashed"],
        )
        write_run_summary(run_dir, completed)

        data = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert data["status"] == "error"
        assert data["errors"] == ["crashed"]
