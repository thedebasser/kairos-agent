"""Unit tests for kairos.ai.tracing.writers."""

from __future__ import annotations

import json
from pathlib import Path

from kairos.ai.tracing.events import ConsoleMessage, Decision, StepCompleted
from kairos.ai.tracing.writers import JSONLWriter, RunFileWriter


class TestJSONLWriter:
    """Tests for the low-level JSONL writer."""

    def test_write_creates_file(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        writer = JSONLWriter(path)
        writer.write({"key": "value"})
        writer.close()

        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["key"] == "value"

    def test_write_multiple_lines(self, tmp_path: Path):
        path = tmp_path / "multi.jsonl"
        writer = JSONLWriter(path)
        for i in range(5):
            writer.write({"i": i})
        writer.close()

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "dir" / "test.jsonl"
        writer = JSONLWriter(path)
        writer.write({"ok": True})
        writer.close()
        assert path.exists()


class TestRunFileWriter:
    """Tests for the per-run file structure manager."""

    def test_creates_run_directory(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-123")
        assert (tmp_path / "run-123").is_dir()
        writer.close()

    def test_write_event_to_events_jsonl(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-456")
        event = StepCompleted(
            run_id="run-456",
            step_name="test",
            step_number=1,
            status="success",
            duration_ms=100,
        )
        writer.write_event(event)
        writer.close()

        events_file = tmp_path / "run-456" / "events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert data["event_type"] == "step_completed"

    def test_write_console_to_console_jsonl(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-789")
        event = ConsoleMessage(
            run_id="run-789",
            level="info",
            step_name="idea",
            message="Hello world",
        )
        writer.write_console(event)
        writer.close()

        console_file = tmp_path / "run-789" / "console.jsonl"
        assert console_file.exists()
        data = json.loads(console_file.read_text(encoding="utf-8").strip())
        assert data["message"] == "Hello world"

    def test_write_decision(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-dec")
        event = Decision(
            run_id="run-dec",
            step_name="review",
            saw="score=0.3",
            decided="retry",
            action="loop",
        )
        writer.write_decision("review", 4, event)
        writer.close()

        decisions_file = tmp_path / "run-dec" / "steps" / "04_review" / "decisions.jsonl"
        assert decisions_file.exists()
        data = json.loads(decisions_file.read_text(encoding="utf-8").strip())
        assert data["saw"] == "score=0.3"

    def test_write_prompt_request_and_response(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-prompt")
        messages = [{"role": "system", "content": "You are helpful"}]
        lineage = {"template_name": "test", "template_version": 1}

        req_name = writer.write_prompt_request("idea", 1, messages, lineage)
        assert req_name == "001_request.json"

        resp_name = writer.write_prompt_response("idea", 1, {"text": "output"}, call_id="c1")
        assert resp_name == "001_response.json"

        prompts_dir = tmp_path / "run-prompt" / "steps" / "01_idea" / "prompts"
        req_data = json.loads((prompts_dir / "001_request.json").read_text(encoding="utf-8"))
        assert req_data["messages"] == messages
        assert req_data["lineage"]["template_name"] == "test"

        resp_data = json.loads((prompts_dir / "001_response.json").read_text(encoding="utf-8"))
        assert resp_data["response"]["text"] == "output"

    def test_write_file_arbitrary(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-file")
        path = writer.write_file("steps/02_sim/simulation_code.py", "print('hello')")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "print('hello')"
        writer.close()

    def test_get_output_dir(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-out")
        d = writer.get_output_dir(version=2)
        assert d.is_dir()
        assert d.name == "v2"
        writer.close()

    def test_prompt_counter_increments(self, tmp_path: Path):
        writer = RunFileWriter(tmp_path, "run-cnt")
        writer.write_prompt_request("idea", 1, [{"role": "user", "content": "a"}])
        writer.write_prompt_request("idea", 1, [{"role": "user", "content": "b"}])
        writer.close()

        prompts_dir = tmp_path / "run-cnt" / "steps" / "01_idea" / "prompts"
        assert (prompts_dir / "001_request.json").exists()
        assert (prompts_dir / "002_request.json").exists()
