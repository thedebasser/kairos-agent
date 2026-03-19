"""Kairos Agent — Step Artifact System.

Saves structured JSON artifacts for every pipeline step to enable:
  - Manual review of each step's output
  - Prompt iteration and A/B testing
  - Debugging failed runs
  - Quality tracking over time

New folder layout (§1 of logging-observability spec):
  ``runs/<run_id>/``
  ├── run_summary.json              (enriched overall report)
  ├── steps/
  │   ├── 01_idea_agent.json
  │   ├── 02_simulation_agent.json
  │   ├── 02_simulation_code.py
  │   ├── 03_video_editor.json
  │   ├── 04_video_review.json
  │   └── 05_audio_review.json
  ├── llm_calls/
  │   ├── call_001_concept-developer_prompt.txt
  │   ├── call_001_concept-developer_response.txt
  │   └── ...
  └── output/
      └── v1/
          ├── final.mp4
          └── composition_metadata.json

Old layout (flat step files in run_dir) remains readable by external
tools until the user confirms migration is complete.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Root directory for run artifacts (project_root/runs/)
RUNS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runs"


class RunArtifacts:
    """Manages artifact capture for a single pipeline run.

    Usage:
        artifacts = RunArtifacts(pipeline_run_id)
        artifacts.save_step("idea_agent", step_number=1, data={...})
        artifacts.save_file("simulation_code.py", code_string)
        artifacts.save_summary({...})
    """

    def __init__(self, pipeline_run_id: str, pipeline_name: str = "physics") -> None:
        self.run_id = pipeline_run_id
        self.pipeline_name = pipeline_name
        self.start_time = time.monotonic()
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self._step_timings: list[dict[str, Any]] = []
        self._llm_call_counter = 0  # running counter for llm_calls/ naming

        # Create run directory tree
        self.run_dir = RUNS_DIR / pipeline_run_id
        self.steps_dir = self.run_dir / "steps"
        self.llm_calls_dir = self.run_dir / "llm_calls"
        self.output_dir = self.run_dir / "output"

        for d in (self.run_dir, self.steps_dir, self.llm_calls_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Run artifacts directory: %s", self.run_dir)

    # ── Step Artifacts ──────────────────────────────────────────────

    def save_step(
        self,
        step_name: str,
        *,
        step_number: int,
        status: str = "success",
        duration_ms: int = 0,
        attempt_number: int = 1,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        llm_calls: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a structured artifact for a pipeline step.

        Args:
            step_name: Human-readable step name (e.g. 'idea_agent').
            step_number: Ordering number (1, 2, 3...).
            status: 'success', 'failed', 'skipped'.
            duration_ms: Total wall-clock time for this step.
            attempt_number: 1-based retry counter (Finding 1.2).
            inputs: Key inputs to this step (serialisable).
            outputs: Key outputs from this step (serialisable).
            llm_calls: List of LLM call summaries (from collect_llm_calls).
            errors: List of error messages.
            metadata: Extra metadata.

        Returns:
            Path to the saved JSON file.
        """
        calls = llm_calls or []

        # Persist raw prompt/response files for each LLM call (§3)
        for call_record in calls:
            self._persist_llm_call_files(call_record)

        artifact = {
            "step_name": step_name,
            "step_number": step_number,
            "attempt_number": attempt_number,
            "pipeline_run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "duration_ms": duration_ms,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "llm_calls": calls,
            "errors": errors or [],
            "metadata": metadata or {},
        }

        filename = f"{step_number:02d}_{step_name}.json"
        filepath = self.steps_dir / filename
        filepath.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")

        self._step_timings.append({
            "step": step_name,
            "step_number": step_number,
            "attempt_number": attempt_number,
            "status": status,
            "duration_ms": duration_ms,
            "llm_call_count": len(calls),
            "llm_cost_usd": sum(c.get("cost_usd", 0.0) for c in calls),
        })

        logger.info(
            "[artifacts] Saved step %d/%s -> %s (%s, %dms, %d LLM calls)",
            step_number, step_name, filepath.name, status, duration_ms, len(calls),
        )
        return filepath

    # ── Raw LLM Call Files ──────────────────────────────────────────

    def _persist_llm_call_files(self, call_record: dict[str, Any]) -> None:
        """Write raw prompt/response/thinking text files for one LLM call.

        Files are saved as:
          llm_calls/call_{NNN}_{model_alias}_prompt.txt
          llm_calls/call_{NNN}_{model_alias}_response.txt
          llm_calls/call_{NNN}_{model_alias}_thinking.txt  (if present)

        The call_record dict is updated in-place with relative path refs.
        """
        self._llm_call_counter += 1
        counter = self._llm_call_counter
        alias = call_record.get("model_alias", "unknown").replace("/", "-")
        prefix = f"call_{counter:03d}_{alias}"

        # Prompt text (serialise messages list if present)
        prompt_data = call_record.pop("_raw_prompt", None)
        if prompt_data:
            prompt_path = self.llm_calls_dir / f"{prefix}_prompt.txt"
            prompt_text = (
                json.dumps(prompt_data, indent=2, default=str)
                if isinstance(prompt_data, (list, dict))
                else str(prompt_data)
            )
            prompt_path.write_text(prompt_text, encoding="utf-8")
            call_record["prompt_file"] = f"llm_calls/{prefix}_prompt.txt"

        # Response text
        response_data = call_record.pop("_raw_response", None)
        if response_data:
            response_path = self.llm_calls_dir / f"{prefix}_response.txt"
            response_text = (
                json.dumps(response_data, indent=2, default=str)
                if isinstance(response_data, (list, dict))
                else str(response_data)
            )
            response_path.write_text(response_text, encoding="utf-8")
            call_record["response_file"] = f"llm_calls/{prefix}_response.txt"

        # Thinking text (if model produced reasoning)
        # Prefer full _raw_thinking over truncated thinking_summary
        raw_thinking = call_record.pop("_raw_thinking", None)
        thinking = raw_thinking or call_record.get("thinking_summary")
        if thinking:
            thinking_path = self.llm_calls_dir / f"{prefix}_thinking.txt"
            thinking_path.write_text(str(thinking), encoding="utf-8")
            call_record["thinking_file"] = f"llm_calls/{prefix}_thinking.txt"

    # ── File Artifacts ──────────────────────────────────────────────

    def save_file(self, filename: str, content: str) -> Path:
        """Save a raw file artifact (e.g., simulation code, prompts).

        Files are saved into the ``steps/`` subdirectory alongside step JSON.

        Args:
            filename: Output filename (e.g., '02_simulation_code.py').
            content: File content as string.

        Returns:
            Path to the saved file.
        """
        filepath = self.steps_dir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.debug("[artifacts] Saved file: %s", filepath.name)
        return filepath

    # ── Output Versioning ───────────────────────────────────────────

    def get_output_version_dir(self, version: int = 1) -> Path:
        """Get or create the versioned output directory.

        Returns:
            Path to ``runs/{run_id}/output/v{version}/``.
        """
        version_dir = self.output_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    # ── Run Summary ─────────────────────────────────────────────────

    def save_summary(
        self,
        final_status: str,
        final_state: dict[str, Any] | None = None,
    ) -> Path:
        """Generate and save the enriched run summary.

        Called at the very end of a pipeline run (success or failure).
        Includes aggregated totals for LLM calls, cost, and timing (§2).

        Returns:
            Path to the run_summary.json file.
        """
        total_ms = int((time.monotonic() - self.start_time) * 1000)

        # Aggregate LLM totals from step timings
        total_llm_calls = sum(s.get("llm_call_count", 0) for s in self._step_timings)
        total_llm_cost = sum(s.get("llm_cost_usd", 0.0) for s in self._step_timings)

        summary = {
            "pipeline_run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "started_at": self.start_timestamp,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_duration_ms": total_ms,
            "final_status": final_status,
            "totals": {
                "llm_calls": total_llm_calls,
                "total_cost_usd": round(total_llm_cost, 6),
                "steps_succeeded": sum(1 for s in self._step_timings if s["status"] == "success"),
                "steps_failed": sum(1 for s in self._step_timings if s["status"] == "failed"),
            },
            "steps": self._step_timings,
            "concept_title": (
                final_state.get("concept", {}).get("title")
                if final_state and isinstance(final_state.get("concept"), dict)
                else None
            ),
            "final_video_path": final_state.get("final_video_path") if final_state else None,
            "total_cost_usd": round(total_llm_cost, 6),
            "errors": final_state.get("errors", []) if final_state else [],
        }

        filepath = self.run_dir / "run_summary.json"
        filepath.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        logger.info(
            "[artifacts] Run summary saved -> %s (status=%s, %dms total, %d LLM calls, $%.4f)",
            filepath, final_status, total_ms, total_llm_calls, total_llm_cost,
        )
        return filepath


# ---------------------------------------------------------------------------
# Module-level singleton per run (set by graph.py at pipeline start)
# ---------------------------------------------------------------------------

_current_artifacts: RunArtifacts | None = None


def init_run_artifacts(pipeline_run_id: str, pipeline_name: str = "physics") -> RunArtifacts:
    """Initialise the artifact system for a new pipeline run."""
    global _current_artifacts  # noqa: PLW0603
    _current_artifacts = RunArtifacts(pipeline_run_id, pipeline_name)
    return _current_artifacts


def get_run_artifacts() -> RunArtifacts | None:
    """Get the current run's artifact manager (or None if not initialised)."""
    return _current_artifacts
