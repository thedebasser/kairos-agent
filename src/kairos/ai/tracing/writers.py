"""Kairos Agent -- JSONL File Writers.

Writes tracing events to the per-run file structure specified in D8:

    runs/<job_id>/
    +-- events.jsonl            # Machine-readable lifecycle events
    +-- console.jsonl           # Human-readable real-time feed
    +-- steps/
    |   +-- NN_step_name/
    |       +-- decisions.jsonl # Agent reasoning chains
    |       +-- prompts/
    |           +-- NNN_request.json
    |           +-- NNN_response.json

All writes are append-only and immediately flushed so a Ctrl+C never
loses buffered entries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kairos.ai.tracing.events import TraceEvent

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Append-only JSONL writer with immediate flush."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, data: dict[str, Any]) -> None:
        """Serialise *data* as a single JSON line and flush."""
        self._file.write(json.dumps(data, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class RunFileWriter:
    """Manages all JSONL and JSON files for a single pipeline run.

    Owns the run directory layout and provides typed write helpers.
    """

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._events_writer = JSONLWriter(self.run_dir / "events.jsonl")
        self._console_writer = JSONLWriter(self.run_dir / "console.jsonl")
        self._step_writers: dict[str, JSONLWriter] = {}
        self._prompt_counter = 0

    # -- Core event writers ------------------------------------------------

    def write_event(self, event: TraceEvent) -> None:
        """Append to ``events.jsonl``."""
        self._events_writer.write(event.model_dump(mode="json"))

    def write_console(self, event: TraceEvent) -> None:
        """Append to ``console.jsonl`` (human-readable feed)."""
        self._console_writer.write(event.model_dump(mode="json"))

    # -- Step-level writers ------------------------------------------------

    def _step_dir(self, step_name: str, step_number: int) -> Path:
        """Ensure the step directory exists and return it."""
        d = self.run_dir / "steps" / f"{step_number:02d}_{step_name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_decision(
        self,
        step_name: str,
        step_number: int,
        event: TraceEvent,
    ) -> None:
        """Append to ``steps/NN_step_name/decisions.jsonl``."""
        key = f"{step_number:02d}_{step_name}"
        if key not in self._step_writers:
            d = self._step_dir(step_name, step_number)
            self._step_writers[key] = JSONLWriter(d / "decisions.jsonl")
        self._step_writers[key].write(event.model_dump(mode="json"))

    # -- Prompt artifact writers -------------------------------------------

    def write_prompt_request(
        self,
        step_name: str,
        step_number: int,
        messages: list[dict[str, Any]],
        lineage: dict[str, Any] | None = None,
    ) -> str:
        """Write a prompt request JSON file.

        Returns the generated filename (e.g. ``001_request.json``).
        """
        self._prompt_counter += 1
        d = self._step_dir(step_name, step_number) / "prompts"
        d.mkdir(parents=True, exist_ok=True)

        filename = f"{self._prompt_counter:03d}_request.json"
        payload: dict[str, Any] = {"messages": messages}
        if lineage:
            payload["lineage"] = lineage
        (d / filename).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return filename

    def write_prompt_response(
        self,
        step_name: str,
        step_number: int,
        response: Any,
        call_id: str = "",
    ) -> str:
        """Write a prompt response JSON file.

        Returns the generated filename (e.g. ``001_response.json``).
        """
        d = self._step_dir(step_name, step_number) / "prompts"
        d.mkdir(parents=True, exist_ok=True)

        # Use the same counter as the last request
        filename = f"{self._prompt_counter:03d}_response.json"
        payload: dict[str, Any] = {
            "call_id": call_id,
            "response": response if isinstance(response, (dict, list, str)) else str(response),
        }
        (d / filename).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return filename

    # -- File artifact helpers ---------------------------------------------

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write an arbitrary file under the run directory."""
        p = self.run_dir / relative_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def get_output_dir(self, version: int = 1) -> Path:
        """Return (and create) ``assets/`` or versioned output dir."""
        d = self.run_dir / "assets" / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- Cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._events_writer.close()
        self._console_writer.close()
        for w in self._step_writers.values():
            w.close()
