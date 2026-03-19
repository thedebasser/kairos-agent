"""Kairos Agent -- Rich Live Run Display.

Real-time terminal UI that tails ``console.jsonl`` during a pipeline run
and renders completed steps as Rich panels.  Designed to be driven from
the CLI ``run`` command or used for post-run replay via ``inspect``.

Architecture:
  RunDisplay owns a Rich ``Live`` context.  It receives events either
  by polling a JSONL file (``tail_file``) or by direct push from the
  RunTracer via ``on_event()``.  Each event updates the internal
  display state and triggers a Live refresh.

Usage (live mode)::

    async with RunDisplay() as display:
        display.on_event(event_dict)   # called by tracer callback

Usage (replay)::

    display = RunDisplay()
    display.replay(run_dir)            # reads console.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from kairos.cli.ui.formatters import (
    format_console_event,
    format_llm_call_completed,
    format_run_completed,
    format_step_completed,
)
from kairos.cli.ui.step_panel import build_step_panel

# Max lines in the scrolling log region
_MAX_LOG_LINES = 40


class RunDisplay:
    """Rich live terminal display for pipeline runs.

    Can be used as an async context manager for live display, or
    synchronously for replay mode.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._live: Live | None = None

        # Display state
        self._run_id: str = ""
        self._pipeline: str = ""
        self._log_lines: list[Text] = []
        self._completed_panels: list[Panel] = []

        # Per-step collectors (populated from events.jsonl during replay)
        self._step_llm_calls: dict[str, list[dict[str, Any]]] = {}
        self._step_decisions: dict[str, list[dict[str, Any]]] = {}
        self._run_summary: dict[str, Any] | None = None

    # -- Context manager (live mode) ---------------------------------------

    async def __aenter__(self) -> RunDisplay:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self._live.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    # -- Event ingestion ---------------------------------------------------

    def on_event(self, event: dict[str, Any]) -> None:
        """Process a single trace event (called from tracer or file)."""
        etype = event.get("event_type", "")

        if etype == "run_started":
            self._run_id = event.get("pipeline_run_id", event.get("run_id", ""))
            self._pipeline = event.get("pipeline", "")

        elif etype == "console":
            line = format_console_event(event)
            self._log_lines.append(line)
            # Keep bounded
            if len(self._log_lines) > _MAX_LOG_LINES:
                self._log_lines = self._log_lines[-_MAX_LOG_LINES:]

        elif etype == "step_completed":
            line = format_step_completed(event)
            self._log_lines.append(line)
            # Build panel from collected data
            step_name = event.get("step_name", "")
            panel = build_step_panel(
                event,
                llm_calls=self._step_llm_calls.get(step_name),
                decisions=self._step_decisions.get(step_name),
            )
            self._completed_panels.append(panel)

        elif etype == "llm_call_completed":
            line = format_llm_call_completed(event)
            self._log_lines.append(line)
            step = event.get("step_name", "")
            self._step_llm_calls.setdefault(step, []).append(event)

        elif etype == "decision":
            step = event.get("step_name", "")
            self._step_decisions.setdefault(step, []).append(event)

        elif etype == "run_completed":
            self._run_summary = event
            line = format_run_completed(event)
            self._log_lines.append(Text())  # blank line
            self._log_lines.append(Rule("Run Complete"))
            self._log_lines.append(line)

        # Refresh live display if active
        if self._live:
            self._live.update(self._render())

    # -- Replay mode -------------------------------------------------------

    def replay(
        self,
        run_dir: Path,
        *,
        speed: float = 1.0,
        events_file: str = "events.jsonl",
    ) -> None:
        """Replay a completed run from its events file.

        Reads ``events.jsonl`` (which has all event types) and feeds
        them through ``on_event`` with optional time scaling.

        Parameters
        ----------
        run_dir:
            Path to the run directory containing ``events.jsonl``.
        speed:
            Replay speed multiplier (2.0 = 2x faster, 0 = instant).
        events_file:
            Filename to read (default ``events.jsonl``).
        """
        path = run_dir / events_file
        if not path.exists():
            self._console.print(f"[red]File not found:[/red] {path}")
            return

        events: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not events:
            self._console.print("[yellow]No events found in file.[/yellow]")
            return

        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=8,
            vertical_overflow="visible",
        ) as live:
            self._live = live
            prev_ts: str | None = None

            for event in events:
                # Simulate timing if speed > 0
                if speed > 0 and prev_ts is not None:
                    cur_ts = event.get("timestamp", "")
                    delay = _timestamp_delta(prev_ts, cur_ts) / speed
                    if 0 < delay < 5.0:  # cap at 5s real time
                        time.sleep(delay)
                prev_ts = event.get("timestamp", "")

                self.on_event(event)

            self._live = None

        # Print final summary outside live context
        if self._run_summary:
            self._console.print()
            self._console.print(format_run_completed(self._run_summary))

    # -- Rendering ---------------------------------------------------------

    def _render(self) -> Group:
        """Build the full display renderable."""
        parts: list[Any] = []

        # Header
        header = Text()
        header.append("Kairos Pipeline", style="bold bright_white")
        if self._pipeline:
            header.append(f" | {self._pipeline}", style="dim")
        if self._run_id:
            header.append(f" | {self._run_id[:12]}", style="dim")
        parts.append(Panel(header, border_style="blue"))

        # Completed step panels
        for panel in self._completed_panels:
            parts.append(panel)

        # Scrolling log
        if self._log_lines:
            log_group = Group(*self._log_lines[-_MAX_LOG_LINES:])
            parts.append(
                Panel(
                    log_group,
                    title=Text("Live Log", style="bold"),
                    border_style="dim",
                    expand=True,
                )
            )

        return Group(*parts) if parts else Group(Text("Waiting for events...", style="dim"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp_delta(ts1: str, ts2: str) -> float:
    """Return seconds between two ISO timestamps (best effort)."""
    from datetime import datetime

    try:
        t1 = datetime.fromisoformat(ts1)
        t2 = datetime.fromisoformat(ts2)
        return max((t2 - t1).total_seconds(), 0.0)
    except (ValueError, TypeError):
        return 0.0
