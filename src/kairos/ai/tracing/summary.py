"""Kairos Agent -- Run Summary Generator.

Writes ``run_summary.json`` from a ``RunCompleted`` event.  Always
generated, whether the run succeeded or failed, so every run tells
a complete story.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kairos.ai.tracing.events import RunCompleted

logger = logging.getLogger(__name__)


def write_run_summary(run_dir: Path, event: RunCompleted) -> Path:
    """Write ``run_summary.json`` into *run_dir*.

    Reads ``events.jsonl`` to aggregate step timings and LLM call costs.
    """
    steps: list[dict[str, Any]] = []
    total_llm_cost = 0.0
    total_llm_calls = 0

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("event_type") == "step_completed":
                steps.append({
                    "step": data.get("step_name", ""),
                    "step_number": data.get("step_number", 0),
                    "attempt": data.get("attempt", 1),
                    "status": data.get("status", ""),
                    "duration_ms": data.get("duration_ms", 0),
                })

            if data.get("event_type") == "llm_call_completed":
                total_llm_calls += 1
                total_llm_cost += data.get("cost_usd", 0.0)

    summary: dict[str, Any] = {
        "pipeline_run_id": event.pipeline_run_id,
        "pipeline": event.pipeline,
        "status": event.status,
        "total_duration_ms": event.total_duration_ms,
        "total_cost_usd": round(event.total_cost_usd, 6),
        "total_llm_calls": event.total_llm_calls,
        "concept_title": event.concept_title,
        "final_video_path": event.final_video_path,
        "errors": event.errors,
        "steps": steps,
        "aggregated": {
            "llm_calls_from_events": total_llm_calls,
            "llm_cost_from_events": round(total_llm_cost, 6),
            "steps_succeeded": sum(1 for s in steps if s["status"] == "success"),
            "steps_failed": sum(1 for s in steps if s["status"] != "success"),
        },
    }

    path = run_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Run summary written: %s", path)
    return path
