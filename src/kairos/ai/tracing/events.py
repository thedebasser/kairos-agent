"""Kairos Agent -- Tracing Event Models.

Pydantic models for every event type emitted during a pipeline run.
Events are the atomic unit of observability: they flow from the
``RunTracer`` through ``TracingSink`` implementations (JSONL files,
Langfuse, database) and power the Rich terminal UI and run replay.

Design decisions:
  - No emojis anywhere -- plain ASCII for machine readability.
  - All timestamps are timezone-aware UTC.
  - Every event carries ``run_id`` and ``event_id`` for join/trace.
  - ``event_type`` is a plain string discriminator (not an enum)
    so sinks can filter cheaply without importing models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """Base for every tracing event."""

    event_id: str = Field(default_factory=_new_id)
    run_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    event_type: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

class RunStarted(TraceEvent):
    """Emitted once when a pipeline run begins."""

    event_type: Literal["run_started"] = "run_started"
    pipeline: str
    pipeline_run_id: str


class RunCompleted(TraceEvent):
    """Emitted once when a pipeline run finishes (success or failure)."""

    event_type: Literal["run_completed"] = "run_completed"
    pipeline: str
    pipeline_run_id: str
    status: str
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    final_video_path: str | None = None
    concept_title: str | None = None


# ---------------------------------------------------------------------------
# Step lifecycle
# ---------------------------------------------------------------------------

class StepStarted(TraceEvent):
    """Emitted when a graph node begins execution."""

    event_type: Literal["step_started"] = "step_started"
    step_name: str
    step_number: int
    attempt: int = 1


class StepCompleted(TraceEvent):
    """Emitted when a graph node finishes."""

    event_type: Literal["step_completed"] = "step_completed"
    step_name: str
    step_number: int
    attempt: int = 1
    status: str = "success"
    duration_ms: int = 0
    outputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM call lifecycle
# ---------------------------------------------------------------------------

class LLMCallStarted(TraceEvent):
    """Emitted before an LLM call is dispatched."""

    event_type: Literal["llm_call_started"] = "llm_call_started"
    call_id: str
    step_name: str
    model_alias: str
    model_resolved: str
    call_pattern: str = "direct"  # direct | quality_fallback


class LLMCallCompleted(TraceEvent):
    """Emitted after an LLM call returns (success or error)."""

    event_type: Literal["llm_call_completed"] = "llm_call_completed"
    call_id: str
    step_name: str
    model_alias: str
    model_resolved: str
    model_type: str = "local"  # local | cloud
    provider: str = ""
    call_pattern: str = "direct"
    routing_outcome: str = "direct"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "success"
    error: str | None = None
    has_thinking: bool = False


# ---------------------------------------------------------------------------
# Prompt lineage
# ---------------------------------------------------------------------------

class PromptRendered(TraceEvent):
    """Emitted when a prompt template is rendered for an LLM call."""

    event_type: Literal["prompt_rendered"] = "prompt_rendered"
    call_id: str
    step_name: str
    template_name: str
    template_version: int = 0
    template_hash: str = ""
    pipeline: str = ""
    variables: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decision / reasoning
# ---------------------------------------------------------------------------

class Decision(TraceEvent):
    """An agent reasoning record: saw -> decided -> action.

    Captured in ``steps/NN_step_name/decisions.jsonl``.
    """

    event_type: Literal["decision"] = "decision"
    step_name: str
    saw: str = ""       # What the agent observed
    decided: str = ""   # What it chose to do
    action: str = ""    # The concrete action taken
    reasoning: str = "" # Thinking/chain-of-thought (truncated)


# ---------------------------------------------------------------------------
# Console events (human-readable, for Rich UI + console.jsonl)
# ---------------------------------------------------------------------------

class ConsoleMessage(TraceEvent):
    """A human-readable message for the live console / replay."""

    event_type: Literal["console"] = "console"
    level: Literal["info", "warning", "error", "debug", "success"] = "info"
    step_name: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Type union for deserialization
# ---------------------------------------------------------------------------

EVENT_TYPES: dict[str, type[TraceEvent]] = {
    "run_started": RunStarted,
    "run_completed": RunCompleted,
    "step_started": StepStarted,
    "step_completed": StepCompleted,
    "llm_call_started": LLMCallStarted,
    "llm_call_completed": LLMCallCompleted,
    "prompt_rendered": PromptRendered,
    "decision": Decision,
    "console": ConsoleMessage,
}


def parse_event(data: dict[str, Any]) -> TraceEvent:
    """Deserialize a dict into the appropriate event subclass."""
    event_type = data.get("event_type", "")
    cls = EVENT_TYPES.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type!r}")
    return cls.model_validate(data)
