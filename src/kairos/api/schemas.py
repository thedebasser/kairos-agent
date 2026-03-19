"""Kairos Agent -- API Response Schemas.

Pydantic models for all REST API responses.  These are *not* DB models
or pipeline contracts — they exist purely for the JSON API layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run list / detail
# ---------------------------------------------------------------------------

class RunSummaryResponse(BaseModel):
    """Compact run representation for list endpoints."""

    pipeline_run_id: str
    pipeline: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_cost_usd: float | None = None
    total_duration_ms: int | None = None
    concept_title: str | None = None


class StepSummary(BaseModel):
    """One step inside a run detail response."""

    step: str
    step_number: int
    attempt: int = 1
    status: str
    duration_ms: int = 0


class RunDetailResponse(BaseModel):
    """Full detail for a single run."""

    pipeline_run_id: str
    pipeline: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_cost_usd: float | None = None
    total_duration_ms: int | None = None
    total_llm_calls: int | None = None
    concept_title: str | None = None
    final_video_path: str | None = None
    errors: list[str] = Field(default_factory=list)
    steps: list[StepSummary] = Field(default_factory=list)


class RunListResponse(BaseModel):
    """Paginated run list."""

    runs: list[RunSummaryResponse]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Events (for inspect / replay)
# ---------------------------------------------------------------------------

class EventResponse(BaseModel):
    """A single trace event from events.jsonl."""

    event_type: str
    event_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class EventListResponse(BaseModel):
    """All events for a run."""

    run_id: str
    events: list[dict[str, Any]]
    count: int


# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------

class PipelineStartRequest(BaseModel):
    """Request body for starting a new pipeline run."""

    pipeline: str = "physics"


class PipelineStartResponse(BaseModel):
    """Response when a pipeline run is queued."""

    pipeline_run_id: str
    pipeline: str
    status: str = "started"
    message: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "0.3.0"
    database: str = "unknown"
