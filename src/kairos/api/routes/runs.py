"""Kairos Agent -- Run Endpoints.

REST endpoints for listing, viewing, and inspecting pipeline runs.
Data comes from two sources: the PostgreSQL database (run metadata)
and the ``runs/`` file tree (events, steps, prompts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kairos.api.deps import get_db, get_runs_dir
from kairos.api.schemas import (
    EventListResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummaryResponse,
    StepSummary,
)
from kairos.db.models import PipelineRun

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


# ---------------------------------------------------------------------------
# GET /runs  — paginated list
# ---------------------------------------------------------------------------

@router.get("", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    pipeline: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> RunListResponse:
    """List pipeline runs with optional filters."""
    base = select(PipelineRun)
    if pipeline:
        base = base.where(PipelineRun.pipeline == pipeline)
    if status:
        base = base.where(PipelineRun.status == status)

    # Total count
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Page
    stmt = base.order_by(PipelineRun.started_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return RunListResponse(
        runs=[_row_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}  — detail
# ---------------------------------------------------------------------------

@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    runs_dir: Path = Depends(get_runs_dir),
) -> RunDetailResponse:
    """Get full detail for a single run (DB + file artifacts)."""
    from uuid import UUID

    try:
        uid = UUID(run_id)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid UUID: {run_id}") from exc

    row = await db.get(PipelineRun, uid)
    if row is None:
        raise HTTPException(404, f"Run {run_id} not found")

    # Enrich with file artifacts
    run_dir = runs_dir / run_id
    summary_data = _load_run_summary(run_dir)

    steps: list[StepSummary] = []
    if summary_data and summary_data.get("steps"):
        for s in summary_data["steps"]:
            steps.append(StepSummary(
                step=s.get("step", ""),
                step_number=s.get("step_number", 0),
                attempt=s.get("attempt", 1),
                status=s.get("status", ""),
                duration_ms=s.get("duration_ms", 0),
            ))

    return RunDetailResponse(
        pipeline_run_id=str(row.pipeline_run_id),
        pipeline=row.pipeline,
        status=row.status,
        started_at=row.started_at,
        completed_at=row.completed_at,
        total_cost_usd=float(row.total_cost_usd) if row.total_cost_usd else None,
        total_duration_ms=summary_data.get("total_duration_ms") if summary_data else None,
        total_llm_calls=summary_data.get("total_llm_calls") if summary_data else None,
        concept_title=summary_data.get("concept_title") if summary_data else None,
        final_video_path=summary_data.get("final_video_path") if summary_data else None,
        errors=summary_data.get("errors", []) if summary_data else [],
        steps=steps,
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/events  — raw events
# ---------------------------------------------------------------------------

@router.get("/{run_id}/events", response_model=EventListResponse)
async def get_run_events(
    run_id: str,
    runs_dir: Path = Depends(get_runs_dir),
) -> EventListResponse:
    """Return all events from a run's events.jsonl."""
    events_path = runs_dir / run_id / "events.jsonl"
    if not events_path.exists():
        raise HTTPException(404, f"No events found for run {run_id}")

    events = _read_jsonl(events_path)
    return EventListResponse(run_id=run_id, events=events, count=len(events))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_summary(row: PipelineRun) -> RunSummaryResponse:
    return RunSummaryResponse(
        pipeline_run_id=str(row.pipeline_run_id),
        pipeline=row.pipeline,
        status=row.status,
        started_at=row.started_at,
        completed_at=row.completed_at,
        total_cost_usd=float(row.total_cost_usd) if row.total_cost_usd else None,
    )


def _load_run_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
