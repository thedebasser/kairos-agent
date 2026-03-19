"""Kairos Agent -- Pipeline Control Endpoints.

REST endpoints for starting, resuming, and querying pipeline status.
Pipeline execution is kicked off asynchronously via ``asyncio.create_task``
so the HTTP call returns immediately.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter

from kairos.api.schemas import PipelineStartRequest, PipelineStartResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ---------------------------------------------------------------------------
# POST /pipeline/start  — kick off a new run
# ---------------------------------------------------------------------------

@router.post("/start", response_model=PipelineStartResponse)
async def start_pipeline(body: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline run asynchronously.

    Returns immediately with the new run ID.  The pipeline executes in
    a background task; progress is available via the WebSocket endpoint.
    """
    run_id = uuid4().hex

    async def _run() -> None:
        try:
            from kairos.orchestrator.graph import run_pipeline
            await run_pipeline(body.pipeline)
        except Exception:
            logger.exception("Background pipeline run %s failed", run_id)

    asyncio.create_task(_run())
    logger.info("Pipeline %s started as background task: run=%s", body.pipeline, run_id)

    return PipelineStartResponse(
        pipeline_run_id=run_id,
        pipeline=body.pipeline,
        status="started",
        message=f"Pipeline '{body.pipeline}' started. Track via /ws/runs/{run_id}",
    )


# ---------------------------------------------------------------------------
# GET /pipeline/status  — lightweight health / availability
# ---------------------------------------------------------------------------

@router.get("/status")
async def pipeline_status() -> dict:
    """Return available pipelines and basic system status."""
    return {
        "available_pipelines": ["physics", "domino"],
        "status": "ready",
    }
