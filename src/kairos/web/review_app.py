"""Kairos Agent — Human Review Dashboard.

Minimal FastAPI + Jinja2 web application for reviewing pipeline outputs.
Single page, no auth for POC (localhost only).

Features:
- Inline video player with concept summary and stats
- One-click approve/reject with reason codes
- Optional feedback text capture
- Queue navigation (previous/next)
- Discord webhook notification on new reviews
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

from kairos.config import get_settings
from kairos.schemas.contracts import ReviewAction

logger = logging.getLogger(__name__)

# Paths
_THIS_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _THIS_DIR / "templates"
_STATIC_DIR = _THIS_DIR / "static"

# Ensure directories exist
_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
_STATIC_DIR.mkdir(parents=True, exist_ok=True)


def create_review_app() -> FastAPI:
    """Create and configure the review dashboard FastAPI application."""
    app = FastAPI(
        title="Kairos Review Dashboard",
        description="Human review gate for the simulation content pipeline.",
        version="0.1.0",
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Mount static files
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # -------------------------------------------------------------------------
    # In-memory review store (for POC — production uses PostgreSQL)
    # -------------------------------------------------------------------------
    review_store: dict[str, dict[str, Any]] = {}

    def _get_store() -> dict[str, dict[str, Any]]:
        return review_store

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def review_queue(request: Request, index: int = Query(0, ge=0)):
        """Main review page — shows one video at a time with navigation."""
        store = _get_store()
        pending = [
            (oid, data)
            for oid, data in store.items()
            if data.get("status") == "pending_review"
        ]
        pending.sort(key=lambda x: x[1].get("created_at", ""))

        total = len(pending)

        if total == 0:
            return templates.TemplateResponse(
                "review.html",
                {
                    "request": request,
                    "has_items": False,
                    "total_pending": 0,
                    "current_index": 0,
                },
            )

        # Clamp index
        index = min(index, total - 1)
        output_id, output_data = pending[index]

        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "has_items": True,
                "total_pending": total,
                "current_index": index,
                "output_id": output_id,
                "output": output_data,
                "prev_index": max(0, index - 1),
                "next_index": min(total - 1, index + 1),
                "has_prev": index > 0,
                "has_next": index < total - 1,
                "review_actions": [
                    {"value": ReviewAction.APPROVED.value, "label": "✅ Approve", "style": "success"},
                    {"value": ReviewAction.BAD_CONCEPT.value, "label": "❌ Bad Concept", "style": "danger"},
                    {"value": ReviewAction.BAD_SIMULATION.value, "label": "❌ Bad Sim", "style": "danger"},
                    {"value": ReviewAction.BAD_EDIT.value, "label": "❌ Bad Edit", "style": "danger"},
                    {"value": ReviewAction.REQUEST_REEDIT.value, "label": "🔄 Re-edit", "style": "warning"},
                ],
            },
        )

    @app.post("/review/{output_id}")
    async def submit_review(
        output_id: str,
        action: str = Form(...),
        feedback: str = Form(""),
    ):
        """Process a review decision."""
        store = _get_store()

        if output_id not in store:
            raise HTTPException(status_code=404, detail="Output not found")

        # Validate action
        valid_actions = {a.value for a in ReviewAction}
        if action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

        # Update store
        status = "approved" if action == ReviewAction.APPROVED.value else "rejected"
        store[output_id]["status"] = status
        store[output_id]["review_action"] = action
        store[output_id]["review_feedback"] = feedback

        logger.info(
            "Review submitted: output=%s action=%s feedback=%s",
            output_id,
            action,
            feedback[:100] if feedback else "",
        )

        # Send Discord notification
        try:
            await _send_discord_notification(output_id, action, feedback)
        except Exception as e:
            logger.warning("Failed to send Discord notification: %s", e)

        return RedirectResponse(url="/", status_code=303)

    @app.get("/video/{output_id}")
    async def serve_video(output_id: str):
        """Serve a video file for the inline player."""
        store = _get_store()

        if output_id not in store:
            raise HTTPException(status_code=404, detail="Output not found")

        video_path = store[output_id].get("final_video_path", "")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"{output_id}.mp4",
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        store = _get_store()
        pending = sum(1 for d in store.values() if d.get("status") == "pending_review")
        return {"status": "ok", "pending_reviews": pending}

    @app.get("/api/queue")
    async def api_queue():
        """API endpoint returning the review queue as JSON."""
        store = _get_store()
        pending = [
            {"output_id": oid, **data}
            for oid, data in store.items()
            if data.get("status") == "pending_review"
        ]
        pending.sort(key=lambda x: x.get("created_at", ""))
        return {"total": len(pending), "items": pending}

    @app.get("/api/failed")
    async def api_failed():
        """API endpoint returning failed pipeline runs with recovery options."""
        store = _get_store()
        failed = [
            {"output_id": oid, **data}
            for oid, data in store.items()
            if data.get("status") == "failed"
        ]
        failed.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"total": len(failed), "items": failed}

    @app.post("/api/resume/{pipeline_run_id}")
    async def api_resume(pipeline_run_id: str):
        """Resume a failed pipeline from its last checkpoint."""
        try:
            from kairos.orchestrator.graph import resume_pipeline

            final_state = await resume_pipeline(pipeline_run_id)
            return {
                "status": "resumed",
                "pipeline_run_id": pipeline_run_id,
                "result_status": final_state.get("status", "unknown"),
            }
        except Exception as e:
            logger.error("Failed to resume pipeline %s: %s", pipeline_run_id, e)
            raise HTTPException(status_code=500, detail=f"Resume failed: {e}")

    @app.post("/api/restart/{pipeline_run_id}")
    async def api_restart(pipeline_run_id: str):
        """Restart a failed pipeline entirely from scratch."""
        try:
            from kairos.orchestrator.graph import run_pipeline

            # Look up original pipeline name
            pipeline_name = "physics"  # default
            store = _get_store()
            for data in store.values():
                if data.get("pipeline_run_id") == pipeline_run_id:
                    pipeline_name = data.get("pipeline", "physics")
                    break

            final_state = await run_pipeline(pipeline_name)
            return {
                "status": "restarted",
                "original_run_id": pipeline_run_id,
                "result_status": final_state.get("status", "unknown"),
            }
        except Exception as e:
            logger.error("Failed to restart pipeline %s: %s", pipeline_run_id, e)
            raise HTTPException(status_code=500, detail=f"Restart failed: {e}")

    @app.post("/api/add")
    async def api_add_to_queue(item: dict[str, Any]):
        """Add a video output to the review queue (called by pipeline)."""
        store = _get_store()
        output_id = item.get("output_id", str(UUID(int=0)))
        item.setdefault("status", "pending_review")
        store[output_id] = item

        logger.info("Added to review queue: %s", output_id)

        # Notify via Discord
        try:
            settings = get_settings()
            if settings.discord_webhook_url:
                await _send_discord_new_review(output_id, item)
        except Exception as e:
            logger.warning("Failed to send Discord notification: %s", e)

        return {"status": "queued", "output_id": output_id}

    return app


async def _send_discord_notification(output_id: str, action: str, feedback: str) -> None:
    """Send a Discord webhook notification for a review decision."""
    settings = get_settings()
    webhook_url = getattr(settings, "discord_webhook_url", None)
    if not webhook_url:
        return

    import httpx

    emoji = "✅" if action == "approved" else "❌"
    colour = 0x2ECC71 if action == "approved" else 0xE74C3C
    message = {
        "embeds": [
            {
                "title": f"{emoji} Review Decision",
                "color": colour,
                "fields": [
                    {"name": "Output", "value": f"`{output_id}`", "inline": True},
                    {"name": "Action", "value": f"**{action}**", "inline": True},
                    {"name": "Feedback", "value": feedback or "(none)", "inline": False},
                ],
            }
        ],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(webhook_url, json=message, timeout=10)
        resp.raise_for_status()


async def _send_discord_new_review(output_id: str, item: dict[str, Any]) -> None:
    """Send a Discord webhook notification when a new video enters review."""
    settings = get_settings()
    webhook_url = getattr(settings, "discord_webhook_url", None)
    if not webhook_url:
        return

    import httpx

    title = item.get("title", "Untitled")
    category = item.get("category", "Unknown")
    message = {
        "embeds": [
            {
                "title": "🎬 New Video Ready for Review",
                "color": 0x3498DB,
                "fields": [
                    {"name": "Title", "value": title, "inline": True},
                    {"name": "Category", "value": category, "inline": True},
                    {"name": "Output", "value": f"`{output_id}`", "inline": False},
                ],
            }
        ],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(webhook_url, json=message, timeout=10)
        resp.raise_for_status()


# Convenience: create default app instance
app = create_review_app()
