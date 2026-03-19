"""Kairos Agent -- FastAPI Application Factory.

Creates and configures the main API application.  All route modules are
registered here.  The review dashboard is mounted as a sub-application
at ``/review`` for backward compatibility.

Usage::

    uvicorn kairos.api.app:create_app --factory --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the main Kairos API application."""
    app = FastAPI(
        title="Kairos Agent API",
        description=(
            "REST + WebSocket API for the Kairos simulation content pipeline. "
            "Provides run management, live event streaming, and pipeline control."
        ),
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- CORS (allow Angular dev server) -----------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:4200",   # Angular dev server
            "http://localhost:3000",   # Alternative dev port
            "http://127.0.0.1:4200",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- REST routes -------------------------------------------------------
    from kairos.api.routes.runs import router as runs_router
    from kairos.api.routes.pipeline import router as pipeline_router
    from kairos.api.routes.ws import router as ws_router

    app.include_router(runs_router, prefix="/api")
    app.include_router(pipeline_router, prefix="/api")
    app.include_router(ws_router)

    # -- Health endpoint ---------------------------------------------------
    from kairos.api.schemas import HealthResponse

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        db_status = "unknown"
        try:
            from kairos.db.session import get_engine
            engine = get_engine()
            db_status = "connected" if engine else "not configured"
        except Exception:
            db_status = "unavailable"

        return HealthResponse(status="ok", database=db_status)

    # -- Mount review dashboard (legacy) -----------------------------------
    try:
        from kairos.web.review_app import create_review_app
        review_app = create_review_app()
        app.mount("/review", review_app)
        logger.info("Review dashboard mounted at /review")
    except Exception:
        logger.warning("Review dashboard unavailable — skipping mount")

    # -- Shutdown hook -----------------------------------------------------
    @app.on_event("shutdown")
    async def _shutdown() -> None:
        try:
            from kairos.db.session import dispose_engine
            await dispose_engine()
        except Exception:
            pass

    logger.info("Kairos API created: docs at /docs, WS at /ws/events")
    return app
