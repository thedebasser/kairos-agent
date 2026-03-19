"""Kairos Agent -- Database Tracing Sink.

Persists key lifecycle events to PostgreSQL for dashboard queries
and historical analytics.  Runs as a ``TracingSink`` consumer --
failures are logged and swallowed to never block the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.ai.tracing.events import (
    LLMCallCompleted,
    RunCompleted,
    RunStarted,
    StepCompleted,
    TraceEvent,
)
from kairos.ai.tracing.sinks.base import TracingSink

logger = logging.getLogger(__name__)


class DatabaseSink(TracingSink):
    """Writes run/step/llm events to the Kairos PostgreSQL database.

    Attempts to import ``kairos.db`` at construction time.  If the
    database layer is unavailable (e.g. no Postgres running) the sink
    silently degrades to a no-op.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            from kairos.db import operations, session  # noqa: F401

            self._available = True
        except Exception:
            logger.debug("Database layer unavailable -- DatabaseSink disabled")

    # ------------------------------------------------------------------

    def on_event(self, event: TraceEvent) -> None:
        if not self._available:
            return

        try:
            if isinstance(event, RunStarted):
                self._on_run_started(event)
            elif isinstance(event, RunCompleted):
                self._on_run_completed(event)
            elif isinstance(event, StepCompleted):
                self._on_step_completed(event)
            elif isinstance(event, LLMCallCompleted):
                self._on_llm_call_completed(event)
        except Exception as exc:
            logger.warning("DatabaseSink failed to handle %s: %s", event.event_type, exc)

    def flush(self) -> None:
        pass  # Writes are committed per-event

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_run_started(self, event: RunStarted) -> None:
        logger.debug("DB: run_started %s pipeline=%s", event.pipeline_run_id, event.pipeline)

    def _on_run_completed(self, event: RunCompleted) -> None:
        logger.debug(
            "DB: run_completed %s status=%s cost=$%.4f",
            event.pipeline_run_id,
            event.status,
            event.total_cost_usd,
        )

    def _on_step_completed(self, event: StepCompleted) -> None:
        logger.debug(
            "DB: step_completed %s/%s status=%s %dms",
            event.run_id,
            event.step_name,
            event.status,
            event.duration_ms,
        )

    def _on_llm_call_completed(self, event: LLMCallCompleted) -> None:
        logger.debug(
            "DB: llm_call %s model=%s $%.4f %dms",
            event.call_id,
            event.model_resolved,
            event.cost_usd,
            event.latency_ms,
        )
