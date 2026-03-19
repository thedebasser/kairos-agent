"""Kairos Agent -- WebSocket Endpoints.

Live event streaming for the Angular dashboard.

Two modes:
1. **Live** (``/ws/runs/{run_id}``) — tails ``events.jsonl`` and pushes
   new events to connected clients in real-time.
2. **Broadcast** (``/ws/events``) — global feed of all events across all
   active runs (dashboard overview).

Protocol:
  - Server sends JSON messages: ``{"type": "event", "data": <event_dict>}``
  - Client can send ``{"type": "ping"}`` to keep alive.
  - Server sends ``{"type": "pong"}`` in response.
  - Server sends ``{"type": "complete"}`` when the run finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from kairos.api.deps import get_runs_dir

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

# Active WebSocket connections per run_id
_connections: dict[str, set[WebSocket]] = {}
# Global broadcast subscribers
_broadcast_connections: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Per-run live stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/runs/{run_id}")
async def ws_run_events(websocket: WebSocket, run_id: str) -> None:
    """Stream events for a specific run via a tailing loop."""
    await websocket.accept()
    _connections.setdefault(run_id, set()).add(websocket)
    logger.info("WebSocket connected for run %s", run_id)

    runs_dir = get_runs_dir()
    events_path = runs_dir / run_id / "events.jsonl"

    try:
        offset = 0
        stale_ticks = 0
        max_stale = 300  # ~5 minutes with 1s ticks

        while True:
            # Read new lines from events.jsonl
            new_events = _read_new_lines(events_path, offset)
            if new_events:
                stale_ticks = 0
                for event_line, new_offset in new_events:
                    offset = new_offset
                    try:
                        event_data = json.loads(event_line)
                        await websocket.send_json({"type": "event", "data": event_data})

                        # Check for run completion
                        if event_data.get("event_type") == "run_completed":
                            await websocket.send_json({"type": "complete"})
                            return
                    except json.JSONDecodeError:
                        continue
            else:
                stale_ticks += 1
                if stale_ticks >= max_stale:
                    await websocket.send_json({
                        "type": "complete",
                        "reason": "timeout",
                    })
                    return

            # Check for incoming messages (ping/pong keepalive)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=1.0
                )
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for run %s", run_id)
    except Exception:
        logger.exception("WebSocket error for run %s", run_id)
    finally:
        _connections.get(run_id, set()).discard(websocket)


# ---------------------------------------------------------------------------
# Global broadcast
# ---------------------------------------------------------------------------

@router.websocket("/ws/events")
async def ws_global_events(websocket: WebSocket) -> None:
    """Global event broadcast — receives all events from all runs."""
    await websocket.accept()
    _broadcast_connections.add(websocket)
    logger.info("Global WebSocket connected")

    try:
        while True:
            # Keep connection alive, handle client messages
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=30.0
                )
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("Global WebSocket disconnected")
    except Exception:
        logger.exception("Global WebSocket error")
    finally:
        _broadcast_connections.discard(websocket)


# ---------------------------------------------------------------------------
# Broadcast helper (called by tracer or middleware)
# ---------------------------------------------------------------------------

async def broadcast_event(event_data: dict[str, Any]) -> None:
    """Push an event to all global subscribers and per-run subscribers.

    Safe to call from anywhere — silently drops failed sends.
    """
    run_id = event_data.get("run_id", "")
    message = {"type": "event", "data": event_data}

    # Per-run subscribers
    for ws in list(_connections.get(run_id, set())):
        try:
            await ws.send_json(message)
        except Exception:
            _connections.get(run_id, set()).discard(ws)

    # Global subscribers
    for ws in list(_broadcast_connections):
        try:
            await ws.send_json(message)
        except Exception:
            _broadcast_connections.discard(ws)


# ---------------------------------------------------------------------------
# File tailing
# ---------------------------------------------------------------------------

def _read_new_lines(
    path: Path,
    offset: int,
) -> list[tuple[str, int]]:
    """Read new lines from a file starting at byte *offset*.

    Returns list of (line_text, new_offset) pairs.
    """
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            results: list[tuple[str, int]] = []
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    results.append((line, f.tell()))
                else:
                    # Still advance offset past blank lines
                    results.append(("", f.tell()))
            return [(text, off) for text, off in results if text]
    except OSError:
        return []
