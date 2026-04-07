"""Marble run track pieces — parametric building blocks.

Each track piece is a pure-data function returning build params for
the Blender mesh builder. Pieces connect via a standard connector
system with entry/exit port descriptors.

Track piece types:
    - straight_track: Linear track segment
    - curved_track: Arc/banked curve
    - funnel: Wide-to-narrow funnel (marble collection)
    - loop: Vertical loop-the-loop
    - drop: Vertical drop segment
"""

from __future__ import annotations

import math
from typing import Any

from kairos.skills.contracts import Vector3


# ---------------------------------------------------------------------------
# Connector port descriptors
# ---------------------------------------------------------------------------

def _port(
    position: tuple[float, float, float],
    direction: tuple[float, float, float],
    *,
    diameter: float | None = None,
) -> dict[str, Any]:
    """Describe a connector port on a track piece."""
    from kairos.skills.marble_run import TRACK_CONNECTOR_DIAMETER

    return {
        "position": list(position),
        "direction": list(direction),
        "diameter": diameter or TRACK_CONNECTOR_DIAMETER,
    }


# ---------------------------------------------------------------------------
# Track piece functions
# ---------------------------------------------------------------------------


def create_straight_track(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    width: float = 0.10,
    wall_height: float | None = None,
) -> dict[str, Any]:
    """Create a straight track segment.

    Args:
        start: Entry point (x, y, z).
        end: Exit point (x, y, z).
        width: Track width in metres.
        wall_height: Side wall height. Defaults to TRACK_WALL_HEIGHT.

    Returns:
        Track piece build-params dict.
    """
    from kairos.skills.marble_run import TRACK_WALL_HEIGHT

    wh = wall_height if wall_height is not None else TRACK_WALL_HEIGHT

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Direction unit vector
    if length > 1e-9:
        d = (dx / length, dy / length, dz / length)
    else:
        d = (1.0, 0.0, 0.0)

    return {
        "piece_type": "straight_track",
        "start": list(start),
        "end": list(end),
        "length": round(length, 6),
        "width": width,
        "wall_height": wh,
        "height_drop": round(start[2] - end[2], 6),
        "entry_port": _port(start, d),
        "exit_port": _port(end, d),
    }


def create_curved_track(
    center: tuple[float, float, float],
    radius: float,
    start_angle_deg: float,
    sweep_deg: float,
    *,
    start_height: float | None = None,
    end_height: float | None = None,
    width: float = 0.10,
    wall_height: float | None = None,
    bank_angle_deg: float = 0.0,
) -> dict[str, Any]:
    """Create a banked curved track segment.

    Args:
        center: Centre of arc circle (x, y, z).
        radius: Arc radius in metres.
        start_angle_deg: Starting angle in degrees (0 = +X axis).
        sweep_deg: Arc sweep in degrees (positive = CCW).
        start_height: Override Z at entry. Defaults to center[2].
        end_height: Override Z at exit. Defaults to center[2].
        width: Track width in metres.
        wall_height: Side wall height.
        bank_angle_deg: Inward bank angle for the curve surface.

    Returns:
        Track piece build-params dict.
    """
    from kairos.skills.marble_run import TRACK_WALL_HEIGHT

    wh = wall_height if wall_height is not None else TRACK_WALL_HEIGHT
    h_start = start_height if start_height is not None else center[2]
    h_end = end_height if end_height is not None else center[2]

    arc_length = abs(sweep_deg) / 360.0 * 2.0 * math.pi * radius

    # Compute entry/exit positions on the arc
    sa = math.radians(start_angle_deg)
    ea = math.radians(start_angle_deg + sweep_deg)

    entry_pos = (
        center[0] + radius * math.cos(sa),
        center[1] + radius * math.sin(sa),
        h_start,
    )
    exit_pos = (
        center[0] + radius * math.cos(ea),
        center[1] + radius * math.sin(ea),
        h_end,
    )

    # Tangent directions (perpendicular to radius, in sweep direction)
    sign = 1.0 if sweep_deg >= 0 else -1.0
    entry_dir = (-sign * math.sin(sa), sign * math.cos(sa), 0.0)
    exit_dir = (-sign * math.sin(ea), sign * math.cos(ea), 0.0)

    return {
        "piece_type": "curved_track",
        "center": list(center),
        "radius": radius,
        "start_angle_deg": start_angle_deg,
        "sweep_deg": sweep_deg,
        "arc_length": round(arc_length, 6),
        "width": width,
        "wall_height": wh,
        "bank_angle_deg": bank_angle_deg,
        "height_drop": round(h_start - h_end, 6),
        "entry_port": _port(entry_pos, entry_dir),
        "exit_port": _port(exit_pos, exit_dir),
    }


def create_funnel(
    position: tuple[float, float, float],
    *,
    entry_width: float = 0.40,
    exit_width: float = 0.10,
    length: float = 0.30,
    height_drop: float = 0.05,
    wall_height: float | None = None,
) -> dict[str, Any]:
    """Create a funnel piece — wide entry narrowing to standard track width.

    Used at the start of a marble race to collect balls from a wider
    release area into a single track.

    Args:
        position: Entry centre point (x, y, z).
        entry_width: Width at the wide end.
        exit_width: Width at the narrow end.
        length: Funnel length along the travel direction.
        height_drop: Height decrease from entry to exit.
        wall_height: Side wall height.

    Returns:
        Track piece build-params dict.
    """
    from kairos.skills.marble_run import TRACK_WALL_HEIGHT

    wh = wall_height if wall_height is not None else TRACK_WALL_HEIGHT

    exit_pos = (
        position[0] + length,
        position[1],
        position[2] - height_drop,
    )

    return {
        "piece_type": "funnel",
        "position": list(position),
        "entry_width": entry_width,
        "exit_width": exit_width,
        "length": length,
        "wall_height": wh,
        "height_drop": round(height_drop, 6),
        "entry_port": _port(position, (1.0, 0.0, 0.0), diameter=entry_width),
        "exit_port": _port(exit_pos, (1.0, 0.0, 0.0)),
    }


def create_loop(
    entry: tuple[float, float, float],
    *,
    radius: float = 0.15,
    width: float = 0.10,
    wall_height: float | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Create a vertical loop-the-loop track piece.

    The loop sits in the plane defined by (direction, Z-up).  The entry
    point is at the bottom of the loop and the exit is also at the bottom,
    offset forward by 2× the track width.

    Args:
        entry: Entry point at the bottom of the loop.
        radius: Loop radius in metres.
        width: Track width in metres.
        wall_height: Side wall height.
        direction: Forward direction unit vector (travel direction).

    Returns:
        Track piece build-params dict.
    """
    from kairos.skills.marble_run import TRACK_WALL_HEIGHT

    wh = wall_height if wall_height is not None else TRACK_WALL_HEIGHT

    loop_circumference = 2.0 * math.pi * radius

    # Exit is slightly forward from entry (track overlaps at bottom)
    exit_pos = (
        entry[0] + direction[0] * width * 2.0,
        entry[1] + direction[1] * width * 2.0,
        entry[2],
    )

    return {
        "piece_type": "loop",
        "entry": list(entry),
        "radius": radius,
        "width": width,
        "wall_height": wh,
        "direction": list(direction),
        "loop_circumference": round(loop_circumference, 6),
        "height_drop": 0.0,  # Loop returns to same height
        "entry_port": _port(entry, direction),
        "exit_port": _port(exit_pos, direction),
    }


def create_drop(
    start: tuple[float, float, float],
    *,
    height: float = 0.50,
    width: float = 0.10,
    wall_height: float | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    forward_offset: float = 0.05,
) -> dict[str, Any]:
    """Create a vertical drop track piece.

    A short near-vertical segment that drops the marble to a lower level.
    A small forward offset prevents purely vertical motion.

    Args:
        start: Entry point at the top.
        height: Vertical drop distance in metres.
        width: Track width in metres.
        wall_height: Side wall height.
        direction: Forward direction.
        forward_offset: Horizontal offset to give the marble trajectory.

    Returns:
        Track piece build-params dict.
    """
    from kairos.skills.marble_run import TRACK_WALL_HEIGHT

    wh = wall_height if wall_height is not None else TRACK_WALL_HEIGHT
    segment_length = math.sqrt(height ** 2 + forward_offset ** 2)

    exit_pos = (
        start[0] + direction[0] * forward_offset,
        start[1] + direction[1] * forward_offset,
        start[2] - height,
    )

    return {
        "piece_type": "drop",
        "start": list(start),
        "height": height,
        "width": width,
        "wall_height": wh,
        "direction": list(direction),
        "forward_offset": forward_offset,
        "length": round(segment_length, 6),
        "height_drop": round(height, 6),
        "entry_port": _port(start, direction),
        "exit_port": _port(exit_pos, direction),
    }


# ---------------------------------------------------------------------------
# Track piece registry
# ---------------------------------------------------------------------------

TRACK_PIECES: dict[str, Any] = {
    "straight_track": create_straight_track,
    "curved_track": create_curved_track,
    "funnel": create_funnel,
    "loop": create_loop,
    "drop": create_drop,
}
