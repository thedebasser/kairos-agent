"""Plank bridge connector primitive.

Generates a simple plank bridge connecting two surfaces at similar
heights — a thin flat surface spanning a gap.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import ConnectorResult, Vector3


def create_plank_bridge(
    start: Vector3,
    end: Vector3,
    *,
    width: float = 0.12,
) -> ConnectorResult:
    """Generate a plank bridge between two positions.

    Args:
        start: One end of the plank.
        end: Other end of the plank.
        width: Plank width.

    Returns:
        ConnectorResult with start/end waypoints and build parameters.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    total_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    horizontal = math.sqrt(dx * dx + dy * dy)

    gradient = math.degrees(math.atan2(dz, horizontal)) if horizontal > 1e-9 else 0.0

    # Simple: two endpoints, the mesh builder creates the plank between them
    waypoints = [start, end]

    footprint_w = max(abs(dx), width)
    footprint_d = max(abs(dy), width)

    return ConnectorResult(
        waypoints=waypoints,
        connector_type="plank_bridge",
        params={
            "start": start.to_tuple(),
            "end": end.to_tuple(),
            "width": width,
            "thickness": 0.02,
        },
        total_length=total_length,
        max_gradient=abs(gradient),
        footprint=(footprint_w, footprint_d),
    )
