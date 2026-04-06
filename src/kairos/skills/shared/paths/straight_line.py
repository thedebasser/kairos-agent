"""Straight-line path primitive.

Generates evenly spaced waypoints along a 3D line segment.
Pure math — no Blender dependency.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import PathResult, Vector3


def create_straight_line(
    start: Vector3,
    end: Vector3,
    *,
    point_spacing: float = 0.05,
) -> PathResult:
    """Generate waypoints along a straight line from *start* to *end*.

    Args:
        start: Start position.
        end: End position.
        point_spacing: Distance between consecutive waypoints (metres).

    Returns:
        PathResult with evenly spaced waypoints.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    total_length = math.sqrt(dx * dx + dy * dy + dz * dz)

    if total_length < 1e-9:
        return PathResult(
            waypoints=[start],
            total_length=0.0,
            segment_type="straight_line",
            gradients=[0.0],
        )

    num_points = max(2, int(total_length / point_spacing) + 1)
    waypoints: list[Vector3] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        waypoints.append(Vector3(
            start.x + t * dx,
            start.y + t * dy,
            start.z + t * dz,
        ))

    # Gradient is constant along a straight line
    horizontal = math.sqrt(dx * dx + dy * dy)
    if horizontal < 1e-9:
        gradient = 90.0 if dz > 0 else -90.0
    else:
        gradient = math.degrees(math.atan2(dz, horizontal))

    gradients = [gradient] * num_points

    return PathResult(
        waypoints=waypoints,
        total_length=total_length,
        segment_type="straight_line",
        gradients=gradients,
    )
