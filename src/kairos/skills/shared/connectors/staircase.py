"""Staircase connector primitive.

Generates waypoints and build parameters for a staircase that bridges
two heights with discrete flat steps and vertical risers.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import ConnectorResult, Vector3


def create_staircase(
    start: Vector3,
    end: Vector3,
    *,
    step_count: int | None = None,
    max_step_height: float = 0.15,
    width: float = 0.15,
    points_per_step: int = 4,
) -> ConnectorResult:
    """Generate a staircase connector between two 3D positions.

    Args:
        start: Bottom of the staircase.
        end: Top of the staircase.
        step_count: Number of steps (auto-calculated from height if None).
        max_step_height: Maximum riser height per step (metres).
        width: Staircase width.
        points_per_step: Waypoints per step tread.

    Returns:
        ConnectorResult with waypoints on step treads and build params.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    height_diff = abs(dz)
    horizontal = math.sqrt(dx * dx + dy * dy)

    if height_diff < 1e-9:
        return ConnectorResult(
            waypoints=[start, end],
            connector_type="staircase",
            total_length=horizontal,
            max_gradient=0.0,
        )

    if step_count is None:
        step_count = max(1, math.ceil(height_diff / max_step_height))

    step_rise = dz / step_count
    step_run = horizontal / step_count if step_count > 0 else 0

    # Direction
    if horizontal > 1e-9:
        dir_x = dx / horizontal
        dir_y = dy / horizontal
    else:
        dir_x, dir_y = 1.0, 0.0

    waypoints: list[Vector3] = []
    for s in range(step_count):
        step_z = start.z + s * step_rise
        h_start = s * step_run

        for p in range(points_per_step):
            t = p / max(points_per_step - 1, 1)
            h = h_start + t * step_run
            waypoints.append(Vector3(
                start.x + dir_x * h,
                start.y + dir_y * h,
                step_z,
            ))

    # Add final point at the top
    waypoints.append(end)

    total_length = sum(
        waypoints[i].distance_to(waypoints[i - 1])
        for i in range(1, len(waypoints))
    )

    # Overall gradient
    overall_gradient = math.degrees(math.atan2(dz, horizontal)) if horizontal > 1e-9 else 90.0

    footprint_w = max(abs(dx), width)
    footprint_d = max(abs(dy), width)

    return ConnectorResult(
        waypoints=waypoints,
        connector_type="staircase",
        params={
            "start": start.to_tuple(),
            "end": end.to_tuple(),
            "step_count": step_count,
            "step_rise": step_rise,
            "step_run": step_run,
            "width": width,
            "thickness": 0.02,
        },
        total_length=total_length,
        max_gradient=abs(overall_gradient),
        footprint=(footprint_w, footprint_d),
    )
