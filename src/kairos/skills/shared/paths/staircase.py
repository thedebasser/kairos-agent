"""Staircase path primitive.

Generates waypoints along a staircase (flat steps with vertical risers).
Pure math — no Blender dependency.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import PathResult, Vector3


def create_staircase_path(
    start: Vector3,
    end: Vector3,
    *,
    step_count: int = 10,
    points_per_step: int = 4,
) -> PathResult:
    """Generate waypoints along a staircase from *start* to *end*.

    Each step is a horizontal flat segment; the vertical rise is implicit
    (the riser itself has no waypoints — dominos only stand on flat treads).

    Args:
        start: Bottom of the staircase.
        end: Top of the staircase.
        step_count: Number of steps.
        points_per_step: Waypoints per step tread (for domino placement).

    Returns:
        PathResult with waypoints on each step tread.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z

    horizontal_dist = math.sqrt(dx * dx + dy * dy)

    if step_count < 1:
        step_count = 1

    step_rise = dz / step_count
    step_run = horizontal_dist / step_count

    # Direction of horizontal travel
    if horizontal_dist > 1e-9:
        dir_x = dx / horizontal_dist
        dir_y = dy / horizontal_dist
    else:
        dir_x, dir_y = 1.0, 0.0

    waypoints: list[Vector3] = []
    gradients: list[float] = []

    for s in range(step_count):
        step_z = start.z + s * step_rise
        step_start_h = s * step_run

        for p in range(points_per_step):
            t = p / max(points_per_step - 1, 1)
            h = step_start_h + t * step_run
            waypoints.append(Vector3(
                start.x + dir_x * h,
                start.y + dir_y * h,
                step_z,
            ))
            gradients.append(0.0)  # flat on each tread

    # Compute total path length (horizontal only, along treads)
    total_length = 0.0
    for i in range(1, len(waypoints)):
        total_length += waypoints[i].distance_to(waypoints[i - 1])

    # Overall gradient for the staircase
    overall_gradient = math.degrees(math.atan2(dz, horizontal_dist)) if horizontal_dist > 1e-9 else 90.0

    return PathResult(
        waypoints=waypoints,
        total_length=total_length,
        segment_type="staircase",
        gradients=gradients,
    )
