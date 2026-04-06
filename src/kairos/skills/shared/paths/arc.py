"""Arc path primitive.

Generates waypoints along a circular arc in 3D space.
Pure math — no Blender dependency.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import PathResult, Vector3


def create_arc(
    center: Vector3,
    radius: float,
    start_angle: float = 0.0,
    end_angle: float = math.pi,
    *,
    height_start: float = 0.0,
    height_end: float = 0.0,
    point_spacing: float = 0.05,
) -> PathResult:
    """Generate waypoints along a circular arc.

    The arc lies in the XY plane around *center*, with optional
    linear height interpolation from *height_start* to *height_end*.

    Args:
        center: Center of the arc circle (Z ignored, uses height params).
        radius: Radius of the arc.
        start_angle: Start angle in radians (0 = +X axis).
        end_angle: End angle in radians.
        height_start: Z height at start of arc.
        height_end: Z height at end of arc.
        point_spacing: Approximate distance between consecutive waypoints.

    Returns:
        PathResult with waypoints along the arc.
    """
    if radius < 1e-9:
        pt = Vector3(center.x, center.y, height_start)
        return PathResult(
            waypoints=[pt],
            total_length=0.0,
            segment_type="arc",
            gradients=[0.0],
        )

    arc_length = abs(end_angle - start_angle) * radius
    num_points = max(2, int(arc_length / point_spacing) + 1)

    height_diff = height_end - height_start
    waypoints: list[Vector3] = []
    gradients: list[float] = []

    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = height_start + t * height_diff

        waypoints.append(Vector3(x, y, z))

        # Gradient: rise per horizontal distance
        if arc_length > 1e-9:
            gradients.append(math.degrees(math.atan2(height_diff, arc_length)))
        else:
            gradients.append(0.0)

    return PathResult(
        waypoints=waypoints,
        total_length=arc_length + abs(height_diff) * 0.01,  # approximate
        segment_type="arc",
        gradients=gradients,
    )
