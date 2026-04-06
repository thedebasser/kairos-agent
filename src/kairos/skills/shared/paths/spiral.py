"""Spiral path primitive.

Generates waypoints along a helical spiral in 3D space.
Pure math — no Blender dependency.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import PathResult, Vector3


def create_spiral(
    center: Vector3,
    radius: float,
    turns: float,
    start_height: float,
    end_height: float,
    *,
    direction: str = "counterclockwise",
    point_spacing: float = 0.05,
) -> PathResult:
    """Generate waypoints along a helical spiral.

    Args:
        center: Center of the spiral (XY position; Z ignored).
        radius: Radius of the spiral.
        turns: Number of full turns.
        start_height: Z height at spiral start.
        end_height: Z height at spiral end.
        direction: "clockwise" or "counterclockwise".
        point_spacing: Approximate distance between consecutive waypoints.

    Returns:
        PathResult with dense waypoints along the helix.
    """
    if radius < 1e-9 or turns < 1e-9:
        pt = Vector3(center.x, center.y, start_height)
        return PathResult(
            waypoints=[pt],
            total_length=0.0,
            segment_type="spiral",
            gradients=[0.0],
        )

    # Total arc length of a helix: sqrt((2*pi*r*turns)^2 + height_diff^2)
    height_diff = end_height - start_height
    circumference = 2 * math.pi * radius * turns
    total_length = math.sqrt(circumference ** 2 + height_diff ** 2)

    num_points = max(2, int(total_length / point_spacing) + 1)

    sign = -1.0 if direction == "clockwise" else 1.0
    total_angle = turns * 2 * math.pi

    waypoints: list[Vector3] = []
    gradients: list[float] = []

    for i in range(num_points):
        t = i / (num_points - 1)
        angle = sign * t * total_angle
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = start_height + t * height_diff

        waypoints.append(Vector3(x, y, z))

        # Local gradient: height change per horizontal distance per step
        horizontal_per_step = circumference / max(num_points - 1, 1)
        vertical_per_step = abs(height_diff) / max(num_points - 1, 1)
        if horizontal_per_step > 1e-9:
            grad = math.degrees(math.atan2(
                height_diff / max(num_points - 1, 1),
                horizontal_per_step,
            ))
        else:
            grad = 0.0
        gradients.append(grad)

    return PathResult(
        waypoints=waypoints,
        total_length=total_length,
        segment_type="spiral",
        gradients=gradients,
    )
