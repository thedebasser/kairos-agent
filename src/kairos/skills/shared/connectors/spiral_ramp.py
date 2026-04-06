"""Spiral ramp connector primitive.

Generates waypoints and build parameters for a helical spiral ramp
connecting two heights within a compact footprint.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import ConnectorResult, Vector3


def create_spiral_ramp(
    center: Vector3,
    radius: float,
    turns: float,
    start_height: float,
    end_height: float,
    *,
    direction: str = "clockwise",
    width: float = 0.12,
    point_spacing: float = 0.03,
) -> ConnectorResult:
    """Generate a spiral ramp connector.

    Args:
        center: Center of the spiral (XY; Z ignored).
        radius: Radius of the spiral.
        turns: Number of full turns (e.g. 3.5).
        start_height: Z at the spiral entry.
        end_height: Z at the spiral exit.
        direction: "clockwise" or "counterclockwise".
        width: Ramp surface width.
        point_spacing: Distance between waypoints.

    Returns:
        ConnectorResult with dense waypoints and build parameters.
    """
    if radius < 1e-9 or turns < 1e-9:
        pt = Vector3(center.x, center.y, start_height)
        return ConnectorResult(
            waypoints=[pt],
            connector_type="spiral_ramp",
            total_length=0.0,
            max_gradient=0.0,
        )

    height_diff = end_height - start_height
    circumference = 2 * math.pi * radius * turns
    total_length = math.sqrt(circumference ** 2 + height_diff ** 2)

    num_points = max(2, int(total_length / point_spacing) + 1)

    sign = -1.0 if direction == "clockwise" else 1.0
    total_angle = turns * 2 * math.pi

    waypoints: list[Vector3] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = sign * t * total_angle
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = start_height + t * height_diff
        waypoints.append(Vector3(x, y, z))

    # Max gradient (constant for a helix)
    horizontal_per_turn = 2 * math.pi * radius
    vertical_per_turn = abs(height_diff) / turns if turns > 0 else 0
    if horizontal_per_turn > 1e-9:
        max_gradient = abs(math.degrees(math.atan2(vertical_per_turn, horizontal_per_turn)))
    else:
        max_gradient = 90.0

    footprint = (radius * 2 + width, radius * 2 + width)

    return ConnectorResult(
        waypoints=waypoints,
        connector_type="spiral_ramp",
        params={
            "center": center.to_tuple(),
            "radius": radius,
            "turns": turns,
            "start_height": start_height,
            "end_height": end_height,
            "direction": direction,
            "width": width,
            "thickness": 0.02,
        },
        total_length=total_length,
        max_gradient=max_gradient,
        footprint=footprint,
    )
