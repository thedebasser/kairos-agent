"""Ramp connector primitive.

Generates waypoints and build parameters for a straight inclined ramp
connecting two heights. Pure math for waypoint generation; the Blender
mesh builder uses the params to create geometry.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import ConnectorResult, Vector3


def create_ramp(
    start: Vector3,
    end: Vector3,
    *,
    width: float = 0.15,
    has_rails: bool = False,
    point_spacing: float = 0.05,
) -> ConnectorResult:
    """Generate a straight ramp between two 3D positions.

    Args:
        start: Bottom of the ramp.
        end: Top of the ramp.
        width: Ramp width (metres).
        has_rails: Whether to add side rails for stability.
        point_spacing: Distance between waypoints.

    Returns:
        ConnectorResult with waypoints along the ramp surface and
        build parameters for the Blender mesh builder.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    total_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    horizontal = math.sqrt(dx * dx + dy * dy)

    if total_length < 1e-9:
        return ConnectorResult(
            waypoints=[start],
            connector_type="ramp",
            total_length=0.0,
            max_gradient=0.0,
        )

    gradient = math.degrees(math.atan2(dz, horizontal)) if horizontal > 1e-9 else 90.0

    num_points = max(2, int(total_length / point_spacing) + 1)
    waypoints: list[Vector3] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        waypoints.append(Vector3(
            start.x + t * dx,
            start.y + t * dy,
            start.z + t * dz,
        ))

    # Footprint bounding box on the XY plane
    footprint_w = max(abs(dx), width)
    footprint_d = max(abs(dy), width)

    return ConnectorResult(
        waypoints=waypoints,
        connector_type="ramp",
        params={
            "start": start.to_tuple(),
            "end": end.to_tuple(),
            "width": width,
            "has_rails": has_rails,
            "gradient_degrees": gradient,
            "thickness": 0.02,
        },
        total_length=total_length,
        max_gradient=abs(gradient),
        footprint=(footprint_w, footprint_d),
    )
