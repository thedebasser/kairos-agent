"""Platform connector primitive.

Generates a flat platform at a given height — used as a landing or
staging area between other connectors.
"""

from __future__ import annotations

from kairos.skills.contracts import ConnectorResult, Vector3


def create_platform(
    center: Vector3,
    width: float = 0.5,
    depth: float = 0.5,
    *,
    point_spacing: float = 0.05,
) -> ConnectorResult:
    """Generate a flat platform.

    Args:
        center: Center of the platform (Z = surface height).
        width: Platform width (X axis).
        depth: Platform depth (Y axis).
        point_spacing: Distance between waypoints across the platform.

    Returns:
        ConnectorResult with waypoints and build parameters.
    """
    # Waypoints run across the center of the platform (X axis)
    num_points = max(2, int(width / point_spacing) + 1)
    half_w = width / 2

    waypoints: list[Vector3] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = center.x - half_w + t * width
        waypoints.append(Vector3(x, center.y, center.z))

    total_length = width

    return ConnectorResult(
        waypoints=waypoints,
        connector_type="platform",
        params={
            "center": center.to_tuple(),
            "width": width,
            "depth": depth,
            "thickness": 0.02,
        },
        total_length=total_length,
        max_gradient=0.0,
        footprint=(width, depth),
    )
