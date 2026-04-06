"""S-curve path primitive.

Generates waypoints along a sinusoidal S-curve in 3D space.
Pure math — no Blender dependency.
"""

from __future__ import annotations

import math

from kairos.skills.contracts import PathResult, Vector3


def create_s_curve(
    start: Vector3,
    end: Vector3,
    *,
    amplitude: float = 1.0,
    cycles: float = 1.0,
    height_start: float | None = None,
    height_end: float | None = None,
    point_spacing: float = 0.05,
) -> PathResult:
    """Generate waypoints along an S-curve between two points.

    The primary axis runs from *start* to *end*. The sinusoidal
    oscillation is perpendicular to this axis.

    Args:
        start: Start position.
        end: End position.
        amplitude: Peak lateral displacement from the centreline.
        cycles: Number of full sinusoidal oscillations.
        height_start: Z height at start (defaults to start.z).
        height_end: Z height at end (defaults to end.z).
        point_spacing: Approximate distance between consecutive waypoints.

    Returns:
        PathResult with waypoints along the S-curve.
    """
    if height_start is None:
        height_start = start.z
    if height_end is None:
        height_end = end.z

    # Primary axis direction (XY plane)
    dx = end.x - start.x
    dy = end.y - start.y
    primary_length = math.sqrt(dx * dx + dy * dy)

    if primary_length < 1e-9:
        return PathResult(
            waypoints=[start],
            total_length=0.0,
            segment_type="s_curve",
            gradients=[0.0],
        )

    # Unit vectors: forward along centreline, lateral perpendicular
    fwd_x = dx / primary_length
    fwd_y = dy / primary_length
    lat_x = -fwd_y  # 90° rotation
    lat_y = fwd_x

    # Estimate total arc length for point count
    # For a sine curve: L ≈ integral of sqrt(1 + (A*2*pi*c/L * cos(...))^2) dt
    # Use numerical approximation
    n_sample = 500
    arc_len = 0.0
    prev_x, prev_y, prev_z = start.x, start.y, height_start
    for i in range(1, n_sample + 1):
        t = i / n_sample
        along = t * primary_length
        lateral = amplitude * math.sin(2 * math.pi * cycles * t)
        px = start.x + fwd_x * along + lat_x * lateral
        py = start.y + fwd_y * along + lat_y * lateral
        pz = height_start + t * (height_end - height_start)
        arc_len += math.sqrt(
            (px - prev_x) ** 2 + (py - prev_y) ** 2 + (pz - prev_z) ** 2,
        )
        prev_x, prev_y, prev_z = px, py, pz

    num_points = max(2, int(arc_len / point_spacing) + 1)

    waypoints: list[Vector3] = []
    gradients: list[float] = []

    for i in range(num_points):
        t = i / (num_points - 1)
        along = t * primary_length
        lateral = amplitude * math.sin(2 * math.pi * cycles * t)
        px = start.x + fwd_x * along + lat_x * lateral
        py = start.y + fwd_y * along + lat_y * lateral
        pz = height_start + t * (height_end - height_start)

        waypoints.append(Vector3(px, py, pz))

    # Compute gradients between consecutive points
    for i in range(len(waypoints)):
        if i == 0:
            if len(waypoints) > 1:
                w0, w1 = waypoints[0], waypoints[1]
            else:
                gradients.append(0.0)
                continue
        elif i == len(waypoints) - 1:
            w0, w1 = waypoints[-2], waypoints[-1]
        else:
            w0, w1 = waypoints[i - 1], waypoints[i + 1]

        hdist = math.sqrt((w1.x - w0.x) ** 2 + (w1.y - w0.y) ** 2)
        vdist = w1.z - w0.z
        if hdist > 1e-9:
            gradients.append(math.degrees(math.atan2(vdist, hdist)))
        else:
            gradients.append(0.0)

    return PathResult(
        waypoints=waypoints,
        total_length=arc_len,
        segment_type="s_curve",
        gradients=gradients,
    )
