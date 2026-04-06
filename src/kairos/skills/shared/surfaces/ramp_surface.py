"""Ramp surface primitive.

Creates an inclined rectangular surface between two heights.
Runs inside Blender — requires bpy.
"""

from __future__ import annotations

import math
from typing import Any


def create_ramp_surface(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    width: float,
    *,
    material_name: str | None = None,
    is_static: bool = True,
    friction: float = 0.8,
) -> dict[str, Any]:
    """Create an inclined ramp surface with passive rigid body.

    The ramp spans from *start* to *end*.  The surface is oriented so
    that *width* runs perpendicular to the slope direction.

    Args:
        start: (x, y, z) bottom-edge centre.
        end: (x, y, z) top-edge centre.
        width: Cross-slope width (metres).
        material_name: Optional Blender material name to assign.
        is_static: If True the surface is a passive rigid body.
        friction: Collision friction coefficient.

    Returns:
        Build-params dict for the Blender mesh builder.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    horizontal = math.sqrt(dx * dx + dy * dy)
    slope_length = math.sqrt(horizontal * horizontal + dz * dz)
    angle_deg = math.degrees(math.atan2(dz, horizontal)) if horizontal > 1e-9 else 90.0

    return {
        "type": "ramp_surface",
        "start": list(start),
        "end": list(end),
        "width": width,
        "slope_length": slope_length,
        "angle_degrees": angle_deg,
        "material_name": material_name,
        "is_static": is_static,
        "friction": friction,
    }
