"""Flat surface primitive.

Creates a flat rectangular surface at a given position and height.
Runs inside Blender — requires bpy.
"""

from __future__ import annotations

import math
from typing import Any


def create_flat_surface(
    position: tuple[float, float, float],
    width: float,
    depth: float,
    *,
    rotation_z: float = 0.0,
    material_name: str | None = None,
    is_static: bool = True,
    friction: float = 0.8,
) -> dict[str, Any]:
    """Create a flat rectangular surface with passive rigid body.

    Args:
        position: (x, y, z) centre of the surface.
        width: Size along the local X axis (metres).
        depth: Size along the local Y axis (metres).
        rotation_z: Rotation about Z axis (degrees).
        material_name: Optional Blender material name to assign.
        is_static: If True the surface is a passive rigid body.
        friction: Collision friction coefficient.

    Returns:
        Build-params dict for the Blender mesh builder.
    """
    return {
        "type": "flat_surface",
        "position": list(position),
        "width": width,
        "depth": depth,
        "rotation_z": rotation_z,
        "material_name": material_name,
        "is_static": is_static,
        "friction": friction,
    }
