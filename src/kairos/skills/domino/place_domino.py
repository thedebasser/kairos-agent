"""Place a domino in the Blender scene.

Runs inside Blender headless — requires bpy.
"""

from __future__ import annotations

import math
from typing import Any


def place_domino(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    *,
    scale: float = 1.0,
    domino_dims: tuple[float, float, float] | None = None,
    colour_hex: str = "#808080",
    name: str | None = None,
) -> dict[str, Any]:
    """Return build-params for a single domino placement.

    This is a pure-data function that returns a dict consumed by the
    Blender mesh builder script.  No bpy calls here — keeps the skill
    testable outside of Blender.

    Args:
        position: (x, y, z) world-space centre of the domino base.
        rotation: (pitch, roll, yaw) in degrees.
        scale: Uniform scale multiplier applied to dims.
        domino_dims: (height, width, depth) in metres.  Falls back
            to DEFAULT_DOMINO_DIMS if not provided.
        colour_hex: Hex colour string for the material.
        name: Optional Blender object name.

    Returns:
        Build-params dict with all placement data.
    """
    from kairos.skills.domino import DEFAULT_DOMINO_DIMS

    h, w, d = domino_dims or DEFAULT_DOMINO_DIMS

    return {
        "type": "domino",
        "position": list(position),
        "rotation_deg": list(rotation),
        "scale": scale,
        "dimensions": {
            "height": h * scale,
            "width": w * scale,
            "depth": d * scale,
        },
        "colour_hex": colour_hex,
        "name": name,
    }
