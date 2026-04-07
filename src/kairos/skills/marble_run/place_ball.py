"""Place a marble ball in the Blender scene.

Pure-data function — returns build params consumed by Blender scripts.
No bpy calls, testable outside Blender.
"""

from __future__ import annotations

from typing import Any


def place_ball(
    position: tuple[float, float, float],
    *,
    radius: float | None = None,
    colour_hex: str = "#4488FF",
    name: str | None = None,
    material: str = "glass",
) -> dict[str, Any]:
    """Return build-params for a single marble ball placement.

    Args:
        position: (x, y, z) world-space centre of the marble.
        radius: Marble radius in metres. Falls back to DEFAULT_MARBLE_RADIUS.
        colour_hex: Hex colour string for the material.
        name: Optional Blender object name.
        material: Material type — "glass", "metal", "clay".

    Returns:
        Build-params dict with all placement data.
    """
    from kairos.skills.marble_run import DEFAULT_MARBLE_RADIUS

    r = radius or DEFAULT_MARBLE_RADIUS

    return {
        "type": "marble",
        "position": list(position),
        "radius": r,
        "colour_hex": colour_hex,
        "material": material,
        "name": name,
    }
