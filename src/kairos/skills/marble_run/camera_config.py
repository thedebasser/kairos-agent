"""Camera configuration for marble run content.

Provides camera parameters tuned for marble runs:
- Closer follow distance (marbles are smaller than dominos)
- Higher tracking influence (marbles move faster)
- Tighter DOF for dramatic close-up feel

These params are consumed by the Blender camera setup script
(generate_course.py) via the config JSON.
"""

from __future__ import annotations

from typing import Any


# Camera presets for marble runs
# Each preset returns a dict that merges into the Blender config JSON.

def marble_follow_config(
    *,
    marble_radius: float = 0.04,
    marble_count: int = 5,
    world_scale: float = 1.0,
) -> dict[str, Any]:
    """Camera config for following the lead marble.

    Closer and tighter than the domino wavefront camera because
    marbles are much smaller objects.

    Args:
        marble_radius: Marble radius in metres.
        marble_count: Number of marbles (affects framing width).
        world_scale: Blender world scale multiplier.

    Returns:
        Camera config dict to merge into Blender config.
    """
    # Follow distance scaled to marble size and world scale
    # Marbles: ~0.6m behind (dominos use 1.5m)
    follow_distance = max(0.3, 0.6 * world_scale)

    # Camera height above marble — lower for dramatic angle
    camera_height = max(0.15, 0.3 * world_scale)

    # Higher tracking influence — marbles move faster
    tracking_influence = 0.92

    # Tighter lens for close-up feel
    lens_mm = 40.0

    # Faster DOF for sharp marble focus
    aperture_fstop = 2.0

    return {
        "camera_style": "marble_follow",
        "camera_follow_distance": round(follow_distance, 3),
        "camera_height_offset": round(camera_height, 3),
        "camera_tracking_influence": tracking_influence,
        "camera_lens_mm": lens_mm,
        "camera_aperture_fstop": aperture_fstop,
    }


def marble_race_overview_config(
    *,
    world_scale: float = 1.0,
) -> dict[str, Any]:
    """Camera config for a race overview — wider shot of the full course.

    Used for multi-marble races where all marbles should be visible.

    Args:
        world_scale: Blender world scale multiplier.

    Returns:
        Camera config dict to merge into Blender config.
    """
    return {
        "camera_style": "front_static",
        "camera_lens_mm": 24.0,
        "camera_frame_padding": 1.4,
    }


# Registry of available camera presets for marble runs
MARBLE_CAMERA_PRESETS: dict[str, Any] = {
    "marble_follow": marble_follow_config,
    "race_overview": marble_race_overview_config,
}
