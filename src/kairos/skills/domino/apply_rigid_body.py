"""Apply Blender Bullet rigid-body physics to a domino.

Runs inside Blender headless — requires bpy.
"""

from __future__ import annotations

from typing import Any


def apply_rigid_body(
    *,
    mass: float = 0.02,
    friction: float = 0.8,
    restitution: float = 0.1,
    linear_damping: float = 0.1,
    angular_damping: float = 0.05,
    collision_shape: str = "BOX",
) -> dict[str, Any]:
    """Return rigid-body config params for a domino.

    Pure-data function — returns a dict consumed by the Blender mesh
    builder when it calls ``bpy.ops.rigidbody.object_add()``.

    Args:
        mass: Mass in kg (real domino ≈ 0.02 kg).
        friction: Surface friction [0–1].
        restitution: Bounciness [0–1].
        linear_damping: Linear drag [0–1].
        angular_damping: Angular drag [0–1].
        collision_shape: Bullet collision shape name.

    Returns:
        Rigid-body params dict.
    """
    return {
        "type": "ACTIVE",
        "mass": mass,
        "friction": friction,
        "restitution": restitution,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "collision_shape": collision_shape,
    }
