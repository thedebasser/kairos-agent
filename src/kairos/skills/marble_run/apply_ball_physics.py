"""Apply Blender Bullet rigid-body physics to a marble.

Pure-data function — returns a dict consumed by the Blender mesh
builder when it calls ``bpy.ops.rigidbody.object_add()``.
"""

from __future__ import annotations

from typing import Any


def apply_ball_physics(
    *,
    mass: float | None = None,
    friction: float = 0.6,
    restitution: float = 0.4,
    linear_damping: float = 0.04,
    angular_damping: float = 0.1,
    collision_shape: str = "SPHERE",
) -> dict[str, Any]:
    """Return rigid-body config params for a marble ball.

    Args:
        mass: Mass in kg. Defaults to DEFAULT_MARBLE_MASS.
        friction: Surface friction [0-1].
        restitution: Bounciness [0-1]. Marbles bounce more than dominos.
        linear_damping: Linear drag [0-1]. Low for rolling marbles.
        angular_damping: Angular drag [0-1].
        collision_shape: Bullet collision shape — always SPHERE for marbles.

    Returns:
        Rigid-body params dict.
    """
    from kairos.skills.marble_run import DEFAULT_MARBLE_MASS

    return {
        "type": "ACTIVE",
        "mass": mass if mass is not None else DEFAULT_MARBLE_MASS,
        "friction": friction,
        "restitution": restitution,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "collision_shape": collision_shape,
    }
