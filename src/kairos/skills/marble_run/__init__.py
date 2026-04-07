"""Marble-run specific skills for marble course content generation.

Default marble dimensions:
    radius = 0.04m (40mm diameter)
    mass   = 0.028 kg (glass marble)

Track piece connector system:
    connector_diameter = 0.05m (inner diameter of track connector rings)
    wall_height = 0.03m (side wall height on track pieces)
"""

# Default marble dimensions
DEFAULT_MARBLE_RADIUS: float = 0.04  # metres
DEFAULT_MARBLE_MASS: float = 0.028  # kg (glass marble)

# Track connector system
TRACK_CONNECTOR_DIAMETER: float = 0.05  # metres — inner ring diameter
TRACK_WALL_HEIGHT: float = 0.03  # metres — side rail height

from kairos.skills.marble_run.place_ball import place_ball
from kairos.skills.marble_run.apply_ball_physics import apply_ball_physics
from kairos.skills.marble_run.momentum_calculator import compute_momentum_required
from kairos.skills.marble_run.pieces import TRACK_PIECES

__all__ = [
    "DEFAULT_MARBLE_MASS",
    "DEFAULT_MARBLE_RADIUS",
    "TRACK_CONNECTOR_DIAMETER",
    "TRACK_WALL_HEIGHT",
    "apply_ball_physics",
    "compute_momentum_required",
    "place_ball",
    "TRACK_PIECES",
]
