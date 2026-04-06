"""Domino-specific skills for domino run content generation.

Default domino dimensions (metres):
    height = 0.08
    width  = 0.04
    depth  = 0.006
"""

# Default dimensions in metres (design doc §4)
DEFAULT_DOMINO_DIMS: tuple[float, float, float] = (0.08, 0.04, 0.006)  # H, W, D

from kairos.skills.domino.apply_rigid_body import apply_rigid_body
from kairos.skills.domino.place_domino import place_domino
from kairos.skills.domino.setup_trigger import setup_trigger
from kairos.skills.domino.size_gradient import compute_size_gradient
from kairos.skills.domino.spacing_calculator import compute_spacing

__all__ = [
    "DEFAULT_DOMINO_DIMS",
    "apply_rigid_body",
    "compute_size_gradient",
    "compute_spacing",
    "place_domino",
    "setup_trigger",
]
