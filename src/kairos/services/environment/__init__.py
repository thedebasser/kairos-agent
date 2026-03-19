"""Kairos Environment Theming Service.

Reusable across all pipelines (domino, marble, ball-pit, etc.).
Downloads HDRI + ground textures from Poly Haven, manages SFX
from Freesound, and provides theme configuration for Blender scenes.
"""

from kairos.services.environment.theme_catalogue import (
    THEME_CATALOGUE,
    ThemeConfig,
    get_theme,
    list_themes,
)

__all__ = [
    "THEME_CATALOGUE",
    "ThemeConfig",
    "get_theme",
    "list_themes",
]
