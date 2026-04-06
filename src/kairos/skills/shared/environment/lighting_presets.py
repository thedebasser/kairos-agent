"""Lighting presets for the creative pipeline.

Maps preset names to Blender lighting configurations.
Indoor presets use area/point lights with no skybox.
Outdoor presets use sun lights with HDRI skybox.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LightingPreset:
    """A lighting configuration for a Blender scene."""
    name: str
    light_type: str  # "area_light", "sun_light", "point_light"
    color: tuple[float, float, float]  # RGB 0-1
    intensity: float  # Watts (area/point) or strength (sun)
    environment_type: str  # "indoor" or "outdoor"
    uses_skybox: bool
    ambient: str = "neutral"  # warm, cool, neutral, dramatic
    themes: list[str] = field(default_factory=list)


# Built-in presets matching the design doc Section 8
LIGHTING_PRESETS: dict[str, LightingPreset] = {
    "warm_indoor": LightingPreset(
        name="warm_indoor",
        light_type="area_light",
        color=(1.0, 0.9, 0.8),
        intensity=500,
        environment_type="indoor",
        uses_skybox=False,
        ambient="warm",
        themes=["dinner", "bedroom", "rustic", "living_room"],
    ),
    "cool_indoor": LightingPreset(
        name="cool_indoor",
        light_type="area_light",
        color=(0.9, 0.95, 1.0),
        intensity=600,
        environment_type="indoor",
        uses_skybox=False,
        ambient="neutral",
        themes=["kitchen", "bathroom", "modern", "office"],
    ),
    "studio": LightingPreset(
        name="studio",
        light_type="area_light",
        color=(1.0, 1.0, 1.0),
        intensity=550,
        environment_type="indoor",
        uses_skybox=False,
        ambient="neutral",
        themes=["generic", "modern", "clean"],
    ),
    "daylight_outdoor": LightingPreset(
        name="daylight_outdoor",
        light_type="sun_light",
        color=(1.0, 1.0, 0.95),
        intensity=3.0,
        environment_type="outdoor",
        uses_skybox=True,
        ambient="neutral",
        themes=["outdoor", "garden", "cityscape", "park"],
    ),
    "golden_hour": LightingPreset(
        name="golden_hour",
        light_type="sun_light",
        color=(1.0, 0.85, 0.6),
        intensity=2.0,
        environment_type="outdoor",
        uses_skybox=True,
        ambient="warm",
        themes=["outdoor", "dramatic", "beach", "desert"],
    ),
    "overcast": LightingPreset(
        name="overcast",
        light_type="sun_light",
        color=(0.85, 0.88, 0.95),
        intensity=2.5,
        environment_type="outdoor",
        uses_skybox=True,
        ambient="cool",
        themes=["outdoor", "garden", "medieval", "industrial"],
    ),
    "dramatic": LightingPreset(
        name="dramatic",
        light_type="area_light",
        color=(1.0, 0.8, 0.6),
        intensity=700,
        environment_type="indoor",
        uses_skybox=False,
        ambient="dramatic",
        themes=["dinner", "luxury", "vintage"],
    ),
}


def get_lighting_preset(name: str) -> LightingPreset | None:
    """Look up a lighting preset by name."""
    return LIGHTING_PRESETS.get(name)


def find_presets_by_theme(theme: str) -> list[LightingPreset]:
    """Find lighting presets that match a given theme tag."""
    return [p for p in LIGHTING_PRESETS.values() if theme in p.themes]


def find_presets_by_environment(environment_type: str) -> list[LightingPreset]:
    """Find all presets for indoor or outdoor."""
    return [p for p in LIGHTING_PRESETS.values() if p.environment_type == environment_type]
