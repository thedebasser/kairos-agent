"""Theme catalogue and configuration schema.

Each theme fully drives: HDRI, ground texture, domino materials,
compositor post-processing, and SFX search queries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompositorConfig:
    """Post-processing compositor settings for Blender."""
    bloom_type: str = "NONE"       # BLOOM | FOG_GLOW | STREAKS | NONE
    bloom_mix: float = 0.0         # 0.0–1.0
    color_balance_lift: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_balance_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_balance_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    vignette_strength: float = 0.0  # 0.0–1.0


@dataclass(frozen=True)
class ThemeConfig:
    """Full environment theme configuration.

    Consumed by setup_environment.py (Blender) and mix_audio.py (FFmpeg).
    """
    theme_name: str
    hdri_category: str              # Poly Haven category
    hdri_tags: list[str] = field(default_factory=list)  # preferred tags
    hdri_strength: float = 1.0      # 0.5–3.0

    # Ground texture
    ground_texture_category: str = ""
    ground_texture_tags: list[str] = field(default_factory=list)
    ground_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ground_uv_scale: float = 12.0  # UV tiling (higher = smaller texture detail)

    # Domino / object materials
    domino_palette: list[tuple[float, float, float]] = field(default_factory=list)
    domino_roughness: float = 0.4
    domino_metallic: float = 0.05

    # SFX
    sfx_search_query: str = "impact click"
    sfx_pitch_range: tuple[float, float] = (-2.0, 2.0)   # semitones
    sfx_gain_range_db: tuple[float, float] = (-3.0, 2.0)  # dB

    # Collision & ambient audio (sound-design layer)
    collision_material: str = "wood"        # wood | plastic | ceramic | metal | heavy
    ambient_preset: str = "wood"            # matches AMBIENT_PRESETS in ambient_bed.py

    # Caption styling
    caption_colour: str = "#FFFFFF"          # Primary text colour
    caption_stroke_colour: str = "#000000"  # Outline/border colour

    # Compositor
    compositor: CompositorConfig = field(default_factory=CompositorConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict for Blender scripts."""
        return {
            "theme_name": self.theme_name,
            "hdri_category": self.hdri_category,
            "hdri_tags": self.hdri_tags,
            "hdri_strength": self.hdri_strength,
            "ground_texture_category": self.ground_texture_category,
            "ground_texture_tags": self.ground_texture_tags,
            "ground_tint": list(self.ground_tint),
            "ground_uv_scale": self.ground_uv_scale,
            "domino_palette": [list(c) for c in self.domino_palette],
            "domino_roughness": self.domino_roughness,
            "domino_metallic": self.domino_metallic,
            "sfx_search_query": self.sfx_search_query,
            "sfx_pitch_range": list(self.sfx_pitch_range),
            "sfx_gain_range_db": list(self.sfx_gain_range_db),
            "collision_material": self.collision_material,
            "ambient_preset": self.ambient_preset,
            "caption_colour": self.caption_colour,
            "caption_stroke_colour": self.caption_stroke_colour,
            "compositor": {
                "bloom_type": self.compositor.bloom_type,
                "bloom_mix": self.compositor.bloom_mix,
                "color_balance_lift": list(self.compositor.color_balance_lift),
                "color_balance_gamma": list(self.compositor.color_balance_gamma),
                "color_balance_gain": list(self.compositor.color_balance_gain),
                "vignette_strength": self.compositor.vignette_strength,
            },
        }


# ---------------------------------------------------------------------------
# Built-in theme catalogue
# ---------------------------------------------------------------------------

THEME_CATALOGUE: dict[str, ThemeConfig] = {
    "deep_space": ThemeConfig(
        theme_name="deep_space",
        hdri_category="night",
        hdri_tags=["night", "stars", "dark"],
        hdri_strength=0.8,
        ground_texture_category="metal",
        ground_texture_tags=["metal_plate", "dark"],
        ground_tint=(0.15, 0.15, 0.25),
        ground_uv_scale=15.0,
        domino_palette=[
            (0.0, 1.0, 0.9), (0.6, 0.0, 1.0), (1.0, 0.0, 0.5),
            (0.0, 0.5, 1.0), (0.9, 0.0, 1.0), (0.0, 1.0, 0.5),
        ],
        domino_roughness=0.15,
        domino_metallic=0.7,
        sfx_search_query="futuristic click synthetic sci-fi",
        sfx_pitch_range=(-3.0, 3.0),
        sfx_gain_range_db=(-4.0, 1.0),
        collision_material="metal",
        ambient_preset="metal",
        caption_colour="#00FFDD",
        caption_stroke_colour="#0A0A2E",
        compositor=CompositorConfig(
            bloom_type="FOG_GLOW",
            bloom_mix=0.6,
            color_balance_lift=(0.85, 0.85, 1.12),
            color_balance_gamma=(0.9, 0.9, 1.1),
            color_balance_gain=(0.8, 0.8, 1.2),
            vignette_strength=0.7,
        ),
    ),
    "enchanted_forest": ThemeConfig(
        theme_name="enchanted_forest",
        hdri_category="nature",
        hdri_tags=["forest", "trees", "green"],
        hdri_strength=1.2,
        ground_texture_category="terrain",
        ground_texture_tags=["forest", "moss", "leaves"],
        ground_tint=(0.7, 0.85, 0.6),
        ground_uv_scale=10.0,
        domino_palette=[
            (0.2, 0.8, 0.3), (0.8, 0.6, 0.2), (0.9, 0.3, 0.1),
            (0.6, 0.9, 0.4), (0.4, 0.3, 0.15), (1.0, 0.85, 0.3),
        ],
        domino_roughness=0.55,
        domino_metallic=0.0,
        sfx_search_query="wooden click knock tap wood",
        sfx_pitch_range=(-2.0, 2.0),
        sfx_gain_range_db=(-3.0, 2.0),
        collision_material="wood",
        ambient_preset="wood",
        caption_colour="#F0E68C",
        caption_stroke_colour="#2E4A1E",
        compositor=CompositorConfig(
            bloom_type="FOG_GLOW",
            bloom_mix=0.3,
            color_balance_lift=(1.02, 1.05, 1.02),
            color_balance_gamma=(1.0, 1.05, 0.95),
            color_balance_gain=(1.1, 1.0, 0.85),
            vignette_strength=0.4,
        ),
    ),
    "golden_hour": ThemeConfig(
        theme_name="golden_hour",
        hdri_category="outdoor",
        hdri_tags=["sunset", "golden", "warm"],
        hdri_strength=1.5,
        ground_texture_category="sand",
        ground_texture_tags=["sand", "dirt", "dry"],
        ground_tint=(1.0, 0.9, 0.7),
        ground_uv_scale=8.0,
        domino_palette=[
            (1.0, 0.4, 0.2), (1.0, 0.7, 0.1), (0.9, 0.2, 0.1),
            (1.0, 0.85, 0.4), (0.8, 0.3, 0.15), (1.0, 0.6, 0.3),
        ],
        domino_roughness=0.45,
        domino_metallic=0.1,
        sfx_search_query="stone marble impact click",
        sfx_pitch_range=(-1.5, 1.5),
        sfx_gain_range_db=(-2.0, 2.0),
        collision_material="ceramic",
        ambient_preset="ceramic",
        caption_colour="#FFE4B5",
        caption_stroke_colour="#5C3A1E",
        compositor=CompositorConfig(
            bloom_type="BLOOM",
            bloom_mix=0.4,
            color_balance_lift=(1.08, 1.04, 1.01),
            color_balance_gamma=(1.05, 1.0, 0.9),
            color_balance_gain=(1.2, 1.0, 0.8),
            vignette_strength=0.5,
        ),
    ),
    "arctic_lab": ThemeConfig(
        theme_name="arctic_lab",
        hdri_category="indoor",
        hdri_tags=["studio", "white", "clean"],
        hdri_strength=1.0,
        ground_texture_category="floor",
        ground_texture_tags=["tile", "white", "concrete"],
        ground_tint=(0.92, 0.95, 1.0),
        ground_uv_scale=14.0,
        domino_palette=[
            (0.1, 0.6, 0.9), (0.9, 0.9, 0.95), (0.0, 0.8, 0.7),
            (0.15, 0.3, 0.8), (0.7, 0.7, 0.75), (0.0, 0.5, 0.9),
        ],
        domino_roughness=0.2,
        domino_metallic=0.3,
        sfx_search_query="plastic snap crisp click",
        sfx_pitch_range=(-1.0, 2.0),
        sfx_gain_range_db=(-2.0, 1.0),
        collision_material="plastic",
        ambient_preset="plastic",
        caption_colour="#E0F0FF",
        caption_stroke_colour="#1A3A5A",
        compositor=CompositorConfig(
            bloom_type="NONE",
            bloom_mix=0.0,
            color_balance_lift=(1.05, 1.06, 1.08),
            color_balance_gamma=(1.0, 1.0, 1.05),
            color_balance_gain=(0.95, 0.95, 1.0),
            vignette_strength=0.2,
        ),
    ),
    "neon_city": ThemeConfig(
        theme_name="neon_city",
        hdri_category="urban",
        hdri_tags=["city", "night", "urban"],
        hdri_strength=0.6,
        ground_texture_category="floor",
        ground_texture_tags=["concrete", "asphalt", "wet"],
        ground_tint=(0.3, 0.3, 0.35),
        ground_uv_scale=12.0,
        domino_palette=[
            (1.0, 0.08, 0.58), (0.0, 1.0, 0.5), (1.0, 0.84, 0.0),
            (0.0, 0.75, 1.0), (1.0, 0.27, 0.0), (0.5, 1.0, 0.0),
        ],
        domino_roughness=0.1,
        domino_metallic=0.8,
        sfx_search_query="metal ping resonant click",
        sfx_pitch_range=(-3.0, 3.0),
        sfx_gain_range_db=(-3.0, 2.0),
        collision_material="metal",
        ambient_preset="metal",
        caption_colour="#FF1493",
        caption_stroke_colour="#0D0D1A",
        compositor=CompositorConfig(
            bloom_type="STREAKS",
            bloom_mix=0.5,
            color_balance_lift=(1.03, 1.05, 1.07),
            color_balance_gamma=(1.05, 0.95, 1.0),
            color_balance_gain=(1.1, 0.9, 0.8),
            vignette_strength=0.8,
        ),
    ),
    "candy_land": ThemeConfig(
        theme_name="candy_land",
        hdri_category="indoor",
        hdri_tags=["studio", "bright", "colorful"],
        hdri_strength=1.3,
        ground_texture_category="floor",
        ground_texture_tags=["tiles", "white", "clean"],
        ground_tint=(1.0, 0.88, 0.92),
        ground_uv_scale=12.0,
        domino_palette=[
            (1.0, 0.7, 0.73), (0.73, 1.0, 0.79), (0.73, 0.88, 1.0),
            (1.0, 1.0, 0.73), (0.91, 0.73, 1.0), (1.0, 0.87, 0.73),
        ],
        domino_roughness=0.35,
        domino_metallic=0.0,
        sfx_search_query="plastic light tap click pop",
        sfx_pitch_range=(-1.0, 3.0),
        sfx_gain_range_db=(-2.0, 3.0),
        collision_material="plastic",
        ambient_preset="plastic",
        caption_colour="#FF69B4",
        caption_stroke_colour="#FFFFFF",
        compositor=CompositorConfig(
            bloom_type="BLOOM",
            bloom_mix=0.2,
            color_balance_lift=(1.05, 1.03, 1.06),
            color_balance_gamma=(1.1, 1.05, 1.1),
            color_balance_gain=(1.05, 1.0, 1.05),
            vignette_strength=0.1,
        ),
    ),
    "lava_world": ThemeConfig(
        theme_name="lava_world",
        hdri_category="outdoor",
        hdri_tags=["dramatic", "dark", "sunset"],
        hdri_strength=0.7,
        ground_texture_category="rock",
        ground_texture_tags=["rock", "dark", "volcanic"],
        ground_tint=(0.4, 0.2, 0.15),
        ground_uv_scale=8.0,
        domino_palette=[
            (1.0, 0.2, 0.0), (1.0, 0.5, 0.0), (0.8, 0.0, 0.0),
            (1.0, 0.8, 0.0), (0.6, 0.1, 0.0), (1.0, 0.35, 0.1),
        ],
        domino_roughness=0.3,
        domino_metallic=0.4,
        sfx_search_query="heavy thud bass impact deep",
        sfx_pitch_range=(-4.0, 1.0),
        sfx_gain_range_db=(-2.0, 3.0),
        collision_material="heavy",
        ambient_preset="heavy",
        caption_colour="#FF6B00",
        caption_stroke_colour="#1A0500",
        compositor=CompositorConfig(
            bloom_type="FOG_GLOW",
            bloom_mix=0.5,
            color_balance_lift=(1.1, 1.02, 1.0),
            color_balance_gamma=(1.1, 0.9, 0.85),
            color_balance_gain=(1.15, 0.85, 0.7),
            vignette_strength=0.6,
        ),
    ),
}


def list_themes() -> list[str]:
    """Return available theme names."""
    return list(THEME_CATALOGUE.keys())


def get_theme(name: str | None = None) -> ThemeConfig:
    """Get a theme by name, or pick a random one if name is None."""
    if name and name in THEME_CATALOGUE:
        return THEME_CATALOGUE[name]
    return random.choice(list(THEME_CATALOGUE.values()))


def pick_random_theme() -> ThemeConfig:
    """Pick a weighted-random theme (all equal weight for now)."""
    return random.choice(list(THEME_CATALOGUE.values()))
