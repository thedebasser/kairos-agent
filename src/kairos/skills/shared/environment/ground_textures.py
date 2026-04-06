"""Ground texture catalogue for the creative pipeline.

Maps texture names to filesystem paths and theme tags.
Scans the assets/textures/ directory for available textures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTexture:
    """A single ground texture with metadata."""
    name: str
    path: Path
    themes: list[str] = field(default_factory=list)
    # Texture map files (diffuse, normal, roughness)
    diffuse: Path | None = None
    normal: Path | None = None
    roughness: Path | None = None


# Theme tag mappings based on directory name keywords
_KEYWORD_THEMES: dict[str, list[str]] = {
    "wood": ["indoor", "rustic", "dinner", "bedroom", "living_room"],
    "plank": ["indoor", "rustic", "dinner", "bedroom"],
    "parquet": ["indoor", "modern", "living_room", "bedroom"],
    "laminate": ["indoor", "modern", "office"],
    "concrete": ["outdoor", "industrial", "garage", "modern"],
    "asphalt": ["outdoor", "industrial", "cityscape"],
    "brick": ["outdoor", "rustic", "garden", "cityscape"],
    "cobblestone": ["outdoor", "rustic", "garden", "cityscape", "medieval"],
    "tile": ["indoor", "kitchen", "bathroom", "modern"],
    "marble": ["indoor", "luxury", "bathroom", "modern"],
    "granite": ["indoor", "kitchen", "modern"],
    "grass": ["outdoor", "garden", "park"],
    "dirt": ["outdoor", "garden", "forest"],
    "sand": ["outdoor", "beach", "desert"],
    "gravel": ["outdoor", "garden", "industrial"],
    "stone": ["outdoor", "medieval", "garden", "rustic"],
    "slate": ["indoor", "modern", "kitchen"],
    "carpet": ["indoor", "bedroom", "living_room"],
    "linoleum": ["indoor", "kitchen", "office"],
    "tatami": ["indoor", "japanese"],
    "pebble": ["outdoor", "garden"],
    "snow": ["outdoor", "winter"],
    "mud": ["outdoor", "forest"],
    "rock": ["outdoor", "mountain", "volcanic"],
    "metal": ["industrial", "indoor", "modern"],
    "rubber": ["indoor", "gym", "industrial"],
    "painted": ["indoor", "modern"],
    "mossy": ["outdoor", "forest", "medieval"],
    "weathered": ["outdoor", "rustic"],
    "worn": ["indoor", "rustic", "vintage"],
}


def _infer_themes(name: str) -> list[str]:
    """Infer theme tags from a texture directory name."""
    themes: set[str] = set()
    name_lower = name.lower()
    for keyword, tags in _KEYWORD_THEMES.items():
        if keyword in name_lower:
            themes.update(tags)
    if not themes:
        themes.add("generic")
    return sorted(themes)


def _find_texture_maps(texture_dir: Path) -> dict[str, Path | None]:
    """Find diffuse, normal, and roughness maps in a texture directory."""
    diffuse = normal = roughness = None
    for f in texture_dir.iterdir():
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        suffix = f.suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".exr"):
            continue
        if "diff" in name_lower or "color" in name_lower or "base_color" in name_lower:
            diffuse = f
        elif "nor" in name_lower or "normal" in name_lower:
            normal = f
        elif "rough" in name_lower:
            roughness = f
    return {"diffuse": diffuse, "normal": normal, "roughness": roughness}


def scan_ground_textures(textures_dir: Path) -> list[GroundTexture]:
    """Scan the textures directory and build a catalogue.

    Args:
        textures_dir: Path to assets/textures/ containing subdirectories.

    Returns:
        List of GroundTexture entries.
    """
    textures: list[GroundTexture] = []
    if not textures_dir.exists():
        logger.warning("Textures directory not found: %s", textures_dir)
        return textures

    for entry in sorted(textures_dir.iterdir()):
        if not entry.is_dir():
            continue
        maps = _find_texture_maps(entry)
        if maps["diffuse"] is None:
            # Skip directories with no diffuse map
            continue
        textures.append(GroundTexture(
            name=entry.name,
            path=entry,
            themes=_infer_themes(entry.name),
            diffuse=maps["diffuse"],
            normal=maps["normal"],
            roughness=maps["roughness"],
        ))

    logger.info("Scanned %d ground textures from %s", len(textures), textures_dir)
    return textures


def find_textures_by_theme(
    textures: list[GroundTexture],
    theme: str,
) -> list[GroundTexture]:
    """Filter textures matching a given theme tag."""
    return [t for t in textures if theme in t.themes]
