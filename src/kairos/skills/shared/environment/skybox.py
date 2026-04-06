"""Skybox / HDRI catalogue for the creative pipeline.

Scans the assets/hdris/ directory for available HDRI files and provides
lookup by theme/mood for outdoor scene environment maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HdriEntry:
    """A single HDRI environment map."""
    name: str
    path: Path
    themes: list[str] = field(default_factory=list)


# Keyword → theme inferencing for HDRI filenames
_HDRI_KEYWORD_THEMES: dict[str, list[str]] = {
    "outdoor": ["outdoor", "garden", "park"],
    "indoor": ["indoor", "studio"],
    "studio": ["indoor", "studio", "clean"],
    "sunset": ["outdoor", "golden_hour", "dramatic", "warm"],
    "sunrise": ["outdoor", "golden_hour", "warm"],
    "night": ["outdoor", "dramatic", "dark"],
    "forest": ["outdoor", "forest", "garden"],
    "city": ["outdoor", "cityscape", "urban"],
    "urban": ["outdoor", "cityscape", "urban"],
    "church": ["indoor", "medieval", "dramatic"],
    "hall": ["indoor", "medieval", "dramatic"],
    "factory": ["indoor", "industrial"],
    "warehouse": ["indoor", "industrial"],
    "garage": ["indoor", "industrial"],
    "bakery": ["indoor", "rustic", "warm"],
    "kitchen": ["indoor", "modern"],
    "room": ["indoor"],
    "garden": ["outdoor", "garden"],
    "park": ["outdoor", "park", "garden"],
    "beach": ["outdoor", "beach"],
    "desert": ["outdoor", "desert"],
    "snow": ["outdoor", "winter"],
    "abandoned": ["dramatic", "rustic", "vintage"],
    "construction": ["outdoor", "industrial"],
    "greenhouse": ["indoor", "garden"],
}


def _infer_hdri_themes(name: str) -> list[str]:
    """Infer theme tags from an HDRI filename."""
    themes: set[str] = set()
    name_lower = name.lower()
    for keyword, tags in _HDRI_KEYWORD_THEMES.items():
        if keyword in name_lower:
            themes.update(tags)
    if not themes:
        themes.add("outdoor")  # default HDRIs to outdoor
    return sorted(themes)


def scan_hdris(hdris_dir: Path) -> list[HdriEntry]:
    """Scan the HDRIs directory and build a catalogue.

    Args:
        hdris_dir: Path to assets/hdris/ containing .hdr and .exr files.

    Returns:
        List of HdriEntry entries.
    """
    entries: list[HdriEntry] = []
    if not hdris_dir.exists():
        logger.warning("HDRIs directory not found: %s", hdris_dir)
        return entries

    for f in sorted(hdris_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".hdr", ".exr"):
            continue
        entries.append(HdriEntry(
            name=f.stem,
            path=f,
            themes=_infer_hdri_themes(f.stem),
        ))

    logger.info("Scanned %d HDRIs from %s", len(entries), hdris_dir)
    return entries


def find_hdris_by_theme(entries: list[HdriEntry], theme: str) -> list[HdriEntry]:
    """Filter HDRIs matching a given theme tag."""
    return [e for e in entries if theme in e.themes]
