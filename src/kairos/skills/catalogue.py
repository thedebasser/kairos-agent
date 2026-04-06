"""Asset catalogue schema and loader.

Defines the data model for catalogued 3D assets and provides
functions to load/save the catalogue YAML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AssetSurface:
    """A navigable surface on an asset where courses can travel."""

    name: str
    surface_type: str  # "flat", "ramp", "curved"
    local_height: float  # Z offset from object origin (metres)
    bounds: list[list[float]] = field(default_factory=list)  # 2D polygon [[x,y], ...]


@dataclass
class AssetEntry:
    """A single asset in the catalogue."""

    id: str
    name: str
    source: str  # "polyhaven", "custom", "blenderkit", "sketchfab"
    file: str  # Relative path from project root, e.g. "assets/models/table/table.blend"
    license: str  # "CC0", "CC-BY", etc.
    themes: list[str] = field(default_factory=list)
    category: str = "prop"  # "furniture", "architectural", "prop", "decorative"
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (length, width, height) metres
    surfaces: list[AssetSurface] = field(default_factory=list)
    collision_type: str = "convex_hull"  # "mesh", "convex_hull", "box"
    is_static: bool = True
    description: str = ""


def load_catalogue(catalogue_path: Path) -> list[AssetEntry]:
    """Load asset catalogue from a YAML file.

    Args:
        catalogue_path: Path to the catalogue YAML.

    Returns:
        List of AssetEntry objects.
    """
    if not catalogue_path.exists():
        logger.warning("Catalogue not found: %s", catalogue_path)
        return []

    with open(catalogue_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "assets" not in data:
        return []

    entries: list[AssetEntry] = []
    for raw in data["assets"]:
        surfaces = [
            AssetSurface(
                name=s["name"],
                surface_type=s.get("type", "flat"),
                local_height=s.get("local_height", 0.0),
                bounds=s.get("bounds", []),
            )
            for s in raw.get("surfaces", [])
        ]
        dims = raw.get("dimensions", [0, 0, 0])
        entries.append(
            AssetEntry(
                id=raw["id"],
                name=raw.get("name", raw["id"]),
                source=raw.get("source", "polyhaven"),
                file=raw["file"],
                license=raw.get("license", "CC0"),
                themes=raw.get("themes", []),
                category=raw.get("category", "prop"),
                dimensions=(dims[0], dims[1], dims[2]),
                surfaces=surfaces,
                collision_type=raw.get("collision_type", "convex_hull"),
                is_static=raw.get("is_static", True),
                description=raw.get("description", ""),
            )
        )

    logger.info("Loaded %d assets from %s", len(entries), catalogue_path)
    return entries


def save_catalogue(entries: list[AssetEntry], catalogue_path: Path) -> None:
    """Save asset catalogue to a YAML file.

    Args:
        entries: List of AssetEntry objects to serialise.
        catalogue_path: Output YAML path.
    """
    assets_data: list[dict[str, Any]] = []
    for entry in entries:
        surfaces_data = [
            {
                "name": s.name,
                "type": s.surface_type,
                "local_height": s.local_height,
                "bounds": s.bounds,
            }
            for s in entry.surfaces
        ]
        assets_data.append({
            "id": entry.id,
            "name": entry.name,
            "source": entry.source,
            "file": entry.file,
            "license": entry.license,
            "themes": entry.themes,
            "category": entry.category,
            "dimensions": list(entry.dimensions),
            "surfaces": surfaces_data,
            "collision_type": entry.collision_type,
            "is_static": entry.is_static,
            "description": entry.description,
        })

    catalogue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalogue_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"assets": assets_data},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info("Saved %d assets to %s", len(entries), catalogue_path)


def find_assets_by_theme(entries: list[AssetEntry], theme: str) -> list[AssetEntry]:
    """Filter assets matching a theme tag."""
    return [e for e in entries if theme in e.themes]


def find_assets_by_category(entries: list[AssetEntry], category: str) -> list[AssetEntry]:
    """Filter assets matching a category."""
    return [e for e in entries if e.category == category]
