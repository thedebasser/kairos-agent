"""Asset catalogue generator — Blender headless script.

Scans a models directory for .blend files, opens each one, extracts
bounding-box dimensions and mesh objects, then writes a catalogue YAML.

Run via:
    blender --background --python generate_asset_catalogue.py -- \\
        --models-dir assets/models --output assets/catalogue.yaml

Progress is logged to stdout: "Scanning [12/230] table.blend ..."
Errors per-file are logged but do not abort the full scan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

# bpy is available when running inside Blender
try:
    import bpy  # type: ignore[import-untyped]
except ImportError:
    print("ERROR: This script must be run inside Blender headless.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _extract_bounding_box(obj: Any) -> tuple[float, float, float]:
    """Get world-space bounding-box dimensions (x, y, z) for a mesh object."""
    bbox = [obj.matrix_world @ bpy.mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def _scene_bounding_box() -> tuple[float, float, float]:
    """Compute the combined bounding box of all mesh objects in the scene."""
    all_min = [float("inf")] * 3
    all_max = [float("-inf")] * 3
    found = False

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        found = True
        bbox = [obj.matrix_world @ bpy.mathutils.Vector(corner) for corner in obj.bound_box]
        for v in bbox:
            for i, val in enumerate([v.x, v.y, v.z]):
                all_min[i] = min(all_min[i], val)
                all_max[i] = max(all_max[i], val)

    if not found:
        return (0.0, 0.0, 0.0)

    return (
        round(all_max[0] - all_min[0], 4),
        round(all_max[1] - all_min[1], 4),
        round(all_max[2] - all_min[2], 4),
    )


def _identify_surfaces() -> list[dict[str, Any]]:
    """Heuristic: find approximately-flat mesh regions that could be surfaces.

    Looks for mesh objects whose name contains keywords like "top",
    "surface", "table", "shelf", "seat" — likely navigable surfaces.
    """
    surfaces: list[dict[str, Any]] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name_lower = obj.name.lower()
        # Check if this looks like a navigable surface
        surface_keywords = ["top", "surface", "table", "shelf", "seat", "platform", "board"]
        if not any(kw in name_lower for kw in surface_keywords):
            continue

        bbox = [obj.matrix_world @ bpy.mathutils.Vector(corner) for corner in obj.bound_box]
        xs = [v.x for v in bbox]
        ys = [v.y for v in bbox]
        zs = [v.z for v in bbox]
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        height = max(zs) - min(zs)

        # Surface should be relatively flat (height << width and depth)
        if height < max(width, depth) * 0.3:
            surfaces.append({
                "name": obj.name,
                "type": "flat",
                "local_height": round(max(zs), 4),
                "bounds": [
                    [round(min(xs), 4), round(min(ys), 4)],
                    [round(max(xs), 4), round(min(ys), 4)],
                    [round(max(xs), 4), round(max(ys), 4)],
                    [round(min(xs), 4), round(max(ys), 4)],
                ],
            })

    return surfaces


def _infer_category(name: str) -> str:
    """Infer asset category from the filename."""
    name_lower = name.lower()
    furniture_kw = ["table", "chair", "desk", "shelf", "bookshelf", "sofa", "couch", "bed", "cabinet"]
    architectural_kw = ["wall", "stair", "plank", "board", "block", "pillar", "column", "arch"]
    decorative_kw = ["vase", "lamp", "frame", "rug", "plant", "flower", "painting"]
    prop_kw = ["book", "plate", "cup", "bottle", "box", "bowl", "mug", "can"]

    for kw in furniture_kw:
        if kw in name_lower:
            return "furniture"
    for kw in architectural_kw:
        if kw in name_lower:
            return "architectural"
    for kw in decorative_kw:
        if kw in name_lower:
            return "decorative"
    for kw in prop_kw:
        if kw in name_lower:
            return "prop"
    return "prop"


def _infer_themes(name: str) -> list[str]:
    """Infer theme tags from the asset name."""
    tags: set[str] = set()
    name_lower = name.lower()

    theme_map = {
        "rustic": ["rustic", "wood", "plank"],
        "modern": ["modern", "glass", "chrome", "metal"],
        "kitchen": ["kitchen", "plate", "cup", "bowl", "mug"],
        "dining": ["dining", "table", "chair"],
        "office": ["office", "desk", "monitor", "keyboard"],
        "indoor": ["indoor", "room", "house"],
        "outdoor": ["outdoor", "garden", "park"],
    }

    for theme, keywords in theme_map.items():
        for kw in keywords:
            if kw in name_lower:
                tags.add(theme)
                break

    if not tags:
        tags.add("general")

    return sorted(tags)


def scan_models(models_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory tree for .blend files and extract metadata.

    Args:
        models_dir: Root directory to scan.

    Returns:
        List of asset dicts ready for YAML serialisation.
    """
    blend_files = sorted(models_dir.rglob("*.blend"))
    total = len(blend_files)
    logger.info("Found %d .blend files in %s", total, models_dir)

    assets: list[dict[str, Any]] = []

    for idx, blend_path in enumerate(blend_files, 1):
        rel_path = blend_path.relative_to(models_dir.parent.parent)
        asset_name = blend_path.stem
        logger.info("Scanning [%d/%d] %s ...", idx, total, blend_path.name)

        try:
            # Open the .blend file
            bpy.ops.wm.open_mainfile(filepath=str(blend_path))

            # Extract metadata
            dims = _scene_bounding_box()
            surfaces = _identify_surfaces()
            category = _infer_category(asset_name)
            themes = _infer_themes(asset_name)

            assets.append({
                "id": asset_name.lower().replace(" ", "_").replace("-", "_"),
                "name": asset_name.replace("_", " ").title(),
                "source": "polyhaven",
                "file": str(rel_path).replace("\\", "/"),
                "license": "CC0",
                "themes": themes,
                "category": category,
                "dimensions": [round(d, 4) for d in dims],
                "surfaces": surfaces,
                "collision_type": "convex_hull",
                "is_static": True,
                "description": f"Auto-scanned from {blend_path.name}",
            })

        except Exception:
            logger.error(
                "ERROR scanning [%d/%d] %s:\n%s",
                idx, total, blend_path.name, traceback.format_exc(),
            )

    logger.info("Scan complete: %d/%d assets catalogued successfully", len(assets), total)
    return assets


def main() -> None:
    """CLI entry point when run via ``blender --background --python``."""
    # Parse args after the "--" separator
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Generate asset catalogue YAML")
    parser.add_argument("--models-dir", required=True, help="Path to assets/models/")
    parser.add_argument("--output", required=True, help="Output YAML path")
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir).resolve()
    output_path = Path(args.output).resolve()

    if not models_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        sys.exit(1)

    assets = scan_models(models_dir)

    # Write YAML using plain dicts (no pyyaml dependency inside Blender)
    import yaml  # Blender ships with PyYAML

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"assets": assets},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info("Catalogue written to %s", output_path)

    # Also write JSON summary to stdout for the executor to parse
    print(json.dumps({
        "status": "success",
        "total_scanned": len(assets),
        "output_path": str(output_path),
    }))


if __name__ == "__main__":
    main()
