"""Environment orchestrator — ties together theme selection, asset download,
Blender environment setup, and SFX manifest generation.

This is the main entry point for the environment pipeline, called by
the simulation agent between course generation and bake+render.

Designed to be reusable across all pipelines (domino, marble, ball-pit).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from kairos.services.environment.theme_catalogue import (
    ThemeConfig,
    get_theme,
    pick_random_theme,
)
from kairos.services.environment.poly_haven import (
    get_ground_texture,
    get_hdri,
)
from kairos.services.environment.freesound_sfx import fetch_sfx_pool
from kairos.services.environment.sfx_pool import SFXPool
from kairos.services.environment.synthetic_sfx import generate_synthetic_pool

logger = logging.getLogger(__name__)


def prepare_environment(
    work_dir: Path,
    theme_name: str | None = None,
    tip_frames: dict[str, int] | None = None,
    fps: int = 30,
) -> dict[str, Any]:
    """Download assets and prepare environment config for Blender.

    This runs BEFORE the Blender setup_environment.py script.
    It downloads all external assets and writes a theme_config.json
    that Blender can consume.

    Args:
        work_dir: Working directory for this run (e.g. runs/<id>/blender/).
        theme_name: Theme to use, or None for random selection.
        tip_frames: Dict of domino/ball name → tip/impact frame for SFX timing.
        fps: Video frame rate.

    Returns:
        Dict with paths to theme_config.json and sfx_manifest.json.
    """
    start = time.monotonic()

    # 1. Select theme
    theme = get_theme(theme_name)
    logger.info("[env] Selected theme: %s", theme.theme_name)

    # 2. Download HDRI
    hdri_path = None
    try:
        hdri_path = get_hdri(
            theme.hdri_category,
            tags=theme.hdri_tags,
            resolution="2k",
        )
        if hdri_path:
            logger.info("[env] HDRI ready: %s", hdri_path.name)
    except Exception as exc:
        logger.warning("[env] HDRI download failed (non-fatal): %s", exc)

    # 3. Download ground texture
    ground_maps: dict[str, Path] | None = None
    try:
        if theme.ground_texture_category:
            ground_maps = get_ground_texture(
                theme.ground_texture_category,
                tags=theme.ground_texture_tags,
                resolution="1k",
            )
            if ground_maps:
                logger.info("[env] Ground texture ready: %d maps", len(ground_maps))
    except Exception as exc:
        logger.warning("[env] Ground texture download failed (non-fatal): %s", exc)

    # 4. Download SFX pool (or generate synthetic fallback)
    sfx_paths: list[Path] = []
    sfx_manifest: dict[int, str] = {}
    try:
        api_key = os.environ.get("FREESOUND_API_KEY", "")
        if api_key:
            sfx_paths = fetch_sfx_pool(
                theme.sfx_search_query,
                n=10,
                api_key=api_key,
            )
            logger.info("[env] SFX pool: %d sounds", len(sfx_paths))
        else:
            # Fallback: generate synthetic collision sounds
            logger.info("[env] No FREESOUND_API_KEY — using synthetic SFX")
            sfx_paths = generate_synthetic_pool(theme.theme_name, count=8)
            logger.info("[env] Synthetic SFX pool: %d sounds", len(sfx_paths))
    except Exception as exc:
        logger.warning("[env] SFX download failed (non-fatal): %s", exc)

    # 5. Generate SFX manifest (frame → sfx path)
    if sfx_paths and tip_frames:
        pool = SFXPool(
            sfx_paths,
            pitch_range=theme.sfx_pitch_range,
            gain_range_db=theme.sfx_gain_range_db,
        )
        sorted_tips = sorted(tip_frames.items(), key=lambda x: x[1])
        for domino_name, frame in sorted_tips:
            varied = pool.next()
            if varied:
                sfx_manifest[frame] = str(varied)
        logger.info("[env] SFX manifest: %d entries", len(sfx_manifest))

    # 6. Build theme config for Blender
    theme_dict = theme.to_dict()
    if hdri_path:
        theme_dict["hdri_path"] = str(hdri_path)
    if ground_maps:
        theme_dict["ground_texture_maps"] = {k: str(v) for k, v in ground_maps.items()}

    # Write configs to work_dir
    theme_config_path = work_dir / "theme_config.json"
    theme_config_path.write_text(
        json.dumps(theme_dict, indent=2),
        encoding="utf-8",
    )

    sfx_manifest_path = work_dir / "sfx_manifest.json"
    sfx_manifest_path.write_text(
        json.dumps({str(k): v for k, v in sfx_manifest.items()}, indent=2),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - start
    logger.info(
        "[env] Environment prepared in %.1fs: theme=%s, hdri=%s, "
        "ground=%s, sfx=%d",
        elapsed,
        theme.theme_name,
        "yes" if hdri_path else "no",
        "yes" if ground_maps else "no",
        len(sfx_manifest),
    )

    return {
        "theme_name": theme.theme_name,
        "theme_config_path": str(theme_config_path),
        "sfx_manifest_path": str(sfx_manifest_path),
        "hdri_downloaded": hdri_path is not None,
        "ground_texture_downloaded": ground_maps is not None,
        "sfx_count": len(sfx_manifest),
        "elapsed_sec": round(elapsed, 2),
    }


def prepare_environment_without_sfx(
    work_dir: Path,
    theme_name: str | None = None,
) -> dict[str, Any]:
    """Prepare environment without SFX (SFX requires post-bake tip_frames).

    Call this BEFORE bake to set up HDRI + textures + materials + compositor.
    Call generate_sfx_manifest() AFTER bake when tip_frames are available.
    """
    return prepare_environment(
        work_dir,
        theme_name=theme_name,
        tip_frames=None,
        fps=30,
    )


def generate_sfx_manifest(
    work_dir: Path,
    theme_name: str | None = None,
    tip_frames: dict[str, int] | None = None,
    fps: int = 30,
) -> Path | None:
    """Generate SFX manifest after physics bake provides tip_frames.

    Returns path to sfx_manifest.json, or None if no SFX available.
    """
    if not tip_frames:
        return None

    theme = get_theme(theme_name)
    api_key = os.environ.get("FREESOUND_API_KEY", "")

    sfx_paths: list[Path] = []
    if api_key:
        try:
            sfx_paths = fetch_sfx_pool(
                theme.sfx_search_query,
                n=10,
                api_key=api_key,
            )
        except Exception:
            pass

    # Fallback to synthetic SFX
    if not sfx_paths:
        logger.info("[env] Using synthetic SFX for manifest generation")
        sfx_paths = generate_synthetic_pool(theme.theme_name, count=8)

    if not sfx_paths:
        return None

    pool = SFXPool(
        sfx_paths,
        pitch_range=theme.sfx_pitch_range,
        gain_range_db=theme.sfx_gain_range_db,
    )
    manifest: dict[int, str] = {}
    for _, frame in sorted(tip_frames.items(), key=lambda x: x[1]):
        varied = pool.next()
        if varied:
            manifest[frame] = str(varied)

    sfx_path = work_dir / "sfx_manifest.json"
    sfx_path.write_text(
        json.dumps({str(k): v for k, v in manifest.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info("[env] SFX manifest written: %d entries -> %s", len(manifest), sfx_path)
    return sfx_path
