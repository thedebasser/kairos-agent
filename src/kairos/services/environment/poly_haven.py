"""Poly Haven API client for HDRI and texture downloads.

All assets are CC0 — no API key required.
Downloads are cached locally in assets/hdris/ and assets/textures/.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "KairosAgent/1.0 (domino-pipeline)"}
REQUEST_TIMEOUT = 30

# Local cache directories (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
HDRI_CACHE_DIR = _PROJECT_ROOT / "assets" / "hdris"
TEXTURE_CACHE_DIR = _PROJECT_ROOT / "assets" / "textures"


def _ensure_dirs() -> None:
    HDRI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HDRI
# ---------------------------------------------------------------------------

def list_hdris(category: str | None = None) -> dict[str, Any]:
    """List available HDRIs, optionally filtered by category.

    Returns {slug: {name, categories, tags, ...}, ...}
    """
    params: dict[str, str] = {"t": "hdris"}
    if category:
        params["categories"] = category
    try:
        resp = requests.get(
            f"{BASE_URL}/assets",
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to list HDRIs (category=%s): %s", category, exc)
        return {}


def _pick_hdri_slug(
    category: str,
    tags: list[str] | None = None,
) -> str | None:
    """Pick a random HDRI slug from a category, preferring those matching tags."""
    assets = list_hdris(category)
    if not assets:
        # Fallback: try without category filter
        assets = list_hdris()
    if not assets:
        return None

    slugs = list(assets.keys())

    # Prefer slugs whose tags overlap with requested tags
    if tags:
        tag_set = set(t.lower() for t in tags)
        scored = []
        for slug in slugs:
            asset_tags = set(
                t.lower()
                for t in assets[slug].get("tags", [])
            )
            overlap = len(tag_set & asset_tags)
            scored.append((slug, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1]
        if best_score > 0:
            top = [s for s, sc in scored if sc == best_score]
            return random.choice(top)

    return random.choice(slugs)


def download_hdri(
    slug: str,
    resolution: str = "2k",
    fmt: str = "exr",
) -> Path | None:
    """Download an HDRI from Poly Haven. Returns local path or None on failure.

    Caches downloads so the same slug is only downloaded once.
    """
    _ensure_dirs()
    dest = HDRI_CACHE_DIR / f"{slug}_{resolution}.{fmt}"
    if dest.exists() and dest.stat().st_size > 1024:
        logger.info("[polyhaven] HDRI cache hit: %s", dest.name)
        return dest

    try:
        files_resp = requests.get(
            f"{BASE_URL}/files/{slug}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        files_resp.raise_for_status()
        files = files_resp.json()

        # Navigate: hdri -> resolution -> format -> url
        url = files["hdri"][resolution][fmt]["url"]
        logger.info("[polyhaven] Downloading HDRI: %s (%s %s)", slug, resolution, fmt)

        start = time.monotonic()
        dl = requests.get(url, headers=HEADERS, timeout=120)
        dl.raise_for_status()
        dest.write_bytes(dl.content)
        elapsed = time.monotonic() - start
        logger.info(
            "[polyhaven] Downloaded %s (%.1f MB, %.1fs)",
            dest.name, len(dl.content) / 1e6, elapsed,
        )
        return dest

    except Exception as exc:
        logger.warning("[polyhaven] Failed to download HDRI %s: %s", slug, exc)
        return None


def get_hdri(
    category: str,
    tags: list[str] | None = None,
    resolution: str = "2k",
) -> Path | None:
    """Pick and download an HDRI matching category/tags.

    Returns local file path or None on failure.
    """
    slug = _pick_hdri_slug(category, tags)
    if not slug:
        logger.warning("[polyhaven] No HDRI found for category=%s", category)
        return None
    return download_hdri(slug, resolution=resolution)


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

def list_textures(category: str | None = None) -> dict[str, Any]:
    """List available textures, optionally filtered by category."""
    params: dict[str, str] = {"t": "textures"}
    if category:
        params["categories"] = category
    try:
        resp = requests.get(
            f"{BASE_URL}/assets",
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to list textures (category=%s): %s", category, exc)
        return {}


def _pick_texture_slug(
    category: str,
    tags: list[str] | None = None,
) -> str | None:
    """Pick a random texture slug from a category, preferring tag matches."""
    assets = list_textures(category)
    if not assets:
        assets = list_textures()
    if not assets:
        return None

    slugs = list(assets.keys())

    if tags:
        tag_set = set(t.lower() for t in tags)
        scored = []
        for slug in slugs:
            asset_tags = set(
                t.lower()
                for t in assets[slug].get("tags", [])
            )
            overlap = len(tag_set & asset_tags)
            scored.append((slug, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1]
        if best_score > 0:
            top = [s for s, sc in scored if sc == best_score]
            return random.choice(top)

    return random.choice(slugs)


# Map from our short names to Poly Haven's top-level key names.
# Poly Haven uses mixed-case map names at the top level.
_MAP_ALIASES: dict[str, list[str]] = {
    "diff": ["Diffuse", "diffuse", "diff", "Color", "color"],
    "rough": ["Rough", "rough", "Roughness", "roughness"],
    "nor_gl": ["nor_gl", "Normal", "normal", "nor_dx"],
    "ao": ["AO", "ao", "Ambient Occlusion"],
    "disp": ["Displacement", "displacement", "disp", "height"],
}


def download_texture(
    slug: str,
    maps: tuple[str, ...] = ("diff", "rough", "nor_gl"),
    resolution: str = "1k",
    fmt: str = "jpg",
) -> dict[str, Path] | None:
    """Download texture maps from Poly Haven.

    Returns dict mapping map type to local path, e.g.:
        {"diff": Path(...), "rough": Path(...), "nor_gl": Path(...)}
    Returns None on failure.

    Poly Haven file structure: ``data[MAP_NAME][RESOLUTION][FORMAT] →
    {url, size, md5}``.  MAP_NAME is mixed-case (e.g. "Diffuse", "Rough",
    "nor_gl").
    """
    _ensure_dirs()
    slug_dir = TEXTURE_CACHE_DIR / slug
    slug_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached: dict[str, Path] = {}
    all_cached = True
    for m in maps:
        p = slug_dir / f"{slug}_{resolution}_{m}.{fmt}"
        if p.exists() and p.stat().st_size > 256:
            cached[m] = p
        else:
            all_cached = False

    if all_cached:
        logger.info("[polyhaven] Texture cache hit: %s", slug)
        return cached

    try:
        files_resp = requests.get(
            f"{BASE_URL}/files/{slug}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        files_resp.raise_for_status()
        files = files_resp.json()

        # The top-level keys are MAP names (Diffuse, Rough, nor_gl, etc.)
        # Each map key contains resolution sub-keys (1k, 2k, …) which
        # contain format sub-keys (jpg, png, exr) → {url, size, md5}.

        result: dict[str, Path] = {}
        for m in maps:
            # Resolve the actual top-level key for this map
            aliases = _MAP_ALIASES.get(m, [m])
            map_data = None
            for alias in aliases:
                if alias in files:
                    map_data = files[alias]
                    break
            if map_data is None:
                logger.debug("[polyhaven] Map '%s' not available for %s", m, slug)
                continue

            # Pick resolution (with fallback)
            res_data = map_data.get(resolution)
            if not res_data:
                for fallback_res in ("1k", "2k", "4k"):
                    if fallback_res in map_data:
                        res_data = map_data[fallback_res]
                        break
            if not res_data:
                logger.debug(
                    "[polyhaven] No resolution data for map '%s' in %s", m, slug,
                )
                continue

            # Pick format
            fmt_entry = res_data.get(fmt)
            if not fmt_entry:
                for fallback_fmt in ("jpg", "png", "exr"):
                    if fallback_fmt in res_data:
                        fmt_entry = res_data[fallback_fmt]
                        break
            if not fmt_entry or "url" not in fmt_entry:
                logger.debug(
                    "[polyhaven] No format data for map '%s' in %s", m, slug,
                )
                continue

            url = fmt_entry["url"]
            dest = slug_dir / f"{slug}_{resolution}_{m}.{fmt}"
            if not dest.exists():
                dl = requests.get(url, headers=HEADERS, timeout=60)
                dl.raise_for_status()
                dest.write_bytes(dl.content)
                logger.info("[polyhaven] Downloaded texture map: %s", dest.name)

            result[m] = dest

        if not result:
            logger.warning("[polyhaven] No texture maps downloaded for %s", slug)
            return None

        return result

    except Exception as exc:
        logger.warning("[polyhaven] Failed to download texture %s: %s", slug, exc)
        return None


def get_ground_texture(
    category: str,
    tags: list[str] | None = None,
    resolution: str = "1k",
) -> dict[str, Path] | None:
    """Pick and download ground texture maps matching category/tags.

    Returns dict of map type → local file path, or None on failure.
    """
    slug = _pick_texture_slug(category, tags)
    if not slug:
        logger.warning("[polyhaven] No texture found for category=%s", category)
        return None
    return download_texture(slug, resolution=resolution)
