"""Freesound API client for downloading collision SFX.

Uses preview-quality MP3s (requires only an API key, no OAuth).
Downloads are cached locally in assets/sfx/.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://freesound.org/apiv2"
HEADERS = {"User-Agent": "KairosAgent/1.0 (domino-pipeline)"}
REQUEST_TIMEOUT = 20

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SFX_CACHE_DIR = _PROJECT_ROOT / "assets" / "sfx"


def _get_api_key() -> str | None:
    """Get Freesound API key from environment."""
    return os.environ.get("FREESOUND_API_KEY", "")


def _ensure_dir() -> None:
    SFX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def search_sounds(
    query: str,
    *,
    max_results: int = 15,
    min_duration: float = 0.05,
    max_duration: float = 0.8,
    min_rating: float = 3.0,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search Freesound for short impact/click sounds.

    Returns list of sound metadata dicts with id, name, previews, duration.
    """
    key = api_key or _get_api_key()
    if not key:
        logger.warning("[freesound] No FREESOUND_API_KEY set — SFX disabled")
        return []

    filter_str = (
        f"duration:[{min_duration} TO {max_duration}] "
        f"avg_rating:[{min_rating} TO 5.0]"
    )

    params = {
        "query": query,
        "filter": filter_str,
        "fields": "id,name,duration,previews,avg_rating,license,tags",
        "page_size": max_results,
        "sort": "rating_desc",
        "token": key,
    }

    try:
        resp = requests.get(
            f"{BASE_URL}/search/text/",
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        logger.info(
            "[freesound] Found %d sounds for query '%s'",
            len(results), query,
        )
        return results
    except Exception as exc:
        logger.warning("[freesound] Search failed for '%s': %s", query, exc)
        return []


def download_preview(
    sound: dict[str, Any],
) -> Path | None:
    """Download the MP3 preview of a sound. Returns local path or None."""
    _ensure_dir()

    sound_id = sound.get("id")
    if not sound_id:
        return None

    dest = SFX_CACHE_DIR / f"{sound_id}.mp3"
    if dest.exists() and dest.stat().st_size > 256:
        return dest

    previews = sound.get("previews", {})
    # Prefer HQ MP3 preview
    url = (
        previews.get("preview-hq-mp3")
        or previews.get("preview-lq-mp3")
        or previews.get("preview-hq-ogg")
    )
    if not url:
        logger.warning("[freesound] No preview URL for sound %s", sound_id)
        return None

    try:
        dl = requests.get(url, headers=HEADERS, timeout=30)
        dl.raise_for_status()
        dest.write_bytes(dl.content)
        logger.info("[freesound] Downloaded SFX: %s (%s)", dest.name, sound.get("name", ""))
        return dest
    except Exception as exc:
        logger.warning("[freesound] Download failed for sound %s: %s", sound_id, exc)
        return None


def fetch_sfx_pool(
    query: str,
    n: int = 10,
    api_key: str | None = None,
) -> list[Path]:
    """Search and download a pool of SFX matching the query.

    Returns list of local file paths (may be fewer than n if some fail).
    """
    sounds = search_sounds(query, max_results=n, api_key=api_key)
    if not sounds:
        return []

    paths: list[Path] = []
    for sound in sounds[:n]:
        p = download_preview(sound)
        if p:
            paths.append(p)

    logger.info("[freesound] SFX pool: %d/%d downloaded for '%s'", len(paths), n, query)
    return paths
