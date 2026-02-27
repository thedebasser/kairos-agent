"""Kairos Agent — Music Selector.

Programmatic music selection from pre-downloaded Pixabay library.
No LLM needed — tag/mood-based filtering with recency weighting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from kairos.config import get_settings
from kairos.models.contracts import ConceptBrief, MusicTrackMetadata, SimulationStats

logger = logging.getLogger(__name__)


def load_music_library(music_dir: Path | None = None) -> list[MusicTrackMetadata]:
    """Load the music library metadata from disk.

    Args:
        music_dir: Path to music directory. Defaults to settings.

    Returns:
        List of track metadata entries.
    """
    if music_dir is None:
        settings = get_settings()
        music_dir = settings.project_root / settings.music_dir

    metadata_file = music_dir / "metadata.json"
    if not metadata_file.exists():
        logger.warning("Music metadata file not found at %s", metadata_file)
        return []

    with metadata_file.open() as f:
        data = json.load(f)

    tracks: list[MusicTrackMetadata] = []
    for entry in data.get("tracks", []):
        try:
            tracks.append(MusicTrackMetadata(**entry))
        except Exception:
            logger.warning("Skipping invalid track entry: %s", entry.get("track_id", "unknown"))
    return tracks


def select_music(
    concept: ConceptBrief,
    stats: SimulationStats,
    *,
    library: list[MusicTrackMetadata] | None = None,
) -> MusicTrackMetadata | None:
    """Select a music track matching the concept and simulation stats.

    Selection logic (programmatic, no LLM):
    1. Filter by mood match
    2. Filter by minimum duration (≥ video duration)
    3. Prefer energy_curve match
    4. Deprioritise recently-used tracks
    5. Select top match

    Args:
        concept: The concept brief with audio requirements.
        stats: Simulation statistics.
        library: Pre-loaded track library (loads from disk if None).

    Returns:
        Selected track metadata, or None if no match found.
    """
    if library is None:
        library = load_music_library()

    if not library:
        logger.warning("No tracks available in music library")
        return None

    # Filter: ContentID cleared only
    cleared = [t for t in library if t.contentid_status == "cleared"]
    if not cleared:
        logger.warning("No ContentID-cleared tracks available, using full library")
        cleared = library

    # Filter: minimum duration
    min_duration = stats.duration_sec
    duration_ok = [t for t in cleared if t.duration_sec >= min_duration]
    if not duration_ok:
        # Fall back to any cleared track (will need trimming)
        duration_ok = cleared

    # Score tracks
    scored: list[tuple[MusicTrackMetadata, float]] = []
    target_moods = set(concept.audio_brief.mood)

    for track in duration_ok:
        score = 0.0
        track_moods = set(track.mood)

        # Mood overlap
        overlap = len(target_moods & track_moods)
        score += overlap * 10.0

        # Energy curve match
        if track.energy_curve == concept.audio_brief.energy_curve.value:
            score += 5.0

        # BPM range match
        if concept.audio_brief.tempo_bpm_min <= track.bpm <= concept.audio_brief.tempo_bpm_max:
            score += 3.0

        # Recency penalty (deprioritise recently-used)
        if track.use_count > 0:
            score -= track.use_count * 1.0

        scored.append((track, score))

    # Sort by score (descending)
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        selected = scored[0][0]
        logger.info("Selected music track: %s (score=%.1f)", selected.track_id, scored[0][1])
        return selected

    return None


def update_track_usage(
    track: MusicTrackMetadata,
    music_dir: Path | None = None,
) -> None:
    """Update use_count and last_used_at for a selected track in the metadata file.

    Call this after successfully composing a video with the track.
    """
    if music_dir is None:
        settings = get_settings()
        music_dir = settings.project_root / settings.music_dir

    metadata_file = music_dir / "metadata.json"
    if not metadata_file.exists():
        return

    try:
        with metadata_file.open() as f:
            data = json.load(f)

        for entry in data.get("tracks", []):
            if entry.get("track_id") == track.track_id:
                entry["use_count"] = entry.get("use_count", 0) + 1
                entry["last_used_at"] = datetime.now().isoformat()
                break

        with metadata_file.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info("Updated usage for track %s", track.track_id)
    except Exception as e:
        logger.warning("Failed to update track usage for %s: %s", track.track_id, e)
