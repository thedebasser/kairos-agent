"""Unit tests for music selector.

Tests programmatic track selection logic.
"""

import pytest

from kairos.schemas.contracts import (
    AudioBrief,
    ConceptBrief,
    EnergyLevel,
    MusicTrackMetadata,
    ScenarioCategory,
    SimulationRequirements,
    SimulationStats,
)
from kairos.services.music_selector import select_music

pytestmark = pytest.mark.unit


def _make_track(
    track_id: str,
    mood: list[str],
    bpm: int = 120,
    duration_sec: float = 120.0,
    energy_curve: str = "building",
    contentid_status: str = "cleared",
    use_count: int = 0,
) -> MusicTrackMetadata:
    return MusicTrackMetadata(
        track_id=track_id,
        filename=f"tracks/{track_id}.mp3",
        duration_sec=duration_sec,
        bpm=bpm,
        mood=mood,
        energy_curve=energy_curve,
        contentid_status=contentid_status,
        use_count=use_count,
    )


@pytest.fixture
def sample_library() -> list[MusicTrackMetadata]:
    return [
        _make_track("upbeat_01", ["upbeat", "energetic"], bpm=120),
        _make_track("tense_01", ["tense", "building"], bpm=90, energy_curve="building"),
        _make_track("chill_01", ["chill", "ambient"], bpm=80, energy_curve="low"),
        _make_track("dramatic_01", ["dramatic", "epic"], bpm=140, energy_curve="climax"),
    ]


class TestMusicSelection:
    """Tests for music track selection."""

    def test_mood_match_preferred(self, sample_concept, sample_simulation_stats, sample_library):
        """Track with matching mood should be selected."""
        result = select_music(
            sample_concept,
            sample_simulation_stats,
            library=sample_library,
        )
        assert result is not None
        assert "upbeat" in result.mood or "energetic" in result.mood

    def test_empty_library_returns_none(self, sample_concept, sample_simulation_stats):
        """Empty library returns None."""
        result = select_music(
            sample_concept,
            sample_simulation_stats,
            library=[],
        )
        assert result is None

    def test_recently_used_deprioritised(self, sample_concept, sample_simulation_stats):
        """Heavily used tracks should be selected less often."""
        library = [
            _make_track("used_01", ["upbeat", "energetic"], use_count=50),
            _make_track("fresh_01", ["upbeat", "energetic"], use_count=0),
        ]
        result = select_music(
            sample_concept,
            sample_simulation_stats,
            library=library,
        )
        assert result is not None
        assert result.track_id == "fresh_01"

    def test_contentid_cleared_preferred(self, sample_concept, sample_simulation_stats):
        """ContentID cleared tracks should be preferred."""
        library = [
            _make_track("cleared_01", ["upbeat"], contentid_status="cleared"),
            _make_track("unknown_01", ["upbeat"], contentid_status="unknown"),
        ]
        result = select_music(
            sample_concept,
            sample_simulation_stats,
            library=library,
        )
        assert result is not None
        assert result.track_id == "cleared_01"

    def test_duration_filter(self, sample_concept, sample_simulation_stats):
        """Tracks shorter than video should still be selected (fallback)."""
        library = [
            _make_track("short_01", ["upbeat"], duration_sec=30.0),
        ]
        result = select_music(
            sample_concept,
            sample_simulation_stats,
            library=library,
        )
        # Should still return something (fallback to any cleared track)
        assert result is not None
