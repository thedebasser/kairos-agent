"""Kairos Agent — Test Configuration.

Shared fixtures, database setup, mock LLM helpers, and common test utilities.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kairos.schemas.contracts import (
    AudioBrief,
    Caption,
    CaptionSet,
    CaptionType,
    ConceptBrief,
    EnergyLevel,
    MusicTrackMetadata,
    PipelineState,
    ScenarioCategory,
    SimulationRequirements,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)
from kairos.services.category_rotation import CategoryInfo


# =============================================================================
# Global autouse fixture — clear LRU caches between tests
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_llm_config_cache():
    """Clear LLM config LRU cache and response cache between tests.

    Prevents stale cached config or step results from leaking between tests.
    """
    from kairos.ai.llm import cache as _cache_mod
    from kairos.ai.llm.config import _load_raw_config

    _load_raw_config.cache_clear()
    _cache_mod._current_cache = None
    yield
    _load_raw_config.cache_clear()
    _cache_mod._current_cache = None


# =============================================================================
# Pipeline State Fixtures
# =============================================================================


@pytest.fixture
def pipeline_run_id() -> uuid.UUID:
    """Generate a consistent pipeline run ID for tests."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def pipeline_state(pipeline_run_id: uuid.UUID) -> PipelineState:
    """Create a fresh pipeline state for testing."""
    return PipelineState(pipeline_run_id=pipeline_run_id, pipeline="physics")


# =============================================================================
# Concept Fixtures
# =============================================================================


@pytest.fixture
def sample_concept() -> ConceptBrief:
    """A valid ConceptBrief for testing."""
    return ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="Collision cascade with growing balls",
        visual_brief=(
            "200 colourful balls drop into a pit. Every collision spawns "
            "a new smaller ball until the screen fills completely."
        ),
        simulation_requirements=SimulationRequirements(
            body_count_initial=50,
            body_count_max=500,
            interaction_type="collision_spawn",
        ),
        audio_brief=AudioBrief(
            mood=["upbeat", "energetic"],
            tempo_bpm_min=110,
            tempo_bpm_max=140,
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="What happens when balls multiply?",
        novelty_score=7.5,
        feasibility_score=8.0,
        target_duration_sec=65,
    )


@pytest.fixture
def sample_concept_domino() -> ConceptBrief:
    """A domino chain ConceptBrief for category rotation tests."""
    return ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="Giant domino spiral",
        visual_brief="1000 dominoes arranged in a spiral pattern topple one by one.",
        simulation_requirements=SimulationRequirements(
            body_count_initial=1000,
            body_count_max=1000,
            interaction_type="sequential_topple",
        ),
        audio_brief=AudioBrief(
            mood=["tense", "building"],
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="Will they all fall?",
        novelty_score=6.0,
        feasibility_score=9.0,
    )


# =============================================================================
# Simulation Fixtures
# =============================================================================


@pytest.fixture
def sample_simulation_stats() -> SimulationStats:
    """Valid simulation statistics."""
    return SimulationStats(
        duration_sec=65.0,
        peak_body_count=247,
        avg_fps=32.0,
        min_fps=28.0,
        payoff_timestamp_sec=48.0,
        total_frames=1950,
        file_size_bytes=45_000_000,
    )


@pytest.fixture
def sample_simulation_result() -> SimulationResult:
    """A successful simulation execution result."""
    return SimulationResult(
        returncode=0,
        stdout="Simulation complete. 1950 frames rendered.",
        stderr="",
        output_files=["/workspace/output/simulation.mp4"],
        execution_time_sec=120.5,
    )


@pytest.fixture
def sample_validation_result_pass() -> ValidationResult:
    """A passing validation result."""
    return ValidationResult(
        passed=True,
        tier1_passed=True,
        tier2_passed=None,
        checks=[
            ValidationCheck(name="valid_mp4", passed=True, message="Valid MP4 file"),
            ValidationCheck(name="resolution", passed=True, message="Resolution: 1080x1920"),
            ValidationCheck(name="fps", passed=True, message="FPS: 32.0"),
            ValidationCheck(name="duration", passed=True, message="Duration: 65.0s"),
            ValidationCheck(name="file_size", passed=True, message="File size: 45.0MB"),
            ValidationCheck(name="frame_count", passed=True, message="Frames: 1950"),
            ValidationCheck(name="audio_present", passed=True, message="Audio stream present"),
            ValidationCheck(name="frozen_frames", passed=True, message="Max consecutive: 1"),
            ValidationCheck(name="colour_valid", passed=True, message="Colour check passed"),
        ],
    )


@pytest.fixture
def sample_validation_result_fail() -> ValidationResult:
    """A failing validation result (resolution wrong)."""
    return ValidationResult(
        passed=False,
        tier1_passed=False,
        checks=[
            ValidationCheck(name="valid_mp4", passed=True, message="Valid MP4 file"),
            ValidationCheck(
                name="resolution",
                passed=False,
                message="Resolution: 1920x1080 (expected 1080x1920)",
            ),
            ValidationCheck(name="duration", passed=True, message="Duration: 65.0s"),
        ],
    )


# =============================================================================
# Video Editor Fixtures
# =============================================================================


@pytest.fixture
def sample_caption_set() -> CaptionSet:
    """A valid caption set with hook only (POC)."""
    return CaptionSet(
        captions=[
            Caption(
                caption_type=CaptionType.HOOK,
                text="What happens when balls multiply?",
                start_sec=0.0,
                end_sec=2.5,
            ),
        ]
    )


@pytest.fixture
def sample_music_track() -> MusicTrackMetadata:
    """A sample music track metadata entry."""
    return MusicTrackMetadata(
        track_id="upbeat_120bpm_01",
        filename="tracks/upbeat_120bpm_01.mp3",
        source="pixabay",
        pixabay_id=12345,
        artist="test_artist",
        license="pixabay_content_license",
        duration_sec=120.0,
        bpm=120,
        mood=["upbeat", "energetic"],
        energy_curve="building",
        genre="electronic",
        contentid_status="cleared",
        use_count=0,
    )


# =============================================================================
# Category Rotation Fixtures
# =============================================================================


@pytest.fixture
def all_categories() -> list[CategoryInfo]:
    """All four POC categories with balanced stats."""
    return [
        CategoryInfo(name="ball_pit", total_count=10, videos_last_30_days=3),
        CategoryInfo(name="marble_funnel", total_count=8, videos_last_30_days=2),
        CategoryInfo(name="domino_chain", total_count=12, videos_last_30_days=4),
        CategoryInfo(name="destruction", total_count=6, videos_last_30_days=1),
    ]


@pytest.fixture
def categories_with_last_used() -> list[CategoryInfo]:
    """Categories where ball_pit was last used."""
    return [
        CategoryInfo(
            name="ball_pit", total_count=10, videos_last_30_days=3, is_last_used=True
        ),
        CategoryInfo(name="marble_funnel", total_count=8, videos_last_30_days=2),
        CategoryInfo(name="domino_chain", total_count=12, videos_last_30_days=4),
        CategoryInfo(name="destruction", total_count=6, videos_last_30_days=1),
    ]


# =============================================================================
# Mock Helpers
# =============================================================================


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Generic mock LLM response structure."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"result": "mock response"}',
                    "role": "assistant",
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
