"""Integration tests for end-to-end pipeline execution.

Uses MemorySaver (in-memory checkpointer) and fully mocked agents so these
tests can run without Docker, PostgreSQL, Redis, or Ollama.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from kairos.exceptions import ConceptGenerationError, SimulationExecutionError
from kairos.schemas.contracts import (
    AudioBrief,
    Caption,
    CaptionSet,
    CaptionType,
    ConceptBrief,
    EnergyLevel,
    MusicTrackMetadata,
    PipelineState,
    PipelineStatus,
    ReviewAction,
    ScenarioCategory,
    SimulationRequirements,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
    VideoOutput,
)
from kairos.orchestrator.graph import compile_pipeline

pytestmark = [pytest.mark.integration]


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "pipeline_run_id": str(uuid.uuid4()),
        "pipeline": "physics",
        "status": PipelineStatus.RUNNING.value,
        "concept": None,
        "concept_attempts": 0,
        "simulation_code": "",
        "simulation_result": None,
        "simulation_stats": None,
        "validation_result": None,
        "simulation_iteration": 0,
        "raw_video_path": "",
        "captions": None,
        "music_track": None,
        "final_video_path": "",
        "video_output": None,
        "review_action": None,
        "review_feedback": "",
        "total_cost_usd": 0.0,
        "errors": [],
    }
    state.update(overrides)
    return state


def _build_mock_adapter() -> tuple[MagicMock, ConceptBrief, VideoOutput]:
    """Build a mock adapter that will succeed through the full pipeline."""
    concept = ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="E2E Integration Test",
        visual_brief="Balls cascade into pit.",
        simulation_requirements=SimulationRequirements(
            body_count_initial=50,
            body_count_max=500,
            interaction_type="collision",
        ),
        audio_brief=AudioBrief(mood=["upbeat"], energy_curve=EnergyLevel.BUILDING),
        hook_text="Watch this",
        novelty_score=7.5,
        feasibility_score=8.0,
    )

    sim_state = PipelineState(
        pipeline="physics",
        status=PipelineStatus.EDITING_PHASE,
        simulation_code="code",
        raw_video_path="/tmp/e2e_sim.mp4",
        simulation_iteration=1,
        simulation_result=SimulationResult(
            returncode=0, stdout="OK", output_files=["/tmp/e2e_sim.mp4"],
        ),
        simulation_stats=SimulationStats(
            duration_sec=65.0, peak_body_count=200, avg_fps=30.0,
            min_fps=28.0, payoff_timestamp_sec=48.0,
            total_frames=1950, file_size_bytes=40_000_000,
        ),
        validation_result=ValidationResult(
            passed=True, tier1_passed=True,
            checks=[ValidationCheck(name="ok", passed=True, message="OK")],
        ),
    )

    music = MusicTrackMetadata(
        track_id="test", filename="test.mp3", duration_sec=120.0,
        bpm=120, mood=["upbeat"], energy_curve="building",
        contentid_status="cleared",
    )
    captions = CaptionSet(captions=[
        Caption(caption_type=CaptionType.HOOK, text="Watch!", start_sec=0.0, end_sec=2.5),
    ])
    video_out = VideoOutput(
        pipeline_run_id=uuid.uuid4(), simulation_id=uuid.uuid4(),
        final_video_path="/tmp/e2e_final.mp4", captions=captions,
        music_track=music, title="E2E Test",
    )

    mock_idea = AsyncMock()
    mock_idea.generate_concept.return_value = concept
    mock_sim = AsyncMock()
    mock_sim.run_loop.return_value = sim_state
    mock_editor = AsyncMock()
    mock_editor.select_music.return_value = music
    mock_editor.generate_captions.return_value = captions
    mock_editor.compose_video.return_value = video_out

    adapter = MagicMock()
    adapter.get_idea_agent.return_value = mock_idea
    adapter.get_simulation_agent.return_value = mock_sim
    adapter.get_video_editor_agent.return_value = mock_editor

    return adapter, concept, video_out


class TestPipelineE2E:
    """End-to-end pipeline integration tests with mocked agents."""

    async def test_full_pipeline_reaches_pending_review(self):
        """Pipeline runs from START to pending_review then pauses."""
        adapter, concept, video_out = _build_mock_adapter()

        with patch("kairos.pipeline.graph.get_pipeline", return_value=adapter):
            checkpointer = MemorySaver()
            compiled = compile_pipeline(checkpointer=checkpointer)

            final = await compiled.ainvoke(
                _base_state(),
                {"configurable": {"thread_id": "e2e-full"}},
            )

        assert final["status"] == PipelineStatus.PENDING_REVIEW.value
        assert final["final_video_path"] == "/tmp/e2e_final.mp4"
        assert final["concept"] is not None
        assert final["review_action"] is None  # paused for human

    async def test_pipeline_with_simulation_retry(self):
        """Simulation fails once then succeeds — pipeline continues."""
        adapter, concept, video_out = _build_mock_adapter()
        sim_agent = adapter.get_simulation_agent()

        # First call fails, second succeeds
        fail_state = sim_agent.run_loop.return_value
        sim_agent.run_loop.side_effect = [
            SimulationExecutionError("First attempt failed"),
            fail_state,
        ]

        with patch("kairos.pipeline.graph.get_pipeline", return_value=adapter):
            compiled = compile_pipeline(checkpointer=MemorySaver())

            final = await compiled.ainvoke(
                _base_state(),
                {"configurable": {"thread_id": "e2e-retry"}},
            )

        assert final["status"] == PipelineStatus.PENDING_REVIEW.value
        assert sim_agent.run_loop.call_count == 2

    async def test_pipeline_with_concept_retry(self):
        """Idea agent fails once then succeeds."""
        adapter, concept, video_out = _build_mock_adapter()
        idea_agent = adapter.get_idea_agent()

        idea_agent.generate_concept.side_effect = [
            ConceptGenerationError("Bad LLM response"),
            concept,
        ]

        with patch("kairos.pipeline.graph.get_pipeline", return_value=adapter):
            compiled = compile_pipeline(checkpointer=MemorySaver())

            final = await compiled.ainvoke(
                _base_state(),
                {"configurable": {"thread_id": "e2e-concept-retry"}},
            )

        assert final["status"] == PipelineStatus.PENDING_REVIEW.value
        assert final["concept_attempts"] == 2
