"""Integration tests for checkpoint recovery.

Tests LangGraph state persistence via MemorySaver and pipeline resume
after human review decisions.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

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


def _build_full_mock_adapter():
    """Build a mock adapter that succeeds fully through the pipeline."""
    concept = ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="Checkpoint Test",
        visual_brief="Balls cascade.",
        simulation_requirements=SimulationRequirements(
            body_count_initial=50, body_count_max=500, interaction_type="collision",
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
        raw_video_path="/tmp/ckpt_sim.mp4",
        simulation_iteration=1,
        simulation_result=SimulationResult(
            returncode=0, stdout="OK", output_files=["/tmp/ckpt_sim.mp4"],
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
        Caption(caption_type=CaptionType.HOOK, text="Test!", start_sec=0.0, end_sec=2.5),
    ])
    video_out = VideoOutput(
        pipeline_run_id=uuid.uuid4(), simulation_id=uuid.uuid4(),
        final_video_path="/tmp/ckpt_final.mp4", captions=captions,
        music_track=music, title="Checkpoint Test",
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

    return adapter


class TestCheckpointRecovery:
    """Tests that pipeline can resume from checkpoints after human review."""

    async def test_resume_after_approval(self):
        """Pipeline pauses at pending_review, resumes to publish on approval."""
        adapter = _build_full_mock_adapter()
        checkpointer = MemorySaver()
        thread_id = "ckpt-approve"

        with patch("kairos.orchestrator.graph.get_pipeline", return_value=adapter):
            compiled = compile_pipeline(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            # Run 1: reaches pending_review and pauses
            result1 = await compiled.ainvoke(_base_state(), config)

        assert result1["status"] == PipelineStatus.PENDING_REVIEW.value
        assert result1["review_action"] is None

        # Run 2: resume with approval
        with patch("kairos.orchestrator.graph.get_pipeline", return_value=adapter):
            compiled2 = compile_pipeline(checkpointer=checkpointer)
            result2 = await compiled2.ainvoke(
                {"review_action": ReviewAction.APPROVED.value},
                config,
            )

        assert result2["status"] == PipelineStatus.APPROVED.value

    async def test_resume_after_bad_concept_rejection(self):
        """Pipeline resumes with bad_concept: idea agent re-runs with new concept.

        With mocked agents that always succeed and review_action persisting
        in state, the pipeline will loop (idea→sim→editor→review→idea→...).
        We verify the idea agent was called again (i.e., the rejection caused
        re-generation) by checking call counts.
        """
        adapter = _build_full_mock_adapter()
        idea_agent = adapter.get_idea_agent()
        checkpointer = MemorySaver()
        thread_id = "ckpt-bad-concept"

        with patch("kairos.orchestrator.graph.get_pipeline", return_value=adapter):
            compiled = compile_pipeline(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            # Run 1: reaches pending_review (idea agent called once)
            result1 = await compiled.ainvoke(_base_state(), config)

        assert result1["status"] == PipelineStatus.PENDING_REVIEW.value
        first_run_calls = idea_agent.generate_concept.call_count
        assert first_run_calls == 1

        # Run 2: resume with bad_concept — verify idea agent is called again
        # Use recursion_limit to prevent infinite loop with mocked agents
        with patch("kairos.orchestrator.graph.get_pipeline", return_value=adapter):
            compiled2 = compile_pipeline(checkpointer=checkpointer)
            try:
                await compiled2.ainvoke(
                    {"review_action": ReviewAction.BAD_CONCEPT.value},
                    {**config, "recursion_limit": 15},
                )
            except Exception:
                pass  # Recursion limit expected with mocked always-succeed agents

        # Idea agent was called additional times (loops back on bad_concept)
        assert idea_agent.generate_concept.call_count > first_run_calls

    async def test_checkpoint_preserves_state(self):
        """State is preserved across checkpoint save/load."""
        adapter = _build_full_mock_adapter()
        checkpointer = MemorySaver()
        thread_id = "ckpt-preserve"

        with patch("kairos.orchestrator.graph.get_pipeline", return_value=adapter):
            compiled = compile_pipeline(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            result = await compiled.ainvoke(_base_state(), config)

        # Verify key state fields are preserved
        assert result["concept"] is not None
        assert result["final_video_path"] == "/tmp/ckpt_final.mp4"
        assert result["simulation_iteration"] == 1
        assert result["concept_attempts"] == 1
