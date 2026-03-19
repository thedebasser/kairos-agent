"""Unit tests for the LangGraph Pipeline Orchestrator.

Tests:
- Graph construction and compilation
- Node functions (idea, simulation, video_editor, review, publish)
- Edge routing (conditional branching)
- State dict <-> PipelineState conversion
- End-to-end pipeline with mocked agents
- Checkpoint resume flow
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from kairos.exceptions import (
    ConceptGenerationError,
    SimulationExecutionError,
    VideoAssemblyError,
)
from kairos.orchestrator.graph import (
    MAX_CONCEPT_ATTEMPTS,
    MAX_SIMULATION_ITERATIONS,
    _dict_to_pipeline_state,
    build_pipeline_graph,
    compile_pipeline,
    human_review_node,
    idea_node,
    publish_node,
    route_after_idea,
    route_after_review,
    route_after_simulation,
    simulation_node,
    video_editor_node,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures / Helpers
# =============================================================================


def _make_concept_dict() -> dict[str, Any]:
    """Create a valid concept brief as a dict (as stored in LangGraph state)."""
    concept = ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="Test Collision Cascade",
        visual_brief="200 balls cascade into a pit.",
        simulation_requirements=SimulationRequirements(
            body_count_initial=50,
            body_count_max=500,
            interaction_type="collision_spawn",
        ),
        audio_brief=AudioBrief(
            mood=["upbeat", "energetic"],
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="Watch them all collide",
        novelty_score=7.5,
        feasibility_score=8.0,
        target_duration_sec=65,
    )
    return concept.model_dump(mode="json")


def _make_stats_dict() -> dict[str, Any]:
    """Create simulation stats as a dict."""
    return SimulationStats(
        duration_sec=65.0,
        peak_body_count=247,
        avg_fps=32.0,
        min_fps=28.0,
        payoff_timestamp_sec=48.0,
        total_frames=1950,
        file_size_bytes=45_000_000,
    ).model_dump(mode="json")


def _make_validation_dict(*, passed: bool = True) -> dict[str, Any]:
    """Create a validation result dict."""
    return ValidationResult(
        passed=passed,
        tier1_passed=passed,
        checks=[
            ValidationCheck(name="duration", passed=passed, message="OK"),
        ],
    ).model_dump(mode="json")


def _make_base_state(**overrides: Any) -> dict[str, Any]:
    """Create a base pipeline state dict with reasonable defaults."""
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


# =============================================================================
# Graph Construction Tests
# =============================================================================


class TestGraphConstruction:
    """Tests that the graph is correctly constructed."""

    def test_build_graph_returns_state_graph(self):
        graph = build_pipeline_graph()
        assert graph is not None

    def test_compile_pipeline_returns_compiled(self):
        compiled = compile_pipeline()
        assert compiled is not None

    def test_graph_has_all_nodes(self):
        graph = build_pipeline_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "idea_agent",
            "simulation_agent",
            "video_editor_agent",
            "human_review",
            "publish_queue",
        }
        assert expected.issubset(node_names)


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouteAfterIdea:
    """Tests for idea agent routing logic."""

    def test_routes_to_simulation_on_concept(self):
        state = _make_base_state(concept=_make_concept_dict())
        assert route_after_idea(state) == "simulation_agent"

    def test_routes_to_end_on_max_attempts(self):
        state = _make_base_state(concept=None, concept_attempts=MAX_CONCEPT_ATTEMPTS)
        assert route_after_idea(state) == "__end__"

    def test_routes_to_retry_on_failure(self):
        state = _make_base_state(concept=None, concept_attempts=1)
        assert route_after_idea(state) == "idea_agent"

    def test_routes_to_retry_at_zero_attempts(self):
        state = _make_base_state(concept=None, concept_attempts=0)
        assert route_after_idea(state) == "idea_agent"

    def test_routes_to_end_on_exceeded_attempts(self):
        state = _make_base_state(concept=None, concept_attempts=MAX_CONCEPT_ATTEMPTS + 2)
        assert route_after_idea(state) == "__end__"


class TestRouteAfterSimulation:
    """Tests for simulation agent routing logic."""

    def test_routes_to_editor_on_success(self):
        state = _make_base_state(
            raw_video_path="/tmp/sim.mp4",
            validation_result=_make_validation_dict(passed=True),
        )
        assert route_after_simulation(state) == "video_editor_agent"

    def test_routes_to_idea_on_max_iterations(self):
        state = _make_base_state(
            raw_video_path="",
            simulation_iteration=MAX_SIMULATION_ITERATIONS,
        )
        assert route_after_simulation(state) == "idea_agent"

    def test_routes_to_retry_on_failure(self):
        state = _make_base_state(
            raw_video_path="",
            simulation_iteration=2,
        )
        assert route_after_simulation(state) == "simulation_agent"

    def test_routes_to_retry_on_failed_validation(self):
        state = _make_base_state(
            raw_video_path="/tmp/sim.mp4",
            validation_result=_make_validation_dict(passed=False),
            simulation_iteration=1,
        )
        assert route_after_simulation(state) == "simulation_agent"

    def test_routes_to_retry_with_no_video(self):
        state = _make_base_state(
            raw_video_path="",
            simulation_iteration=1,
        )
        assert route_after_simulation(state) == "simulation_agent"


class TestRouteAfterReview:
    """Tests for human review routing logic."""

    def test_routes_to_publish_on_approved(self):
        state = _make_base_state(review_action=ReviewAction.APPROVED.value)
        assert route_after_review(state) == "publish_queue"

    def test_routes_to_idea_on_bad_concept(self):
        state = _make_base_state(review_action=ReviewAction.BAD_CONCEPT.value)
        assert route_after_review(state) == "idea_agent"

    def test_routes_to_simulation_on_bad_sim(self):
        state = _make_base_state(review_action=ReviewAction.BAD_SIMULATION.value)
        assert route_after_review(state) == "simulation_agent"

    def test_routes_to_editor_on_bad_edit(self):
        state = _make_base_state(review_action=ReviewAction.BAD_EDIT.value)
        assert route_after_review(state) == "video_editor_agent"

    def test_routes_to_editor_on_reedit(self):
        state = _make_base_state(review_action=ReviewAction.REQUEST_REEDIT.value)
        assert route_after_review(state) == "video_editor_agent"

    def test_routes_to_end_on_no_action(self):
        """Pipeline pauses when no review action is set (waiting for human)."""
        state = _make_base_state(review_action=None)
        assert route_after_review(state) == "__end__"

    def test_routes_to_end_on_unknown_action(self):
        """Unknown review action fails safe to __end__."""
        state = _make_base_state(review_action="some_unknown_action")
        assert route_after_review(state) == "__end__"


# =============================================================================
# Node Function Tests
# =============================================================================


class TestIdeaNode:
    """Tests for the idea_agent node function."""

    async def test_success_returns_concept(self):
        concept = ConceptBrief(
            pipeline="physics",
            category=ScenarioCategory.BALL_PIT,
            title="Test Concept",
            visual_brief="Balls everywhere.",
            simulation_requirements=SimulationRequirements(
                body_count_initial=50,
                body_count_max=500,
                interaction_type="collision_spawn",
            ),
            audio_brief=AudioBrief(mood=["upbeat"], energy_curve=EnergyLevel.BUILDING),
            hook_text="Watch them all collide",
            novelty_score=7.5,
            feasibility_score=8.0,
        )

        mock_agent = AsyncMock()
        mock_agent.generate_concept.return_value = concept
        mock_adapter = MagicMock()
        mock_adapter.get_idea_agent.return_value = mock_agent

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await idea_node(_make_base_state())

        assert result["concept"] is not None
        assert result["status"] == PipelineStatus.SIMULATION_PHASE.value
        assert result["concept_attempts"] == 1

    async def test_failure_increments_attempts(self):
        mock_agent = AsyncMock()
        mock_agent.generate_concept.side_effect = ConceptGenerationError("LLM failed")
        mock_adapter = MagicMock()
        mock_adapter.get_idea_agent.return_value = mock_agent

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await idea_node(_make_base_state(concept_attempts=1))

        assert result["concept_attempts"] == 2
        assert len(result["errors"]) > 0

    async def test_unexpected_error_recorded(self):
        mock_agent = AsyncMock()
        mock_agent.generate_concept.side_effect = RuntimeError("Something broke")
        mock_adapter = MagicMock()
        mock_adapter.get_idea_agent.return_value = mock_agent

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await idea_node(_make_base_state())

        assert result["concept_attempts"] == 1
        assert any("Unexpected" in e for e in result["errors"])


class TestSimulationNode:
    """Tests for the simulation_agent node function."""

    async def test_success_returns_video_path(self):
        updated_state = PipelineState(
            pipeline="physics",
            status=PipelineStatus.EDITING_PHASE,
            simulation_code="import pygame; ...",
            raw_video_path="/tmp/output.mp4",
            simulation_iteration=2,
            simulation_result=SimulationResult(
                returncode=0,
                stdout="Complete",
                output_files=["/tmp/output.mp4"],
            ),
            simulation_stats=SimulationStats(
                duration_sec=65.0,
                peak_body_count=200,
                avg_fps=30.0,
                min_fps=28.0,
                payoff_timestamp_sec=48.0,
                total_frames=1950,
                file_size_bytes=40_000_000,
            ),
            validation_result=ValidationResult(
                passed=True,
                tier1_passed=True,
                checks=[ValidationCheck(name="duration", passed=True, message="OK")],
            ),
        )

        mock_agent = AsyncMock()
        mock_agent.run_loop.return_value = updated_state
        mock_adapter = MagicMock()
        mock_adapter.get_simulation_agent.return_value = mock_agent

        state = _make_base_state(concept=_make_concept_dict())

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await simulation_node(state)

        assert result["status"] == PipelineStatus.EDITING_PHASE.value
        assert result["raw_video_path"] == "/tmp/output.mp4"
        assert result["simulation_iteration"] == 2

    async def test_failure_increments_iteration(self):
        mock_agent = AsyncMock()
        mock_agent.run_loop.side_effect = SimulationExecutionError("Sandbox timeout")
        mock_adapter = MagicMock()
        mock_adapter.get_simulation_agent.return_value = mock_agent

        state = _make_base_state(
            concept=_make_concept_dict(),
            simulation_iteration=1,
        )

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await simulation_node(state)

        assert result["simulation_iteration"] == 2
        assert len(result["errors"]) > 0

    async def test_unexpected_error_increments_iteration(self):
        mock_agent = AsyncMock()
        mock_agent.run_loop.side_effect = RuntimeError("Unexpected")
        mock_adapter = MagicMock()
        mock_adapter.get_simulation_agent.return_value = mock_agent

        state = _make_base_state(concept=_make_concept_dict(), simulation_iteration=0)

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await simulation_node(state)

        assert result["simulation_iteration"] == 1
        assert any("Unexpected" in e for e in result["errors"])


class TestVideoEditorNode:
    """Tests for the video_editor_agent node function."""

    async def test_success_returns_video_output(self):
        music = MusicTrackMetadata(
            track_id="test_track",
            filename="test.mp3",
            duration_sec=120.0,
            bpm=120,
            mood=["upbeat"],
            energy_curve="building",
            contentid_status="cleared",
        )
        captions = CaptionSet(
            captions=[
                Caption(
                    caption_type=CaptionType.HOOK,
                    text="Test hook",
                    start_sec=0.0,
                    end_sec=2.5,
                )
            ]
        )
        video_output = VideoOutput(
            pipeline_run_id=uuid.uuid4(),
            simulation_id=uuid.uuid4(),
            final_video_path="/tmp/final.mp4",
            captions=captions,
            music_track=music,
            title="Test Video",
        )

        mock_agent = AsyncMock()
        mock_agent.select_music.return_value = music
        mock_agent.generate_captions.return_value = captions
        mock_agent.compose_video.return_value = video_output
        mock_adapter = MagicMock()
        mock_adapter.get_video_editor_agent.return_value = mock_agent

        state = _make_base_state(
            concept=_make_concept_dict(),
            simulation_stats=_make_stats_dict(),
            raw_video_path="/tmp/raw.mp4",
        )

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await video_editor_node(state)

        assert result["status"] == PipelineStatus.PENDING_REVIEW.value
        assert result["final_video_path"] == "/tmp/final.mp4"
        assert result["video_output"] is not None

    async def test_fails_without_concept(self):
        state = _make_base_state(concept=None)

        with patch("kairos.pipeline.graph.get_pipeline", return_value=MagicMock()):
            result = await video_editor_node(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert any("No concept" in e for e in result["errors"])

    async def test_video_assembly_error_sets_failed(self):
        mock_agent = AsyncMock()
        mock_agent.select_music.side_effect = VideoAssemblyError("No tracks")
        mock_adapter = MagicMock()
        mock_adapter.get_video_editor_agent.return_value = mock_agent

        state = _make_base_state(
            concept=_make_concept_dict(),
            simulation_stats=_make_stats_dict(),
        )

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await video_editor_node(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert len(result["errors"]) > 0

    async def test_unexpected_error_sets_failed(self):
        mock_agent = AsyncMock()
        mock_agent.select_music.side_effect = RuntimeError("ffmpeg exploded")
        mock_adapter = MagicMock()
        mock_adapter.get_video_editor_agent.return_value = mock_agent

        state = _make_base_state(
            concept=_make_concept_dict(),
            simulation_stats=_make_stats_dict(),
        )

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            result = await video_editor_node(state)

        assert result["status"] == PipelineStatus.FAILED.value


class TestHumanReviewNode:
    """Tests for the human review gate node."""

    async def test_sets_pending_review_status(self):
        state = _make_base_state()
        result = await human_review_node(state)
        assert result["status"] == PipelineStatus.PENDING_REVIEW.value

    async def test_preserves_pending_with_review_action(self):
        state = _make_base_state(review_action=ReviewAction.APPROVED.value)
        result = await human_review_node(state)
        assert result["status"] == PipelineStatus.PENDING_REVIEW.value


class TestPublishNode:
    """Tests for the publish_queue node function."""

    async def test_sets_approved_status(self):
        state = _make_base_state()
        result = await publish_node(state)
        assert result["status"] == PipelineStatus.APPROVED.value


# =============================================================================
# State Conversion Tests
# =============================================================================


class TestDictToPipelineState:
    """Tests for converting LangGraph dict state to PipelineState."""

    def test_basic_conversion(self):
        state = _make_base_state(pipeline="physics")
        ps = _dict_to_pipeline_state(state)
        assert ps.pipeline == "physics"
        assert ps.status == PipelineStatus.RUNNING

    def test_concept_reconstruction(self):
        state = _make_base_state(concept=_make_concept_dict())
        ps = _dict_to_pipeline_state(state)
        assert ps.concept is not None
        assert ps.concept.title == "Test Collision Cascade"
        assert ps.concept.category == ScenarioCategory.BALL_PIT

    def test_concept_object_passthrough(self):
        """ConceptBrief objects (not dicts) are passed through."""
        concept = ConceptBrief(
            pipeline="physics",
            category=ScenarioCategory.BALL_PIT,
            title="Direct Object",
            visual_brief="Test",
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
        state = _make_base_state(concept=concept)
        ps = _dict_to_pipeline_state(state)
        assert ps.concept is not None
        assert ps.concept.title == "Direct Object"

    def test_stats_reconstruction(self):
        state = _make_base_state(simulation_stats=_make_stats_dict())
        ps = _dict_to_pipeline_state(state)
        assert ps.simulation_stats is not None
        assert ps.simulation_stats.duration_sec == 65.0

    def test_validation_reconstruction(self):
        state = _make_base_state(validation_result=_make_validation_dict())
        ps = _dict_to_pipeline_state(state)
        assert ps.validation_result is not None
        assert ps.validation_result.passed is True

    def test_none_fields_handled(self):
        state = _make_base_state(concept=None, simulation_stats=None)
        ps = _dict_to_pipeline_state(state)
        assert ps.concept is None
        assert ps.simulation_stats is None

    def test_errors_preserved(self):
        state = _make_base_state(errors=["error1", "error2"])
        ps = _dict_to_pipeline_state(state)
        assert ps.errors == ["error1", "error2"]

    def test_status_enum_conversion(self):
        state = _make_base_state(status=PipelineStatus.EDITING_PHASE.value)
        ps = _dict_to_pipeline_state(state)
        assert ps.status == PipelineStatus.EDITING_PHASE


# =============================================================================
# End-to-End Pipeline Test (Mocked Agents)
# =============================================================================


class TestPipelineE2EMocked:
    """Full pipeline run with all agents mocked."""

    async def test_full_pipeline_to_pending_review(self):
        """Pipeline runs from concept through to pending_review (pauses for human)."""
        concept = ConceptBrief(
            pipeline="physics",
            category=ScenarioCategory.BALL_PIT,
            title="E2E Test Concept",
            visual_brief="Balls cascade.",
            simulation_requirements=SimulationRequirements(
                body_count_initial=50,
                body_count_max=500,
                interaction_type="collision",
            ),
            audio_brief=AudioBrief(mood=["upbeat"], energy_curve=EnergyLevel.BUILDING),
            hook_text="Watch them all collide",
            novelty_score=7.5,
            feasibility_score=8.0,
        )

        sim_state = PipelineState(
            pipeline="physics",
            status=PipelineStatus.EDITING_PHASE,
            simulation_code="code",
            raw_video_path="/tmp/sim.mp4",
            simulation_iteration=1,
            simulation_result=SimulationResult(
                returncode=0,
                stdout="OK",
                output_files=["/tmp/sim.mp4"],
            ),
            simulation_stats=SimulationStats(
                duration_sec=65.0,
                peak_body_count=200,
                avg_fps=30.0,
                min_fps=28.0,
                payoff_timestamp_sec=48.0,
                total_frames=1950,
                file_size_bytes=40_000_000,
            ),
            validation_result=ValidationResult(
                passed=True,
                tier1_passed=True,
                checks=[ValidationCheck(name="ok", passed=True, message="OK")],
            ),
        )

        music = MusicTrackMetadata(
            track_id="test",
            filename="test.mp3",
            duration_sec=120.0,
            bpm=120,
            mood=["upbeat"],
            energy_curve="building",
            contentid_status="cleared",
        )
        captions = CaptionSet(
            captions=[
                Caption(
                    caption_type=CaptionType.HOOK,
                    text="Test",
                    start_sec=0.0,
                    end_sec=2.5,
                ),
            ]
        )
        video_out = VideoOutput(
            pipeline_run_id=uuid.uuid4(),
            simulation_id=uuid.uuid4(),
            final_video_path="/tmp/final.mp4",
            captions=captions,
            music_track=music,
            title="Test Video",
        )

        mock_idea = AsyncMock()
        mock_idea.generate_concept.return_value = concept
        mock_sim = AsyncMock()
        mock_sim.run_loop.return_value = sim_state
        mock_editor = AsyncMock()
        mock_editor.select_music.return_value = music
        mock_editor.generate_captions.return_value = captions
        mock_editor.compose_video.return_value = video_out

        mock_adapter = MagicMock()
        mock_adapter.get_idea_agent.return_value = mock_idea
        mock_adapter.get_simulation_agent.return_value = mock_sim
        mock_adapter.get_video_editor_agent.return_value = mock_editor

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            compiled = compile_pipeline(checkpointer=checkpointer)

            initial_state = _make_base_state()
            config = {"configurable": {"thread_id": "test-e2e"}}

            await compiled.ainvoke(initial_state, config)

            # Use get_state to retrieve the full merged state
            snapshot = compiled.get_state(config)
            full_state = snapshot.values

        # Pipeline reaches pending_review then pauses (review_action=None -> __end__)
        assert full_state["status"] == PipelineStatus.PENDING_REVIEW.value
        assert full_state["final_video_path"] == "/tmp/final.mp4"
        assert full_state["concept"] is not None

    async def test_concept_failure_exhausts_retries(self):
        """Pipeline fails after MAX_CONCEPT_ATTEMPTS failed concept generations."""
        mock_idea = AsyncMock()
        mock_idea.generate_concept.side_effect = ConceptGenerationError("Failed")
        mock_adapter = MagicMock()
        mock_adapter.get_idea_agent.return_value = mock_idea

        with patch("kairos.pipeline.graph.get_pipeline", return_value=mock_adapter):
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            compiled = compile_pipeline(checkpointer=checkpointer)
            initial_state = _make_base_state()
            config = {"configurable": {"thread_id": "test-fail"}}

            await compiled.ainvoke(initial_state, config)

            snapshot = compiled.get_state(config)
            full_state = snapshot.values

        assert full_state.get("concept") is None
        assert full_state["concept_attempts"] >= MAX_CONCEPT_ATTEMPTS
