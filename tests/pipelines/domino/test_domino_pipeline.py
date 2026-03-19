"""End-to-end tests for the Blender domino run pipeline.

Tests the full pipeline: adapter registration → models → idea agent →
simulation agent → video editor agent.

Uses mocked Blender executor and LLM calls so tests run without
Blender or API keys installed.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.schemas.contracts import (
    AudioBrief,
    ConceptBrief,
    DominoArchetype,
    EnergyLevel,
    IdeaAgentInput,
    PipelineState,
    PipelineStatus,
    ScenarioCategory,
    SimulationRequirements,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)
from kairos.pipelines.domino.models import DominoCourseConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def domino_config() -> DominoCourseConfig:
    """A valid DominoCourseConfig for testing."""
    return DominoCourseConfig(
        seed=42,
        archetype=DominoArchetype.S_CURVE,
        title="200 Domino S-Curve — So Satisfying",
        visual_brief="Rainbow dominoes curve across the floor in a smooth S-shape.",
        domino_count=150,
        path_amplitude=1.0,
        path_cycles=1.0,
        palette="rainbow",
        camera_style="tracking",
        lighting_preset="studio",
        hook_text="Watch them all fall!",
    )


@pytest.fixture
def domino_concept(domino_config: DominoCourseConfig) -> ConceptBrief:
    """A ConceptBrief with embedded domino config."""
    config_json = domino_config.model_dump_json()
    return ConceptBrief(
        pipeline="domino",
        category=ScenarioCategory.BALL_PIT,  # compatible category
        title=domino_config.title,
        visual_brief=domino_config.visual_brief,
        simulation_requirements=SimulationRequirements(
            body_count_initial=domino_config.domino_count,
            body_count_max=domino_config.domino_count,
            interaction_type="domino_run",
            colour_palette=["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF"],
            background_colour="#2a2a3e",
            special_effects=[f"domino_config:{config_json}"],
        ),
        audio_brief=AudioBrief(
            mood=["satisfying", "building"],
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="Watch them all fall!",
        novelty_score=7.0,
        feasibility_score=9.0,
        seed=42,
    )


@pytest.fixture
def pipeline_state() -> PipelineState:
    """A fresh pipeline state for domino testing."""
    return PipelineState(
        pipeline_run_id=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        pipeline="domino",
    )


@pytest.fixture
def mock_blender_success() -> dict[str, Any]:
    """Successful Blender script execution result."""
    return {
        "returncode": 0,
        "stdout": "Generation complete",
        "stderr": "",
        "json_output": {
            "seed": 42,
            "archetype": "s_curve",
            "domino_count": 150,
            "path_length": 85.5,
            "course_center": [0.0, 0.0, 2.0],
            "course_bounds_min": [-20.0, -30.0, 0.0],
            "course_bounds_max": [20.0, 30.0, 4.0],
            "duration_frames": 1950,
            "fps": 30,
        },
    }


@pytest.fixture
def mock_validation_success() -> dict[str, Any]:
    """Successful validation result from Blender."""
    return {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "json_output": {
            "passed": True,
            "summary": "PASSED: 7/7 checks",
            "checks": [
                {"name": "ground_exists", "passed": True, "message": "Ground plane found"},
                {"name": "domino_count", "passed": True, "message": "Found 150 dominos"},
                {"name": "domino_rigid_bodies", "passed": True, "message": "150/150 have RB"},
                {"name": "trigger_exists", "passed": True, "message": "Trigger found"},
                {"name": "camera_exists", "passed": True, "message": "Camera found"},
                {"name": "rigid_body_world", "passed": True, "message": "RB world configured"},
                {"name": "frame_range", "passed": True, "message": "1950 frames"},
            ],
        },
    }


@pytest.fixture
def mock_smoke_success() -> dict[str, Any]:
    """Successful smoke test result."""
    return {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "json_output": {
            "passed": True,
            "reason": "All smoke checks passed",
            "checks": [
                {"name": "trigger_works", "passed": True, "message": "45/150 fell by frame 300"},
                {"name": "chain_propagation", "passed": True, "message": "30% fell"},
                {"name": "physics_stability", "passed": True, "message": "Physics stable"},
            ],
            "fallen_count": 45,
            "total_count": 150,
            "completion_ratio": 0.3,
        },
    }


# =============================================================================
# Model Tests
# =============================================================================

class TestDominoModels:
    """Tests for DominoCourseConfig model."""

    def test_config_creation(self, domino_config: DominoCourseConfig) -> None:
        assert domino_config.domino_count == 150
        assert domino_config.archetype == DominoArchetype.S_CURVE
        assert domino_config.spacing_ratio == 0.35

    def test_config_to_blender_config(self, domino_config: DominoCourseConfig) -> None:
        bc = domino_config.to_blender_config()
        assert bc["seed"] == 42
        assert bc["archetype"] == "s_curve"
        assert bc["domino_count"] == 150
        assert bc["domino_height"] == 0.4
        assert bc["spacing_ratio"] == 0.35
        assert "palette" in bc

    def test_config_physics_locked_defaults(self) -> None:
        """Physics parameters should have safe defaults."""
        config = DominoCourseConfig(
            seed=1,
            archetype=DominoArchetype.SPIRAL,
            title="Test",
            visual_brief="test",
        )
        assert config.domino_mass == 0.3
        assert config.domino_friction == 0.6
        assert config.domino_bounce == 0.1
        assert config.ground_friction == 0.8
        assert config.substeps_per_frame == 20

    def test_config_validation_bounds(self) -> None:
        """Domino count should be bounded."""
        with pytest.raises(Exception):
            DominoCourseConfig(
                seed=1,
                archetype=DominoArchetype.S_CURVE,
                title="Too few",
                visual_brief="test",
                domino_count=5,  # below min of 50
            )

    def test_all_archetypes_valid(self) -> None:
        """All DominoArchetype values should be usable."""
        for arch in DominoArchetype:
            config = DominoCourseConfig(
                seed=1,
                archetype=arch,
                title=f"Test {arch.value}",
                visual_brief="test",
            )
            assert config.archetype == arch


# =============================================================================
# Pipeline Adapter Tests
# =============================================================================

class TestDominoPipelineAdapter:
    """Tests for pipeline adapter registration and interface."""

    def test_adapter_registered(self) -> None:
        """Domino pipeline should be registered."""
        from kairos.orchestrator.registry import get_registry
        registry = get_registry()
        assert "domino" in registry

    def test_adapter_properties(self) -> None:
        """Adapter should have correct properties."""
        from kairos.orchestrator.registry import get_pipeline
        adapter = get_pipeline("domino")
        assert adapter.pipeline_name == "domino"
        assert adapter.engine_name == "blender"
        assert "spiral" in adapter.categories
        assert "s_curve" in adapter.categories
        assert "branching" in adapter.categories
        assert len(adapter.categories) == 5

    def test_adapter_returns_agents(self) -> None:
        """Adapter should return agent instances."""
        from kairos.orchestrator.registry import get_pipeline
        adapter = get_pipeline("domino")

        idea = adapter.get_idea_agent()
        sim = adapter.get_simulation_agent()
        editor = adapter.get_video_editor_agent()

        from kairos.pipelines.domino.idea_agent import DominoIdeaAgent
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent
        from kairos.pipelines.domino.video_editor_agent import DominoVideoEditorAgent

        assert isinstance(idea, DominoIdeaAgent)
        assert isinstance(sim, DominoSimulationAgent)
        assert isinstance(editor, DominoVideoEditorAgent)


# =============================================================================
# Idea Agent Tests
# =============================================================================

class TestDominoIdeaAgent:
    """Tests for the domino idea agent."""

    def test_config_extraction(self, domino_concept: ConceptBrief) -> None:
        """Should extract embedded config from ConceptBrief."""
        from kairos.pipelines.domino.idea_agent import extract_domino_config
        config = extract_domino_config(domino_concept)
        assert config is not None
        assert config.archetype == DominoArchetype.S_CURVE
        assert config.domino_count == 150

    def test_config_extraction_missing(self) -> None:
        """Should return None when no config embedded."""
        from kairos.pipelines.domino.idea_agent import extract_domino_config
        concept = ConceptBrief(
            pipeline="domino",
            category=ScenarioCategory.BALL_PIT,
            title="No config",
            visual_brief="test",
            simulation_requirements=SimulationRequirements(
                body_count_initial=100,
                body_count_max=100,
                interaction_type="domino_run",
                special_effects=[],
            ),
            audio_brief=AudioBrief(mood=["satisfying"]),
            hook_text="Watch!",
            novelty_score=7.0,
            feasibility_score=9.0,
        )
        result = extract_domino_config(concept)
        assert result is None

    @pytest.mark.asyncio
    async def test_idea_agent_generates_concept(
        self, domino_config: DominoCourseConfig,
    ) -> None:
        """Idea agent should generate a concept via mocked LLM."""
        from kairos.pipelines.domino.idea_agent import DominoIdeaAgent

        agent = DominoIdeaAgent(force_archetype="s_curve")
        idea_input = IdeaAgentInput(pipeline="domino")

        with patch("kairos.pipelines.domino.idea_agent.call_llm", new_callable=AsyncMock) as mock_llm, \
             patch("kairos.pipelines.domino.idea_agent.call_with_quality_fallback", new_callable=AsyncMock) as mock_qf, \
             patch("kairos.ai.llm.cache.get_cache", return_value=None):
            mock_llm.return_value = domino_config
            mock_qf.return_value = domino_config
            concept = await agent.generate_concept(idea_input)

        assert concept.pipeline == "domino"
        assert concept.title == domino_config.title
        assert "domino_config:" in concept.simulation_requirements.special_effects[0]


# =============================================================================
# Simulation Agent Tests
# =============================================================================

class TestDominoSimulationAgent:
    """Tests for the domino simulation agent."""

    @pytest.mark.asyncio
    async def test_generate_simulation(
        self,
        domino_concept: ConceptBrief,
        mock_blender_success: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """generate_simulation should call Blender and return blend path."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent

        agent = DominoSimulationAgent()

        # Create a fake blend file so the check passes
        fake_blend = tmp_path / "blender" / "domino_course.blend"
        fake_blend.parent.mkdir(parents=True, exist_ok=True)
        fake_blend.write_text("fake blend")

        with patch("kairos.pipelines.domino.simulation_agent.run_blender_script",
                    new_callable=AsyncMock) as mock_blender, \
             patch("kairos.pipelines.domino.simulation_agent.get_cache", return_value=None), \
             patch.object(agent, "_ensure_work_dir", return_value=tmp_path / "blender"):
            mock_blender.return_value = mock_blender_success
            result = await agent.generate_simulation(domino_concept)

        assert result.endswith(".blend")
        mock_blender.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_output(
        self,
        mock_validation_success: dict[str, Any],
        mock_smoke_success: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """validate_output should run validation and smoke test scripts."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent

        agent = DominoSimulationAgent()

        # Create a fake blend file
        fake_blend = tmp_path / "domino_course.blend"
        fake_blend.write_text("fake")

        call_count = 0

        async def mock_run_blender(script_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if "validate" in script_name:
                return mock_validation_success
            return mock_smoke_success

        with patch("kairos.pipelines.domino.simulation_agent.run_blender_script",
                    side_effect=mock_run_blender), \
             patch("kairos.pipelines.domino.simulation_agent.get_cache", return_value=None):
            result = await agent.validate_output(str(fake_blend))

        assert result.passed
        assert call_count == 2  # validation + smoke test
        assert len(result.checks) > 0

    @pytest.mark.asyncio
    async def test_run_loop_full(
        self,
        domino_concept: ConceptBrief,
        mock_blender_success: dict[str, Any],
        mock_validation_success: dict[str, Any],
        mock_smoke_success: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Full run_loop should generate → validate → render."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent

        agent = DominoSimulationAgent()

        fake_blend = tmp_path / "blender" / "domino_course.blend"
        fake_blend.parent.mkdir(parents=True, exist_ok=True)
        fake_blend.write_text("fake blend")

        fake_video = tmp_path / "blender" / "render.mp4"
        fake_video.write_bytes(b"\x00" * 1024)

        render_result = {
            "returncode": 0,
            "stdout": "Render complete",
            "stderr": "",
            "json_output": None,
        }

        async def mock_run_blender(script_name, **kwargs):
            if "generate" in script_name:
                return mock_blender_success
            if "validate" in script_name:
                return mock_validation_success
            if "smoke" in script_name:
                return mock_smoke_success
            if "bake" in script_name:
                return render_result
            return mock_blender_success

        with patch("kairos.pipelines.domino.simulation_agent.run_blender_script",
                    side_effect=mock_run_blender), \
             patch("kairos.pipelines.domino.simulation_agent.get_cache", return_value=None), \
             patch.object(agent, "_ensure_work_dir", return_value=tmp_path / "blender"):
            result_state = await agent.run_loop(domino_concept)

        assert result_state.raw_video_path.endswith(".mp4")
        assert result_state.simulation_stats is not None


# =============================================================================
# Caching Tests
# =============================================================================

class TestDominoCaching:
    """Tests for cache integration in the domino pipeline."""

    def test_config_is_deterministic(self, domino_config: DominoCourseConfig) -> None:
        """Same config should produce same blender config dict."""
        bc1 = domino_config.to_blender_config()
        bc2 = domino_config.to_blender_config()
        assert bc1 == bc2

    @pytest.mark.asyncio
    async def test_generate_uses_cache(
        self,
        domino_concept: ConceptBrief,
        tmp_path: Path,
    ) -> None:
        """generate_simulation should skip Blender when cache hits."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent
        from kairos.ai.llm.cache import ResponseCache

        agent = DominoSimulationAgent()

        # Set up cache with a "hit"
        fake_blend = tmp_path / "cached_course.blend"
        fake_blend.write_text("cached")

        mock_cache = MagicMock(spec=ResponseCache)
        mock_cache.get_step.return_value = {"blend_path": str(fake_blend)}

        with patch("kairos.pipelines.domino.simulation_agent.get_cache",
                    return_value=mock_cache), \
             patch("kairos.pipelines.domino.simulation_agent.run_blender_script",
                    new_callable=AsyncMock) as mock_blender:
            result = await agent.generate_simulation(domino_concept)

        # Should have used cache, not called Blender
        assert result == str(fake_blend)
        mock_blender.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_uses_cache(self, tmp_path: Path) -> None:
        """validate_output should skip scripts when cache hits."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent
        from kairos.ai.llm.cache import ResponseCache

        agent = DominoSimulationAgent()
        fake_blend = tmp_path / "domino_course.blend"
        fake_blend.write_text("fake")

        cached_validation = {
            "passed": True,
            "checks": [{"name": "ground_exists", "passed": True, "message": "ok"}],
            "tier1_passed": True,
        }

        mock_cache = MagicMock(spec=ResponseCache)
        mock_cache.get_step.return_value = cached_validation

        with patch("kairos.pipelines.domino.simulation_agent.get_cache",
                    return_value=mock_cache), \
             patch("kairos.pipelines.domino.simulation_agent.run_blender_script",
                    new_callable=AsyncMock) as mock_blender:
            result = await agent.validate_output(str(fake_blend))

        assert result.passed
        mock_blender.assert_not_called()

    @pytest.mark.asyncio
    async def test_idea_agent_uses_cache(
        self, domino_concept: ConceptBrief,
    ) -> None:
        """Idea agent should return cached concept without LLM call."""
        from kairos.pipelines.domino.idea_agent import DominoIdeaAgent
        from kairos.ai.llm.cache import ResponseCache

        agent = DominoIdeaAgent(force_archetype="s_curve")
        idea_input = IdeaAgentInput(pipeline="domino")

        mock_cache = MagicMock(spec=ResponseCache)
        mock_cache.get_step.return_value = {
            "concept": domino_concept.model_dump(mode="json"),
        }

        with patch("kairos.ai.llm.cache.get_cache",
                    return_value=mock_cache), \
             patch("kairos.pipelines.domino.idea_agent.call_llm",
                    new_callable=AsyncMock) as mock_llm:
            result = await agent.generate_concept(idea_input)

        assert result.title == domino_concept.title
        mock_llm.assert_not_called()


# =============================================================================
# DominoArchetype Enum Tests
# =============================================================================

class TestDominoArchetype:
    """Tests for the DominoArchetype enum."""

    def test_all_values(self) -> None:
        values = [a.value for a in DominoArchetype]
        assert "spiral" in values
        assert "s_curve" in values
        assert "branching" in values
        assert "word_spell" in values
        assert "cascade" in values

    def test_enum_count(self) -> None:
        assert len(DominoArchetype) == 5
