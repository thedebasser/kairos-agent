"""Unit tests for Pydantic data contracts.

Validates that all data models accept valid input and reject invalid input.
"""

import pytest
from pydantic import ValidationError

from kairos.models.contracts import (
    AudioBrief,
    Caption,
    CaptionSet,
    CaptionType,
    ConceptBrief,
    EnergyLevel,
    PipelineState,
    ScenarioCategory,
    SimulationRequirements,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)


pytestmark = pytest.mark.unit


class TestConceptBrief:
    """Tests for ConceptBrief validation."""

    def test_valid_concept(self, sample_concept):
        """Valid concept brief passes validation."""
        assert sample_concept.category == ScenarioCategory.BALL_PIT
        assert sample_concept.pipeline == "physics"
        assert 0 <= sample_concept.novelty_score <= 10

    def test_hook_text_max_length(self):
        """Hook text exceeding 50 chars is rejected."""
        with pytest.raises(ValidationError):
            ConceptBrief(
                category=ScenarioCategory.BALL_PIT,
                title="Test",
                visual_brief="Test visual",
                simulation_requirements=SimulationRequirements(
                    body_count_initial=10,
                    body_count_max=100,
                    interaction_type="collision",
                ),
                audio_brief=AudioBrief(mood=["upbeat"]),
                hook_text="A" * 51,  # Exceeds 50 char limit
                novelty_score=5.0,
                feasibility_score=5.0,
            )

    def test_hook_text_max_words(self):
        """Hook text exceeding 6 words is rejected."""
        with pytest.raises(ValidationError, match="≤6 words"):
            ConceptBrief(
                category=ScenarioCategory.BALL_PIT,
                title="Test",
                visual_brief="Test visual",
                simulation_requirements=SimulationRequirements(
                    body_count_initial=10,
                    body_count_max=100,
                    interaction_type="collision",
                ),
                audio_brief=AudioBrief(mood=["upbeat"]),
                hook_text="one two three four five six seven",
                novelty_score=5.0,
                feasibility_score=5.0,
            )

    def test_novelty_score_bounds(self):
        """Novelty score must be 0-10."""
        with pytest.raises(ValidationError):
            ConceptBrief(
                category=ScenarioCategory.BALL_PIT,
                title="Test",
                visual_brief="Test visual",
                simulation_requirements=SimulationRequirements(
                    body_count_initial=10,
                    body_count_max=100,
                    interaction_type="collision",
                ),
                audio_brief=AudioBrief(mood=["upbeat"]),
                hook_text="Test hook",
                novelty_score=11.0,  # Out of range
                feasibility_score=5.0,
            )

    def test_target_duration_range(self):
        """Target duration must be 62-68s."""
        with pytest.raises(ValidationError):
            ConceptBrief(
                category=ScenarioCategory.BALL_PIT,
                title="Test",
                visual_brief="Test visual",
                simulation_requirements=SimulationRequirements(
                    body_count_initial=10,
                    body_count_max=100,
                    interaction_type="collision",
                ),
                audio_brief=AudioBrief(mood=["upbeat"]),
                hook_text="Test hook",
                novelty_score=5.0,
                feasibility_score=5.0,
                target_duration_sec=30,  # Too short
            )

    def test_concept_is_frozen(self, sample_concept):
        """ConceptBrief should be immutable."""
        with pytest.raises(ValidationError):
            sample_concept.title = "Modified"


class TestSimulationStats:
    """Tests for SimulationStats validation."""

    def test_valid_stats(self, sample_simulation_stats):
        assert sample_simulation_stats.duration_sec == 65.0
        assert sample_simulation_stats.peak_body_count == 247

    def test_stats_frozen(self, sample_simulation_stats):
        with pytest.raises(ValidationError):
            sample_simulation_stats.duration_sec = 100.0


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_successful_result(self, sample_simulation_result):
        assert sample_simulation_result.returncode == 0
        assert len(sample_simulation_result.output_files) == 1

    def test_failed_result(self):
        result = SimulationResult(
            returncode=1,
            stderr="Error: division by zero",
        )
        assert result.returncode != 0
        assert "division by zero" in result.stderr


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_passing_result(self, sample_validation_result_pass):
        assert sample_validation_result_pass.passed is True
        assert sample_validation_result_pass.tier1_passed is True
        assert len(sample_validation_result_pass.failed_checks) == 0

    def test_failing_result(self, sample_validation_result_fail):
        assert sample_validation_result_fail.passed is False
        assert len(sample_validation_result_fail.failed_checks) == 1
        assert sample_validation_result_fail.failed_checks[0].name == "resolution"

    def test_summary(self, sample_validation_result_pass):
        assert "9/9" in sample_validation_result_pass.summary


class TestCaptionSet:
    """Tests for Caption and CaptionSet."""

    def test_valid_hook_caption(self, sample_caption_set):
        assert sample_caption_set.hook is not None
        assert sample_caption_set.hook.caption_type == CaptionType.HOOK

    def test_caption_text_max_length(self):
        with pytest.raises(ValidationError):
            Caption(
                caption_type=CaptionType.HOOK,
                text="A" * 51,
                start_sec=0.0,
                end_sec=2.0,
            )

    def test_empty_caption_set(self):
        cs = CaptionSet()
        assert cs.hook is None
        assert len(cs.captions) == 0


class TestPipelineState:
    """Tests for PipelineState (mutable)."""

    def test_default_state(self, pipeline_state):
        assert pipeline_state.pipeline == "physics"
        assert pipeline_state.simulation_iteration == 0
        assert pipeline_state.concept is None

    def test_state_is_mutable(self, pipeline_state):
        """PipelineState must be mutable (LangGraph mutates it)."""
        pipeline_state.simulation_iteration = 3
        assert pipeline_state.simulation_iteration == 3


class TestScenarioCategory:
    """Tests for scenario category enum."""

    def test_all_poc_categories_exist(self):
        assert ScenarioCategory.BALL_PIT.value == "ball_pit"
        assert ScenarioCategory.MARBLE_FUNNEL.value == "marble_funnel"
        assert ScenarioCategory.DOMINO_CHAIN.value == "domino_chain"
        assert ScenarioCategory.DESTRUCTION.value == "destruction"

    def test_four_categories(self):
        assert len(ScenarioCategory) == 4
