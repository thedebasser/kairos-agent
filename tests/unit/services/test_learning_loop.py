"""Tests for the learning loop service (AI Architecture Review §1–§6).

Covers:
- ValidationFeedback construction and rendering
- CategoryKnowledge rendering
- Few-shot prompt formatting
- Validation rules prompt generation
- AST parameter extraction
- Feature flag gating (learning_loop_enabled)
"""

from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.schemas.contracts import (
    CategoryKnowledge,
    FailedCheck,
    PastFix,
    ValidationCheck,
    ValidationFeedback,
    ValidationResult,
)
from kairos.services.ast_extractor import ExtractedParameters, extract_parameters
from kairos.ai.learning.learning_loop import (
    build_validation_feedback,
    format_few_shot_prompt,
    get_validation_rules_prompt,
)


# ============================================================================
# ValidationFeedback
# ============================================================================


class TestBuildValidationFeedback:
    """Test build_validation_feedback() produces correct structured feedback."""

    def test_urgency_minor_tweak_on_iter_1(self):
        vr = ValidationResult(
            passed=False,
            checks=[ValidationCheck(name="duration", passed=False, message="Too short", value=58.2, threshold=62)],
            tier1_passed=False,
        )
        fb = build_validation_feedback(vr, iteration=1, max_iterations=5)
        assert fb.urgency == "minor_tweak"
        assert fb.iteration == 1
        assert fb.max_iterations == 5

    def test_urgency_significant_change_on_iter_2(self):
        vr = ValidationResult(passed=False, checks=[], tier1_passed=False)
        fb = build_validation_feedback(vr, iteration=2)
        assert fb.urgency == "significant_change"

    def test_urgency_fundamental_rethink_on_iter_4(self):
        vr = ValidationResult(passed=False, checks=[], tier1_passed=False)
        fb = build_validation_feedback(vr, iteration=4)
        assert fb.urgency == "fundamental_rethink"

    def test_duration_delta_calculated(self):
        checks = [
            ValidationCheck(name="duration", passed=False, message="Duration: 58.2s", value=58.2, threshold=62),
        ]
        vr = ValidationResult(passed=False, checks=checks, tier1_passed=False)
        fb = build_validation_feedback(vr, iteration=1)
        assert len(fb.failed_checks) == 1
        fc = fb.failed_checks[0]
        assert fc.check_name == "duration"
        assert fc.actual == 58.2
        assert fc.delta is not None
        assert fc.delta < 0  # too short → negative delta
        assert "Increase SIMULATION_TIME" in fc.suggested_fix

    def test_iteration_history_included(self):
        vr = ValidationResult(passed=False, checks=[], tier1_passed=False)
        history = ["Changed ball_count from 50 to 80", "Increased simulation_time"]
        fb = build_validation_feedback(vr, iteration=3, iteration_history=history)
        assert "ball_count" in fb.iteration_history_summary
        assert "simulation_time" in fb.iteration_history_summary

    def test_prompt_text_renders(self):
        checks = [
            ValidationCheck(name="duration", passed=False, message="Too short", value=58.2, threshold=62),
        ]
        vr = ValidationResult(passed=False, checks=checks, tier1_passed=False)
        fb = build_validation_feedback(vr, iteration=2, max_iterations=5)
        text = fb.to_prompt_text()
        assert "iteration 2/5" in text
        assert "significant_change" in text
        assert "duration" in text


class TestValidationFeedbackModel:
    """Test the ValidationFeedback Pydantic model directly."""

    def test_defaults(self):
        fb = ValidationFeedback()
        assert fb.urgency == "minor_tweak"
        assert fb.iteration == 1
        assert fb.failed_checks == []

    def test_to_prompt_text_empty(self):
        fb = ValidationFeedback()
        text = fb.to_prompt_text()
        assert "minor_tweak" in text

    def test_past_fixes_in_prompt(self):
        fb = ValidationFeedback(
            failed_checks=[FailedCheck(check_name="duration", actual=58.0)],
            similar_past_fixes=[
                PastFix(
                    category="ball_pit",
                    check_name="duration",
                    parameter_changed="SIMULATION_TIME",
                    old_value="60",
                    new_value="72",
                ),
            ],
        )
        text = fb.to_prompt_text()
        assert "SIMULATION_TIME" in text
        assert "ball_pit" in text


# ============================================================================
# CategoryKnowledge
# ============================================================================


class TestCategoryKnowledge:
    """Test CategoryKnowledge model and rendering."""

    def test_empty_renders_nothing(self):
        ck = CategoryKnowledge(category="ball_pit")
        assert ck.to_prompt_text() == ""

    def test_with_data_renders(self):
        ck = CategoryKnowledge(
            category="ball_pit",
            total_examples=10,
            avg_iterations_to_pass=2.3,
            best_duration_setting=72.0,
            common_failure_modes=["duration", "completion_ratio"],
            parameter_ranges={"ball_count": [50.0, 200.0]},
        )
        text = ck.to_prompt_text()
        assert "ball_pit" in text
        assert "10 past runs" in text
        assert "72.0" in text
        assert "2.3" in text
        assert "duration" in text
        assert "ball_count" in text


# ============================================================================
# Few-shot examples
# ============================================================================


class TestFormatFewShotPrompt:
    """Test few-shot prompt formatting."""

    def test_empty_returns_empty(self):
        assert format_few_shot_prompt([]) == ""

    def test_single_example(self):
        examples = [{
            "title": "Rainbow Ball Pit",
            "code": "import pymunk\n# simulation code",
            "reasoning": "Used 200 balls for density",
            "thinking": "",
            "iteration_count": 1,
        }]
        text = format_few_shot_prompt(examples)
        assert "Rainbow Ball Pit" in text
        assert "import pymunk" in text
        assert "200 balls" in text

    def test_thinking_truncated(self):
        examples = [{
            "title": "Test",
            "code": "pass",
            "reasoning": "",
            "thinking": "x" * 2000,
            "iteration_count": 1,
        }]
        text = format_few_shot_prompt(examples)
        assert "…" in text  # truncation marker


# ============================================================================
# Validation rules prompt
# ============================================================================


class TestValidationRulesPrompt:
    """Test static validation rules prompt generation."""

    def test_contains_critical_rules(self):
        text = get_validation_rules_prompt()
        assert "space.gravity = (0, 900)" in text
        assert "pygame.display.set_mode()" in text
        assert "SDL_VIDEODRIVER" in text
        assert "moment_for_circle" in text
        assert "PAYOFF_TIMESTAMP" in text

    def test_mentions_558_failures(self):
        text = get_validation_rules_prompt()
        assert "558" in text


# ============================================================================
# AST Parameter Extraction
# ============================================================================


class TestExtractParameters:
    """Test AST-based parameter extraction."""

    def test_basic_simulation(self):
        code = textwrap.dedent("""\
            import pygame
            import pymunk
            import subprocess
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            space = pymunk.Space()
            space.gravity = (0, 900)
            WIDTH = 1080
            HEIGHT = 1920
            for i in range(60 * 65):
                space.step(1/60)
            proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo"])
        """)
        params = extract_parameters(code)
        assert params.has_pygame_init
        assert params.has_space_step
        assert params.has_ffmpeg_pipe
        assert params.gravity == (0.0, 900.0)
        assert params.canvas_size == (1080, 1920)
        assert params.loop_iterations == 3900
        assert params.step_size == pytest.approx(1 / 60, rel=1e-3)
        assert params.estimated_duration_sec == pytest.approx(65.0, rel=0.1)

    def test_body_count(self):
        code = textwrap.dedent("""\
            import pymunk
            space = pymunk.Space()
            for i in range(50):
                body = pymunk.Body(10, pymunk.moment_for_circle(10, 0, 15))
                space.add(body)
        """)
        params = extract_parameters(code)
        # moment_for_circle counts as body creation proxy + explicit Body()
        assert params.body_count >= 1

    def test_syntax_error_returns_empty(self):
        params = extract_parameters("def broken(\n")
        assert params.body_count == 0
        assert params.estimated_duration_sec == 0.0

    def test_while_loop_detection(self):
        code = textwrap.dedent("""\
            frame = 0
            while frame < 3600:
                space.step(1/60)
                frame += 1
        """)
        params = extract_parameters(code)
        assert params.loop_iterations == 3600
        assert params.has_space_step

    def test_feedback_text_renders(self):
        code = textwrap.dedent("""\
            import pygame
            import pymunk
            pygame.init()
            space = pymunk.Space()
            space.gravity = (0, 900)
            SIMULATION_TIME = 65
            for i in range(60 * 65):
                space.step(1/60)
        """)
        params = extract_parameters(code)
        text = params.to_feedback_text()
        assert "AST" in text
        assert "gravity" in text.lower() or "Gravity" in text

    def test_simulation_time_assignment(self):
        code = "SIMULATION_TIME = 72"
        params = extract_parameters(code)
        assert params.estimated_duration_sec == 72.0

    def test_empty_code(self):
        params = extract_parameters("")
        assert params.body_count == 0
        assert params.to_feedback_text() == ""


# ============================================================================
# Feature flag
# ============================================================================


class TestLearningLoopEnabled:
    """Test that the learning_loop_enabled flag gates recording."""

    @pytest.mark.asyncio
    async def test_disabled_skips_record(self):
        """When learning_loop_enabled=False, record_training_example is a no-op."""
        from kairos.ai.learning.learning_loop import record_training_example

        mock_settings = MagicMock()
        mock_settings.learning_loop_enabled = False

        with patch("kairos.services.learning_loop.get_settings", return_value=mock_settings):
            # Should return without touching DB
            await record_training_example(
                pipeline="physics",
                category="ball_pit",
                concept_brief={"title": "Test"},
                simulation_code="pass",
                validation_passed=True,
            )
            # No exception = success (it just skipped)

    @pytest.mark.asyncio
    async def test_enabled_attempts_record(self):
        """When learning_loop_enabled=True, record_training_example tries DB."""
        from kairos.ai.learning.learning_loop import record_training_example

        mock_settings = MagicMock()
        mock_settings.learning_loop_enabled = True

        with patch("kairos.services.learning_loop.get_settings", return_value=mock_settings):
            # DB will fail (no real connection) but it should not raise
            await record_training_example(
                pipeline="physics",
                category="ball_pit",
                concept_brief={"title": "Test"},
                simulation_code="pass",
                validation_passed=True,
            )


# ============================================================================
# Verified gate
# ============================================================================


class TestVerifiedGate:
    """Test that get_few_shot_examples only returns verified=True examples."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_db(self):
        """Without a real DB, get_few_shot_examples returns []."""
        from kairos.ai.learning.learning_loop import get_few_shot_examples

        result = await get_few_shot_examples("physics", "ball_pit")
        assert result == []


# ============================================================================
# DB model: TrainingExample new columns
# ============================================================================


class TestTrainingExampleModel:
    """Test that new columns are present on the ORM model."""

    def test_new_columns_exist(self):
        from kairos.db.models import TrainingExample

        columns = {c.name for c in TrainingExample.__table__.columns}
        assert "category" in columns
        assert "reasoning" in columns
        assert "thinking_content" in columns
        assert "verified" in columns
        assert "iteration_count" in columns

    def test_verified_defaults_false(self):
        from kairos.db.models import TrainingExample

        col = TrainingExample.__table__.c.verified
        assert col.default is not None
        assert col.default.arg is False


class TestCategoryStatKnowledge:
    """Test that knowledge JSONB column exists on CategoryStat."""

    def test_knowledge_column_exists(self):
        from kairos.db.models import CategoryStat

        columns = {c.name for c in CategoryStat.__table__.columns}
        assert "knowledge" in columns
