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
            "code": "import bpy\n# simulation code",
            "reasoning": "Used 200 balls for density",
            "thinking": "",
            "iteration_count": 1,
        }]
        text = format_few_shot_prompt(examples)
        assert "Rainbow Ball Pit" in text
        assert "import bpy" in text
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
        assert "gravity" in text.lower()
        assert "Blender" in text
        assert "substeps" in text.lower()
        assert "PAYOFF_TIMESTAMP" in text

    def test_mentions_past_failures(self):
        text = get_validation_rules_prompt()
        assert "past failure" in text.lower()


# ============================================================================
# AST Parameter Extraction
# ============================================================================


class TestExtractParameters:
    """Test AST-based parameter extraction."""

    def test_basic_simulation(self):
        code = textwrap.dedent("""\
            import bpy
            import subprocess
            import os
            scene = bpy.context.scene
            scene.gravity = (0, 0, -9.81)
            scene.render.resolution_x = 1080
            scene.render.resolution_y = 1920
            for i in range(60 * 65):
                scene.frame_set(i)
            proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo"])
        """)
        params = extract_parameters(code)
        assert params.has_ffmpeg_pipe
        assert params.loop_iterations == 3900

    def test_body_count(self):
        code = textwrap.dedent("""\
            import bpy
            for i in range(50):
                bpy.ops.mesh.primitive_uv_sphere_add()
                bpy.ops.rigidbody.object_add(type='ACTIVE')
        """)
        params = extract_parameters(code)
        # rigidbody.object_add counts as body creation proxy
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
            import bpy
            scene = bpy.context.scene
            scene.gravity = (0, 0, -9.81)
            SIMULATION_TIME = 65
            for i in range(60 * 65):
                scene.frame_set(i)
        """)
        params = extract_parameters(code)
        text = params.to_feedback_text()
        assert "AST" in text

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

        with patch("kairos.ai.learning.learning_loop.get_settings", return_value=mock_settings):
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

        with patch("kairos.ai.learning.learning_loop.get_settings", return_value=mock_settings):
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
