"""Unit tests for calibration quality gate and scenario mapping.

Pure logic — no I/O, no Blender, no ChromaDB.
"""

from __future__ import annotations

import pytest

from kairos.calibration.models import (
    CalibrationSession,
    CalibrationStatus,
    CorrectionFactors,
    IterationRecord,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.quality_gate import (
    AUTO_APPROVE_AFTER,
    CHAIN_COMPLETION_THRESHOLD,
    MAX_PHYSICS_ANOMALIES,
    PERCEPTUAL_SCORE_THRESHOLD,
    compute_confidence,
    evaluate,
)
from kairos.calibration.scenario import (
    blender_config_to_scenario,
    compute_dimensional_overlap,
    scenario_to_blender_config,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def straight_30() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0, cycles=1.0),
        domino_count=30,
    )


def _make_session(
    scenario: ScenarioDescriptor,
    iterations: int = 3,
    completion: float = 1.0,
) -> CalibrationSession:
    records = [
        IterationRecord(
            iteration=i,
            params_used={},
            validation_passed=(i == iterations),
            completion_ratio=completion if i == iterations else completion * 0.5,
        )
        for i in range(1, iterations + 1)
    ]
    session = CalibrationSession(
        scenario=scenario,
        max_iterations=10,
        iterations=records,
        status=CalibrationStatus.RESOLVED,
        final_corrections=CorrectionFactors(),
    )
    return session


# =============================================================================
# Quality Gate — evaluate()
# =============================================================================

class TestQualityGate:
    def test_perfect_run_passes(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=3, completion=1.0)
        result = evaluate(session, chain_completion=1.0, physics_anomalies=0, perceptual_score=0.9)
        assert result.passed

    def test_low_completion_fails(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=3, completion=0.7)
        result = evaluate(session, chain_completion=0.85, physics_anomalies=0, perceptual_score=0.9)
        assert not result.passed

    def test_physics_anomalies_fail(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=3)
        result = evaluate(session, chain_completion=1.0, physics_anomalies=1, perceptual_score=0.9)
        assert not result.passed

    def test_low_perceptual_score_fails(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=3)
        result = evaluate(session, chain_completion=1.0, physics_anomalies=0, perceptual_score=0.5)
        assert not result.passed

    def test_high_deviation_triggers_human_review(
        self, straight_30: ScenarioDescriptor
    ) -> None:
        # Set a correction factor that's >3× baseline
        session = _make_session(straight_30, iterations=3)
        session.final_corrections = CorrectionFactors(spacing_ratio=4.0)  # way off baseline
        result = evaluate(session, chain_completion=1.0, physics_anomalies=0, perceptual_score=0.9)
        assert result.requires_human_review

    def test_first_of_archetype_triggers_human_review(
        self, straight_30: ScenarioDescriptor
    ) -> None:
        """archetype_review_count=0 (default) should flag for human review."""
        session = _make_session(straight_30, iterations=3)
        # Default archetype_review_count=0 < AUTO_APPROVE_AFTER=20 → requires review
        result = evaluate(
            session,
            chain_completion=1.0,
            physics_anomalies=0,
            perceptual_score=0.9,
        )
        assert result.requires_human_review
        assert "first" in result.human_review_reason.lower()


# =============================================================================
# Quality Gate — compute_confidence()
# =============================================================================

class TestComputeConfidence:
    def test_fast_convergence_is_high_confidence(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=2)
        score = compute_confidence(session)
        assert score > 0.8

    def test_slow_convergence_reduces_confidence(self, straight_30: ScenarioDescriptor) -> None:
        fast = compute_confidence(_make_session(straight_30, iterations=2))
        slow = compute_confidence(_make_session(straight_30, iterations=9))
        assert slow < fast

    def test_confidence_clamped_0_to_1(self, straight_30: ScenarioDescriptor) -> None:
        session = _make_session(straight_30, iterations=20)  # extreme iteration count
        score = compute_confidence(session)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Scenario ↔ Blender Config mapping
# =============================================================================

class TestScenarioToBlenderConfig:
    def test_straight_maps_to_s_curve_archetype(self, straight_30: ScenarioDescriptor) -> None:
        config = scenario_to_blender_config(straight_30)
        # STRAIGHT maps to s_curve archetype with amplitude 0
        assert config["archetype"] == "s_curve"
        assert config["path_amplitude"] == pytest.approx(0.0)

    def test_spiral_maps_to_spiral_archetype(self) -> None:
        scenario = ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=3.0),
            domino_count=200,
        )
        config = scenario_to_blender_config(scenario)
        assert config["archetype"] == "spiral"
        assert config["spiral_turns"] == pytest.approx(3.0)

    def test_branching_maps_to_branching(self) -> None:
        scenario = ScenarioDescriptor(
            path=PathDescriptor(type=PathType.BRANCHING, branch_count=4),
            domino_count=300,
        )
        config = scenario_to_blender_config(scenario)
        assert config["archetype"] == "branching"
        assert config["branch_count"] == 4

    def test_corrections_applied_to_physics(self, straight_30: ScenarioDescriptor) -> None:
        corrections = CorrectionFactors(spacing_ratio=0.9, trigger_impulse=1.3)
        config = scenario_to_blender_config(straight_30, corrections=corrections)
        from kairos.calibration.models import BASELINE_PHYSICS
        expected_spacing = BASELINE_PHYSICS["spacing_ratio"] * 0.9
        expected_impulse = BASELINE_PHYSICS["trigger_impulse"] * 1.3
        assert config["spacing_ratio"] == pytest.approx(expected_spacing, rel=1e-4)
        assert config["trigger_impulse"] == pytest.approx(expected_impulse, rel=1e-4)

    def test_substeps_is_integer(self, straight_30: ScenarioDescriptor) -> None:
        config = scenario_to_blender_config(straight_30)
        assert isinstance(config["substeps_per_frame"], int)
        assert isinstance(config["solver_iterations"], int)

    def test_domino_count_passed_through(self, straight_30: ScenarioDescriptor) -> None:
        config = scenario_to_blender_config(straight_30)
        assert config["domino_count"] == 30


# =============================================================================
# compute_dimensional_overlap
# =============================================================================

class TestDimensionalOverlap:
    def test_identical_scenarios_score_high(self, straight_30: ScenarioDescriptor) -> None:
        # STRAIGHT: amplitude=0.0 skips the amplitude dimension (guarded by stored_amp > 0),
        # so max reachable score is 0.85 (path+surface+size+count+height weights).
        score, _ = compute_dimensional_overlap(straight_30, straight_30.to_metadata())
        assert score > 0.80

    def test_different_path_types_score_less_than_identical(self, straight_30: ScenarioDescriptor) -> None:
        spiral = ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=2.0),
            domino_count=30,
        )
        identical_score, _ = compute_dimensional_overlap(straight_30, straight_30.to_metadata())
        different_score, _ = compute_dimensional_overlap(straight_30, spiral.to_metadata())
        # Different path type loses the 0.30 path_type weight
        assert different_score < identical_score

    def test_same_type_different_count_partial_overlap(
        self, straight_30: ScenarioDescriptor
    ) -> None:
        large = ScenarioDescriptor(
            path=PathDescriptor(type=PathType.STRAIGHT),
            domino_count=300,
        )
        score, _ = compute_dimensional_overlap(straight_30, large.to_metadata())
        # Same path type should give partial overlap despite count difference
        assert 0.2 < score < 1.0
