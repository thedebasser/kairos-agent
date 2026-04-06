"""Unit tests for calibration data models.

Pure logic tests — no external dependencies, no I/O, no Blender.
Covers CorrectionFactors math, ScenarioDescriptor serialisation,
and CalibrationEntry round-trips.
"""

from __future__ import annotations

import pytest

from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CalibrationStatus,
    CorrectionFactors,
    FailureMode,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
    SizeProfile,
    SizeProfileDescriptor,
    SurfaceDescriptor,
    SurfaceType,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def baseline_corrections() -> CorrectionFactors:
    """All factors at 1.0 — identical to baseline."""
    return CorrectionFactors()


@pytest.fixture
def straight_scenario() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0, cycles=1.0),
        domino_count=30,
    )


@pytest.fixture
def s_curve_scenario() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.S_CURVE, amplitude=1.5, cycles=2.0),
        domino_count=300,
    )


# =============================================================================
# CorrectionFactors math
# =============================================================================

class TestCorrectionFactors:
    def test_default_factors_are_all_ones(self, baseline_corrections: CorrectionFactors) -> None:
        for field, val in baseline_corrections.model_dump().items():
            if isinstance(val, float):
                assert val == pytest.approx(1.0), f"{field} should default to 1.0"

    def test_apply_to_baseline_identity(self, baseline_corrections: CorrectionFactors) -> None:
        """1.0 × baseline == baseline."""
        applied = baseline_corrections.apply_to_baseline()
        for param, expected in BASELINE_PHYSICS.items():
            assert applied[param] == pytest.approx(expected, rel=1e-5), \
                f"{param}: expected {expected}, got {applied[param]}"

    def test_apply_to_baseline_scaling(self) -> None:
        """A factor of 2.0 doubles the baseline value."""
        corrections = CorrectionFactors(spacing_ratio=2.0)
        applied = corrections.apply_to_baseline()
        expected = BASELINE_PHYSICS["spacing_ratio"] * 2.0
        assert applied["spacing_ratio"] == pytest.approx(expected)
        # Other params remain at baseline
        assert applied["domino_mass"] == pytest.approx(BASELINE_PHYSICS["domino_mass"])

    def test_from_absolute_roundtrip(self) -> None:
        """from_absolute(apply_to_baseline()) == original corrections."""
        original = CorrectionFactors(
            spacing_ratio=0.9,
            domino_mass=1.1,
            trigger_impulse=1.2,
        )
        absolute = original.apply_to_baseline()
        recovered = CorrectionFactors.from_absolute(absolute)

        assert recovered.spacing_ratio == pytest.approx(original.spacing_ratio, rel=1e-4)
        assert recovered.domino_mass == pytest.approx(original.domino_mass, rel=1e-4)
        assert recovered.trigger_impulse == pytest.approx(original.trigger_impulse, rel=1e-4)

    def test_apply_to_baseline_includes_curve_factor(self) -> None:
        """curve_inner_spacing_factor is passed through directly."""
        corrections = CorrectionFactors(curve_inner_spacing_factor=0.85)
        applied = corrections.apply_to_baseline()
        assert applied["curve_inner_spacing_factor"] == pytest.approx(0.85)

    def test_from_absolute_with_missing_keys(self) -> None:
        """from_absolute with partial dict leaves unspecified factors at 1.0."""
        partial = {"spacing_ratio": BASELINE_PHYSICS["spacing_ratio"] * 0.9}
        factors = CorrectionFactors.from_absolute(partial)
        assert factors.spacing_ratio == pytest.approx(0.9, rel=1e-4)
        assert factors.domino_mass == pytest.approx(1.0)


# =============================================================================
# ScenarioDescriptor
# =============================================================================

class TestScenarioDescriptor:
    def test_to_natural_language_contains_key_terms(
        self, straight_scenario: ScenarioDescriptor
    ) -> None:
        nl = straight_scenario.to_natural_language()
        assert "straight" in nl.lower()
        assert "30" in nl
        assert "flat" in nl.lower()

    def test_to_natural_language_s_curve(self, s_curve_scenario: ScenarioDescriptor) -> None:
        nl = s_curve_scenario.to_natural_language()
        assert "s_curve" in nl.lower()
        assert "1.5" in nl   # amplitude
        assert "300" in nl

    def test_to_metadata_flat(self, straight_scenario: ScenarioDescriptor) -> None:
        meta = straight_scenario.to_metadata()
        assert meta["path_type"] == "straight"
        assert meta["domino_count"] == 30
        assert meta["surface_type"] == "flat"

    def test_to_metadata_s_curve_includes_amplitude(
        self, s_curve_scenario: ScenarioDescriptor
    ) -> None:
        meta = s_curve_scenario.to_metadata()
        assert "path_amplitude" in meta
        assert meta["path_amplitude"] == pytest.approx(1.5)
        assert "path_cycles" in meta

    def test_to_metadata_spiral_includes_turns(self) -> None:
        scenario = ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=3.5),
            domino_count=200,
        )
        meta = scenario.to_metadata()
        assert "spiral_turns" in meta
        assert meta["spiral_turns"] == pytest.approx(3.5)

    def test_domino_count_bounds(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ScenarioDescriptor(
                path=PathDescriptor(type=PathType.STRAIGHT),
                domino_count=5,  # below minimum of 10
            )
        with pytest.raises(ValidationError):
            ScenarioDescriptor(
                path=PathDescriptor(type=PathType.STRAIGHT),
                domino_count=1001,  # above maximum of 1000
            )

    def test_default_surface_is_flat(self, straight_scenario: ScenarioDescriptor) -> None:
        assert straight_scenario.surface.type == SurfaceType.FLAT
        assert straight_scenario.surface.tilt_degrees == 0.0

    def test_default_size_profile_is_uniform(
        self, straight_scenario: ScenarioDescriptor
    ) -> None:
        assert straight_scenario.size_profile.type == SizeProfile.UNIFORM


# =============================================================================
# FailureMode enum
# =============================================================================

class TestFailureMode:
    def test_all_modes_are_strings(self) -> None:
        for mode in FailureMode:
            assert isinstance(mode.value, str)

    def test_unknown_mode_exists(self) -> None:
        assert FailureMode.UNKNOWN in list(FailureMode)
