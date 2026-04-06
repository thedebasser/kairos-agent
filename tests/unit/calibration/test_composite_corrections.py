"""Unit tests for composite_corrections() blending logic.

Pure math tests — no I/O, no external services.
"""

from __future__ import annotations

import pytest

from kairos.calibration.models import (
    CorrectionFactors,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.scenario import composite_corrections


pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================

def _make_match(
    *,
    spacing_ratio: float = 1.0,
    trigger_impulse: float = 1.0,
    combined_score: float = 0.8,
    confidence: float = 0.9,
    matching_dimensions: list[str] | None = None,
) -> dict:
    return {
        "corrections": CorrectionFactors(
            spacing_ratio=spacing_ratio,
            trigger_impulse=trigger_impulse,
        ),
        "combined_score": combined_score,
        "confidence": confidence,
        "matching_dimensions": matching_dimensions or ["path_type", "surface_type"],
    }


def _query_scenario() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.S_CURVE, amplitude=1.5, cycles=2.0),
        domino_count=300,
    )


# =============================================================================
# Tests
# =============================================================================

class TestCompositeCorrectionsSingleMatch:
    def test_single_match_returns_its_corrections(self) -> None:
        match = _make_match(spacing_ratio=0.9)
        result = composite_corrections([match], _query_scenario())
        assert result.spacing_ratio == pytest.approx(0.9)

    def test_default_factors_stay_at_one(self) -> None:
        """Correction factors at 1.0 carry no information and are skipped."""
        match = _make_match(spacing_ratio=0.9, trigger_impulse=1.0)
        result = composite_corrections([match], _query_scenario())
        # trigger_impulse was 1.0 → ignored → stays 1.0
        assert result.trigger_impulse == pytest.approx(1.0)


class TestCompositeCorrectionsMultipleMatches:
    def test_weighted_average_of_two_matches(self) -> None:
        """Two matches with equal weight should average their corrections."""
        m1 = _make_match(spacing_ratio=0.8, combined_score=0.5, confidence=1.0)
        m2 = _make_match(spacing_ratio=1.2, combined_score=0.5, confidence=1.0)
        result = composite_corrections([m1, m2], _query_scenario())
        # Both have weight = 0.5 * 1.0 = 0.5
        # Weighted avg = (0.8*0.5 + 1.2*0.5) / (0.5+0.5) = 1.0
        assert result.spacing_ratio == pytest.approx(1.0)

    def test_higher_score_match_dominates(self) -> None:
        """Match with higher combined_score gets more weight."""
        m_strong = _make_match(spacing_ratio=0.85, combined_score=0.9, confidence=1.0)
        m_weak = _make_match(spacing_ratio=1.1, combined_score=0.1, confidence=1.0)
        result = composite_corrections([m_strong, m_weak], _query_scenario())
        # Strong match should dominate, so result closer to 0.85
        assert result.spacing_ratio < 0.95

    def test_higher_confidence_match_dominates(self) -> None:
        """Match with higher confidence gets more weight."""
        m_confident = _make_match(spacing_ratio=0.85, combined_score=0.5, confidence=1.0)
        m_uncertain = _make_match(spacing_ratio=1.2, combined_score=0.5, confidence=0.1)
        result = composite_corrections([m_confident, m_uncertain], _query_scenario())
        # Confident match should dominate
        assert result.spacing_ratio < 1.0


class TestCompositeCorrectionsEdgeCases:
    def test_empty_matches_returns_default(self) -> None:
        result = composite_corrections([], _query_scenario())
        assert result.spacing_ratio == pytest.approx(1.0)
        assert result.trigger_impulse == pytest.approx(1.0)

    def test_match_with_no_corrections_key_skipped(self) -> None:
        match = {"combined_score": 0.5, "confidence": 0.5, "matching_dimensions": ["path_type"]}
        result = composite_corrections([match], _query_scenario())
        assert result.spacing_ratio == pytest.approx(1.0)

    def test_dict_corrections_accepted(self) -> None:
        """Corrections can be a plain dict instead of CorrectionFactors."""
        match = {
            "corrections": {"spacing_ratio": 0.88, "trigger_impulse": 1.1},
            "combined_score": 0.8,
            "confidence": 0.9,
            "matching_dimensions": ["path_type"],
        }
        result = composite_corrections([match], _query_scenario())
        assert result.spacing_ratio == pytest.approx(0.88)

    def test_dimension_filtering_applies(self) -> None:
        """Corrections only apply when matching_dimensions includes path_type or surface_type."""
        match = _make_match(
            spacing_ratio=0.8,
            matching_dimensions=["domino_count_similar"],  # no path_type or surface_type
        )
        result = composite_corrections([match], _query_scenario())
        # spacing_ratio correction should NOT apply (dimension filter blocks it)
        assert result.spacing_ratio == pytest.approx(1.0)

    def test_solver_iterations_always_applies(self) -> None:
        """substeps_per_frame and solver_iterations bypass dimension filtering."""
        match = {
            "corrections": CorrectionFactors(substeps_per_frame=1.5, solver_iterations=1.2),
            "combined_score": 0.8,
            "confidence": 0.9,
            "matching_dimensions": ["domino_count_similar"],  # no path or surface
        }
        result = composite_corrections([match], _query_scenario())
        assert result.substeps_per_frame == pytest.approx(1.5)
        assert result.solver_iterations == pytest.approx(1.2)
