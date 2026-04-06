"""Unit tests for calibration bootstrap scenario definitions.

Validates the PHASE_1_SCENARIOS list and scenario_to_blender_config
integration for each bootstrap entry.  Pure data tests — no I/O.
"""

from __future__ import annotations

import pytest

from kairos.calibration.bootstrap import PHASE_1_SCENARIOS
from kairos.calibration.models import (
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.scenario import scenario_to_blender_config


pytestmark = pytest.mark.unit


class TestPhase1Scenarios:
    def test_exactly_10_scenarios(self) -> None:
        assert len(PHASE_1_SCENARIOS) == 10

    def test_all_have_required_keys(self) -> None:
        for i, entry in enumerate(PHASE_1_SCENARIOS):
            assert "name" in entry, f"Scenario {i} missing 'name'"
            assert "description" in entry, f"Scenario {i} missing 'description'"
            assert "scenario" in entry, f"Scenario {i} missing 'scenario'"
            assert isinstance(entry["scenario"], ScenarioDescriptor), (
                f"Scenario {i} 'scenario' is not a ScenarioDescriptor"
            )

    def test_all_names_unique(self) -> None:
        names = [s["name"] for s in PHASE_1_SCENARIOS]
        assert len(names) == len(set(names)), "Duplicate scenario names found"

    def test_expected_path_types_covered(self) -> None:
        """Phase 1 covers STRAIGHT, S_CURVE, SPIRAL, CASCADE, BRANCHING."""
        path_types = {s["scenario"].path.type for s in PHASE_1_SCENARIOS}
        expected = {
            PathType.STRAIGHT,
            PathType.S_CURVE,
            PathType.SPIRAL,
            PathType.CASCADE,
            PathType.BRANCHING,
        }
        assert expected == path_types

    def test_domino_counts_within_bounds(self) -> None:
        for entry in PHASE_1_SCENARIOS:
            sc: ScenarioDescriptor = entry["scenario"]
            assert 10 <= sc.domino_count <= 1000, (
                f"{entry['name']} has out-of-bounds domino_count={sc.domino_count}"
            )

    def test_all_produce_valid_blender_configs(self) -> None:
        """Each scenario can be converted to a Blender config without errors."""
        required_keys = {
            "archetype", "domino_count", "spacing_ratio", "domino_mass",
            "trigger_impulse", "substeps_per_frame", "solver_iterations",
        }
        for entry in PHASE_1_SCENARIOS:
            config = scenario_to_blender_config(entry["scenario"])
            missing = required_keys - set(config.keys())
            assert not missing, (
                f"{entry['name']} config missing keys: {missing}"
            )

    def test_straight_scenarios_have_zero_amplitude(self) -> None:
        for entry in PHASE_1_SCENARIOS:
            sc: ScenarioDescriptor = entry["scenario"]
            if sc.path.type == PathType.STRAIGHT:
                assert sc.path.amplitude == pytest.approx(0.0), (
                    f"{entry['name']} straight path should have amplitude=0"
                )

    def test_spiral_scenarios_have_turns(self) -> None:
        for entry in PHASE_1_SCENARIOS:
            sc: ScenarioDescriptor = entry["scenario"]
            if sc.path.type == PathType.SPIRAL:
                assert sc.path.spiral_turns is not None and sc.path.spiral_turns > 0, (
                    f"{entry['name']} spiral should have spiral_turns > 0"
                )

    def test_branching_scenarios_have_branch_count(self) -> None:
        for entry in PHASE_1_SCENARIOS:
            sc: ScenarioDescriptor = entry["scenario"]
            if sc.path.type == PathType.BRANCHING:
                assert sc.path.branch_count is not None and sc.path.branch_count >= 2, (
                    f"{entry['name']} branching should have branch_count >= 2"
                )
