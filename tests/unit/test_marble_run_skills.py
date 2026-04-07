"""Tests for marble run skill library — track pieces, physics,
connector validation, momentum calculator, and camera config.

These run without Blender.
"""

from __future__ import annotations

import math

import pytest

from kairos.skills.marble_run import (
    DEFAULT_MARBLE_MASS,
    DEFAULT_MARBLE_RADIUS,
    TRACK_CONNECTOR_DIAMETER,
    TRACK_WALL_HEIGHT,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestMarbleRunConstants:
    def test_default_marble_radius(self):
        assert DEFAULT_MARBLE_RADIUS == 0.04

    def test_default_marble_mass(self):
        assert DEFAULT_MARBLE_MASS == 0.028

    def test_connector_diameter(self):
        assert TRACK_CONNECTOR_DIAMETER == 0.05

    def test_wall_height(self):
        assert TRACK_WALL_HEIGHT == 0.03


# ---------------------------------------------------------------------------
# Place ball
# ---------------------------------------------------------------------------

class TestPlaceBall:
    def test_returns_params(self):
        from kairos.skills.marble_run.place_ball import place_ball
        params = place_ball((1.0, 2.0, 3.0))
        assert params["type"] == "marble"
        assert params["position"] == [1.0, 2.0, 3.0]
        assert params["radius"] == DEFAULT_MARBLE_RADIUS
        assert params["material"] == "glass"

    def test_custom_radius(self):
        from kairos.skills.marble_run.place_ball import place_ball
        params = place_ball((0, 0, 0), radius=0.06)
        assert params["radius"] == 0.06

    def test_custom_colour_and_material(self):
        from kairos.skills.marble_run.place_ball import place_ball
        params = place_ball((0, 0, 0), colour_hex="#FF0000", material="metal")
        assert params["colour_hex"] == "#FF0000"
        assert params["material"] == "metal"

    def test_custom_name(self):
        from kairos.skills.marble_run.place_ball import place_ball
        params = place_ball((0, 0, 0), name="marble_007")
        assert params["name"] == "marble_007"


# ---------------------------------------------------------------------------
# Apply ball physics
# ---------------------------------------------------------------------------

class TestApplyBallPhysics:
    def test_defaults(self):
        from kairos.skills.marble_run.apply_ball_physics import apply_ball_physics
        rb = apply_ball_physics()
        assert rb["type"] == "ACTIVE"
        assert rb["mass"] == DEFAULT_MARBLE_MASS
        assert rb["collision_shape"] == "SPHERE"
        assert rb["friction"] == 0.6
        assert rb["restitution"] == 0.4

    def test_custom_mass(self):
        from kairos.skills.marble_run.apply_ball_physics import apply_ball_physics
        rb = apply_ball_physics(mass=0.05)
        assert rb["mass"] == 0.05

    def test_low_damping(self):
        from kairos.skills.marble_run.apply_ball_physics import apply_ball_physics
        rb = apply_ball_physics()
        assert rb["linear_damping"] == 0.04
        assert rb["angular_damping"] == 0.1


# ---------------------------------------------------------------------------
# Track pieces
# ---------------------------------------------------------------------------

class TestStraightTrack:
    def test_basic(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        piece = create_straight_track((0, 0, 1), (1, 0, 0.5))
        assert piece["piece_type"] == "straight_track"
        assert piece["length"] > 0
        assert piece["height_drop"] == pytest.approx(0.5, abs=0.001)
        assert piece["entry_port"]["position"] == [0, 0, 1]
        assert piece["exit_port"]["position"] == [1, 0, 0.5]

    def test_zero_length(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        piece = create_straight_track((0, 0, 0), (0, 0, 0))
        assert piece["length"] == 0.0

    def test_connector_diameter(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        piece = create_straight_track((0, 0, 1), (1, 0, 0))
        assert piece["entry_port"]["diameter"] == TRACK_CONNECTOR_DIAMETER
        assert piece["exit_port"]["diameter"] == TRACK_CONNECTOR_DIAMETER

    def test_custom_width(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        piece = create_straight_track((0, 0, 0), (1, 0, 0), width=0.15)
        assert piece["width"] == 0.15


class TestCurvedTrack:
    def test_basic(self):
        from kairos.skills.marble_run.pieces import create_curved_track
        piece = create_curved_track((0, 0, 0), radius=0.5, start_angle_deg=0, sweep_deg=90)
        assert piece["piece_type"] == "curved_track"
        assert piece["arc_length"] > 0
        assert piece["radius"] == 0.5

    def test_90_degree_arc_length(self):
        from kairos.skills.marble_run.pieces import create_curved_track
        piece = create_curved_track((0, 0, 0), radius=1.0, start_angle_deg=0, sweep_deg=90)
        expected = math.pi / 2
        assert piece["arc_length"] == pytest.approx(expected, abs=0.01)

    def test_entry_exit_positions(self):
        from kairos.skills.marble_run.pieces import create_curved_track
        piece = create_curved_track((0, 0, 0), radius=1.0, start_angle_deg=0, sweep_deg=90)
        entry = piece["entry_port"]["position"]
        exit_ = piece["exit_port"]["position"]
        # Entry at angle 0: (1, 0, 0)
        assert entry[0] == pytest.approx(1.0, abs=0.01)
        assert entry[1] == pytest.approx(0.0, abs=0.01)
        # Exit at angle 90: (0, 1, 0)
        assert exit_[0] == pytest.approx(0.0, abs=0.01)
        assert exit_[1] == pytest.approx(1.0, abs=0.01)

    def test_height_change(self):
        from kairos.skills.marble_run.pieces import create_curved_track
        piece = create_curved_track(
            (0, 0, 0), radius=0.5, start_angle_deg=0, sweep_deg=180,
            start_height=1.0, end_height=0.5,
        )
        assert piece["height_drop"] == pytest.approx(0.5, abs=0.001)

    def test_bank_angle(self):
        from kairos.skills.marble_run.pieces import create_curved_track
        piece = create_curved_track(
            (0, 0, 0), radius=0.5, start_angle_deg=0, sweep_deg=90,
            bank_angle_deg=15.0,
        )
        assert piece["bank_angle_deg"] == 15.0


class TestFunnel:
    def test_basic(self):
        from kairos.skills.marble_run.pieces import create_funnel
        piece = create_funnel((0, 0, 1))
        assert piece["piece_type"] == "funnel"
        assert piece["entry_width"] == 0.40
        assert piece["exit_width"] == 0.10
        assert piece["height_drop"] == pytest.approx(0.05, abs=0.001)

    def test_entry_port_diameter_matches_width(self):
        from kairos.skills.marble_run.pieces import create_funnel
        piece = create_funnel((0, 0, 1), entry_width=0.5)
        assert piece["entry_port"]["diameter"] == 0.5

    def test_exit_port_standard_diameter(self):
        from kairos.skills.marble_run.pieces import create_funnel
        piece = create_funnel((0, 0, 1))
        assert piece["exit_port"]["diameter"] == TRACK_CONNECTOR_DIAMETER


class TestLoop:
    def test_basic(self):
        from kairos.skills.marble_run.pieces import create_loop
        piece = create_loop((0, 0, 0))
        assert piece["piece_type"] == "loop"
        assert piece["height_drop"] == 0.0
        assert piece["loop_circumference"] > 0

    def test_circumference(self):
        from kairos.skills.marble_run.pieces import create_loop
        piece = create_loop((0, 0, 0), radius=0.2)
        expected = 2 * math.pi * 0.2
        assert piece["loop_circumference"] == pytest.approx(expected, abs=0.01)


class TestDrop:
    def test_basic(self):
        from kairos.skills.marble_run.pieces import create_drop
        piece = create_drop((0, 0, 2))
        assert piece["piece_type"] == "drop"
        assert piece["height_drop"] == pytest.approx(0.5, abs=0.001)
        assert piece["length"] > 0

    def test_custom_height(self):
        from kairos.skills.marble_run.pieces import create_drop
        piece = create_drop((0, 0, 3), height=1.5)
        assert piece["height_drop"] == pytest.approx(1.5, abs=0.001)

    def test_exit_below_start(self):
        from kairos.skills.marble_run.pieces import create_drop
        piece = create_drop((0, 0, 2), height=0.8)
        exit_z = piece["exit_port"]["position"][2]
        assert exit_z == pytest.approx(1.2, abs=0.01)


class TestTrackPieceRegistry:
    def test_all_pieces_registered(self):
        from kairos.skills.marble_run.pieces import TRACK_PIECES
        assert "straight_track" in TRACK_PIECES
        assert "curved_track" in TRACK_PIECES
        assert "funnel" in TRACK_PIECES
        assert "loop" in TRACK_PIECES
        assert "drop" in TRACK_PIECES

    def test_registry_values_are_callable(self):
        from kairos.skills.marble_run.pieces import TRACK_PIECES
        for name, fn in TRACK_PIECES.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Momentum calculator
# ---------------------------------------------------------------------------

class TestMomentumCalculator:
    def test_no_momentum_needed_downhill(self):
        from kairos.skills.marble_run.momentum_calculator import compute_momentum_required
        # Large drop + short segment — gravity alone provides energy
        required = compute_momentum_required(height_drop=2.0, segment_length=0.5)
        assert required == 0.0

    def test_momentum_needed_uphill(self):
        from kairos.skills.marble_run.momentum_calculator import compute_momentum_required
        # Climbing — definitely needs entry speed
        required = compute_momentum_required(height_drop=-0.5, segment_length=1.0)
        assert required > 0.0

    def test_flat_segment_needs_some_speed(self):
        from kairos.skills.marble_run.momentum_calculator import compute_momentum_required
        # Flat with friction — needs speed
        required = compute_momentum_required(height_drop=0.0, segment_length=2.0)
        assert required > 0.0

    def test_exit_speed_downhill(self):
        from kairos.skills.marble_run.momentum_calculator import compute_exit_speed
        # Starting from rest, downhill 1m over 2m track
        exit_v = compute_exit_speed(0.0, height_drop=1.0, segment_length=2.0)
        assert exit_v > 0.0

    def test_exit_speed_stall(self):
        from kairos.skills.marble_run.momentum_calculator import compute_exit_speed
        # Slow marble going uphill — should stall
        exit_v = compute_exit_speed(0.1, height_drop=-1.0, segment_length=0.5)
        assert exit_v == 0.0

    def test_chain_all_pass(self):
        from kairos.skills.marble_run.momentum_calculator import validate_momentum_chain
        # Steep downhill — marble should gain speed
        segments = [
            {"height_drop": 0.5, "length": 0.5},
            {"height_drop": 0.3, "length": 0.5},
            {"height_drop": 0.2, "length": 0.5},
        ]
        results = validate_momentum_chain(segments, initial_speed=0.0)
        assert len(results) == 3
        assert all(r["passed"] for r in results)

    def test_chain_stalls(self):
        from kairos.skills.marble_run.momentum_calculator import validate_momentum_chain
        # Downhill then steep uphill — marble stalls
        segments = [
            {"height_drop": 0.2, "length": 0.5},
            {"height_drop": -1.0, "length": 0.3},
        ]
        results = validate_momentum_chain(segments, initial_speed=0.0)
        assert len(results) == 2
        assert not results[1]["passed"]


# ---------------------------------------------------------------------------
# Connector validator
# ---------------------------------------------------------------------------

class TestConnectorValidator:
    def test_valid_connection(self):
        from kairos.skills.marble_run.connector_validator import validate_connection
        exit_port = {"position": [1.0, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        entry_port = {"position": [1.01, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        check = validate_connection(exit_port, entry_port, 0, 1)
        assert check.passed
        assert check.gap_m < 0.02

    def test_gap_too_large(self):
        from kairos.skills.marble_run.connector_validator import validate_connection
        exit_port = {"position": [1.0, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        entry_port = {"position": [1.1, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        check = validate_connection(exit_port, entry_port, 0, 1)
        assert not check.passed
        assert "Gap" in check.message

    def test_direction_deviation(self):
        from kairos.skills.marble_run.connector_validator import validate_connection
        exit_port = {"position": [1.0, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        # 90° deviation — should fail (max is 45°)
        entry_port = {"position": [1.0, 0, 0], "direction": [0, 1.0, 0], "diameter": 0.05}
        check = validate_connection(exit_port, entry_port, 0, 1)
        assert not check.passed
        assert "Direction" in check.message

    def test_diameter_mismatch(self):
        from kairos.skills.marble_run.connector_validator import validate_connection
        exit_port = {"position": [1.0, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.05}
        entry_port = {"position": [1.0, 0, 0], "direction": [1.0, 0, 0], "diameter": 0.10}
        check = validate_connection(exit_port, entry_port, 0, 1)
        assert not check.passed
        assert "Diameter" in check.message

    def test_validate_track_empty(self):
        from kairos.skills.marble_run.connector_validator import validate_track
        result = validate_track([])
        assert not result.passed
        assert "No track pieces" in result.issues[0]

    def test_validate_track_connected(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        from kairos.skills.marble_run.connector_validator import validate_track
        # Two connected straight pieces with good downhill
        p1 = create_straight_track((0, 0, 2.0), (1, 0, 1.5))
        p2 = create_straight_track((1, 0, 1.5), (2, 0, 1.0))
        result = validate_track([p1, p2], initial_speed=0.0)
        assert result.total_pieces == 2
        assert result.total_length > 0
        assert len(result.connection_checks) == 1
        assert result.connection_checks[0].passed

    def test_validate_track_disconnected(self):
        from kairos.skills.marble_run.pieces import create_straight_track
        from kairos.skills.marble_run.connector_validator import validate_track
        p1 = create_straight_track((0, 0, 2), (1, 0, 1.5))
        p2 = create_straight_track((5, 0, 1.5), (6, 0, 1.0))  # Big gap
        result = validate_track([p1, p2])
        assert not result.passed
        assert any("Gap" in issue or "Connection" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------

class TestMarbleCameraConfig:
    def test_follow_config_defaults(self):
        from kairos.skills.marble_run.camera_config import marble_follow_config
        cfg = marble_follow_config()
        assert cfg["camera_style"] == "marble_follow"
        assert cfg["camera_follow_distance"] > 0
        assert cfg["camera_follow_distance"] < 1.0  # Closer than domino 1.5m
        assert cfg["camera_tracking_influence"] > 0.8

    def test_follow_config_with_scale(self):
        from kairos.skills.marble_run.camera_config import marble_follow_config
        cfg1 = marble_follow_config(world_scale=1.0)
        cfg2 = marble_follow_config(world_scale=2.0)
        assert cfg2["camera_follow_distance"] > cfg1["camera_follow_distance"]

    def test_overview_config(self):
        from kairos.skills.marble_run.camera_config import marble_race_overview_config
        cfg = marble_race_overview_config()
        assert cfg["camera_style"] == "front_static"

    def test_preset_registry(self):
        from kairos.skills.marble_run.camera_config import MARBLE_CAMERA_PRESETS
        assert "marble_follow" in MARBLE_CAMERA_PRESETS
        assert "race_overview" in MARBLE_CAMERA_PRESETS


# ---------------------------------------------------------------------------
# Marble calibration models
# ---------------------------------------------------------------------------

class TestMarbleCalibrationModels:
    def test_scenario_descriptor_to_nl(self):
        from kairos.calibration.marble_models import (
            MarbleScenarioDescriptor,
            MarbleTrackDescriptor,
            MarbleTrackType,
        )
        scenario = MarbleScenarioDescriptor(
            track=MarbleTrackDescriptor(
                type=MarbleTrackType.CURVED,
                piece_count=6,
                total_length_m=2.5,
                total_height_drop_m=0.8,
            ),
        )
        nl = scenario.to_natural_language()
        assert "curved" in nl.lower()
        assert "6 pieces" in nl
        assert "2.5m" in nl

    def test_scenario_descriptor_to_metadata(self):
        from kairos.calibration.marble_models import (
            MarbleScenarioDescriptor,
            MarbleTrackDescriptor,
            MarbleTrackType,
        )
        scenario = MarbleScenarioDescriptor(
            track=MarbleTrackDescriptor(
                type=MarbleTrackType.SPIRAL_DESCENT,
                piece_count=10,
                total_length_m=3.0,
                total_height_drop_m=2.0,
            ),
        )
        meta = scenario.to_metadata()
        assert meta["content_type"] == "marble_run"
        assert meta["track_type"] == "spiral_descent"
        assert meta["piece_count"] == 10

    def test_correction_factors_apply(self):
        from kairos.calibration.marble_models import (
            MarbleCorrectionFactors,
            MARBLE_BASELINE_PHYSICS,
        )
        corrections = MarbleCorrectionFactors(marble_friction=1.2)
        applied = corrections.apply_to_baseline()
        assert applied["marble_friction"] == pytest.approx(
            MARBLE_BASELINE_PHYSICS["marble_friction"] * 1.2, abs=0.001
        )

    def test_bootstrap_scenarios_count(self):
        from kairos.calibration.marble_models import MARBLE_BOOTSTRAP_SCENARIOS
        assert len(MARBLE_BOOTSTRAP_SCENARIOS) == 10

    def test_bootstrap_scenarios_variety(self):
        from kairos.calibration.marble_models import MARBLE_BOOTSTRAP_SCENARIOS
        track_types = {s.track.type.value for s in MARBLE_BOOTSTRAP_SCENARIOS}
        # Should have multiple distinct track types
        assert len(track_types) >= 5


# ---------------------------------------------------------------------------
# Config marble defaults
# ---------------------------------------------------------------------------

class TestConfigMarbleDefaults:
    def test_marble_radius_default(self):
        from kairos.config import Settings
        s = Settings()
        assert s.marble_radius_m == 0.04

    def test_marble_mass_default(self):
        from kairos.config import Settings
        s = Settings()
        assert s.marble_mass_kg == 0.028

    def test_marble_scene_bounds(self):
        from kairos.config import Settings
        s = Settings()
        assert s.scene_bounds_marble_m == [20.0, 20.0, 10.0]

    def test_marble_course_length_target(self):
        from kairos.config import Settings
        s = Settings()
        assert s.course_length_target_marble_m == 15.0
