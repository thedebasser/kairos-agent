"""Tests for the skill library — pure-math path/connector primitives
and domino placement data functions.

These run without Blender.
"""

from __future__ import annotations

import math

import pytest

from kairos.skills.contracts import PathResult, ConnectorResult, Vector3


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------

class TestVector3:
    def test_distance_to(self):
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert abs(a.distance_to(b) - 5.0) < 1e-9

    def test_add_sub(self):
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        c = a + b
        assert (c.x, c.y, c.z) == (5, 7, 9)
        d = b - a
        assert (d.x, d.y, d.z) == (3, 3, 3)

    def test_scalar_multiply(self):
        v = Vector3(1, 2, 3)
        w = v * 2
        assert (w.x, w.y, w.z) == (2, 4, 6)
        w2 = 3 * v
        assert (w2.x, w2.y, w2.z) == (3, 6, 9)

    def test_normalized(self):
        v = Vector3(3, 0, 4)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 1e-9

    def test_zero_normalized(self):
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n.length() == 0.0

    def test_to_from_tuple(self):
        v = Vector3(1.5, 2.5, 3.5)
        assert v.to_tuple() == (1.5, 2.5, 3.5)
        v2 = Vector3.from_tuple((1.5, 2.5, 3.5))
        assert v2 == v


# ---------------------------------------------------------------------------
# Path primitives
# ---------------------------------------------------------------------------

class TestStraightLine:
    def test_basic(self):
        from kairos.skills.shared.paths.straight_line import create_straight_line
        result = create_straight_line(Vector3(0, 0, 0), Vector3(1, 0, 0))
        assert isinstance(result, PathResult)
        assert result.segment_type == "straight_line"
        assert abs(result.total_length - 1.0) < 0.01
        assert result.point_count >= 2
        # First and last waypoints match
        assert result.waypoints[0].distance_to(Vector3(0, 0, 0)) < 1e-6
        assert result.waypoints[-1].distance_to(Vector3(1, 0, 0)) < 0.05

    def test_zero_length(self):
        from kairos.skills.shared.paths.straight_line import create_straight_line
        result = create_straight_line(Vector3(5, 5, 5), Vector3(5, 5, 5))
        assert result.total_length == 0.0
        assert result.point_count == 1

    def test_vertical(self):
        from kairos.skills.shared.paths.straight_line import create_straight_line
        result = create_straight_line(Vector3(0, 0, 0), Vector3(0, 0, 2))
        assert abs(result.total_length - 2.0) < 0.01
        assert abs(result.gradients[0]) == 90.0


class TestArc:
    def test_basic(self):
        from kairos.skills.shared.paths.arc import create_arc
        result = create_arc(
            center=Vector3(0, 0, 0),
            radius=1.0,
            start_angle=0.0,
            end_angle=math.pi / 2,
        )
        assert isinstance(result, PathResult)
        assert result.segment_type == "arc"
        assert result.point_count >= 2
        # Arc length ≈ πr/2 ≈ 1.5708
        assert abs(result.total_length - math.pi / 2) < 0.1


class TestSpiral:
    def test_basic(self):
        from kairos.skills.shared.paths.spiral import create_spiral
        result = create_spiral(
            center=Vector3(0, 0, 0),
            radius=1.0,
            turns=2.0,
            start_height=0.0,
            end_height=1.0,
        )
        assert isinstance(result, PathResult)
        assert result.segment_type == "spiral"
        assert result.point_count >= 10
        # Start near ground, end near height 1
        assert abs(result.waypoints[0].z - 0.0) < 0.01
        assert abs(result.waypoints[-1].z - 1.0) < 0.05


class TestSCurve:
    def test_basic(self):
        from kairos.skills.shared.paths.s_curve import create_s_curve
        result = create_s_curve(
            start=Vector3(0, 0, 0),
            end=Vector3(2, 0, 0),
            amplitude=0.5,
        )
        assert isinstance(result, PathResult)
        assert result.segment_type == "s_curve"
        assert result.point_count >= 2


class TestStaircasePath:
    def test_basic(self):
        from kairos.skills.shared.paths.staircase import create_staircase_path
        result = create_staircase_path(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 0, 1),
            step_count=5,
        )
        assert isinstance(result, PathResult)
        assert result.segment_type == "staircase"
        assert result.point_count >= 5


# ---------------------------------------------------------------------------
# Connector primitives
# ---------------------------------------------------------------------------

class TestRamp:
    def test_basic(self):
        from kairos.skills.shared.connectors.ramp import create_ramp
        result = create_ramp(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 0, 1),
            width=0.3,
        )
        assert isinstance(result, ConnectorResult)
        assert result.connector_type == "ramp"
        assert result.total_length > 0


class TestSpiralRamp:
    def test_basic(self):
        from kairos.skills.shared.connectors.spiral_ramp import create_spiral_ramp
        result = create_spiral_ramp(
            center=Vector3(0, 0, 0),
            radius=0.3,
            turns=3,
            start_height=2.0,
            end_height=0.0,
        )
        assert isinstance(result, ConnectorResult)
        assert result.connector_type == "spiral_ramp"
        assert result.total_length > 0
        assert len(result.waypoints) >= 10


class TestStaircase:
    def test_basic(self):
        from kairos.skills.shared.connectors.staircase import create_staircase
        result = create_staircase(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 0, 1),
            step_count=10,
        )
        assert isinstance(result, ConnectorResult)
        assert result.connector_type == "staircase"


class TestPlatform:
    def test_basic(self):
        from kairos.skills.shared.connectors.platform import create_platform
        result = create_platform(
            center=Vector3(0, 0, 1),
            width=0.5,
            depth=0.5,
        )
        assert isinstance(result, ConnectorResult)
        assert result.connector_type == "platform"


class TestPlankBridge:
    def test_basic(self):
        from kairos.skills.shared.connectors.plank_bridge import create_plank_bridge
        result = create_plank_bridge(
            start=Vector3(0, 0, 1),
            end=Vector3(1, 0, 0.8),
            width=0.2,
        )
        assert isinstance(result, ConnectorResult)
        assert result.connector_type == "plank_bridge"


# ---------------------------------------------------------------------------
# Domino skills
# ---------------------------------------------------------------------------

class TestSpacingCalculator:
    def test_default(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        s = compute_spacing()
        # Base: 0.08 * 0.35 = 0.028
        assert abs(s - 0.028) < 0.001

    def test_uphill_tighter(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        flat = compute_spacing(gradient_deg=0.0)
        uphill = compute_spacing(gradient_deg=15.0)
        assert uphill < flat

    def test_downhill_wider(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        flat = compute_spacing(gradient_deg=0.0)
        downhill = compute_spacing(gradient_deg=-15.0)
        assert downhill > flat

    def test_curvature_tighter(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        straight = compute_spacing(curvature=0.0)
        curved = compute_spacing(curvature=0.5)
        assert curved < straight

    def test_clamp(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        # Extreme uphill should be clamped to min
        s = compute_spacing(gradient_deg=80.0)
        assert s >= 0.08 * 0.15  # min_spacing_ratio

    def test_calibration_correction(self):
        from kairos.skills.domino.spacing_calculator import compute_spacing
        base = compute_spacing()
        corrected = compute_spacing(calibration_correction=1.2)
        assert abs(corrected - base * 1.2) < 0.001


class TestPlaceDomino:
    def test_returns_params(self):
        from kairos.skills.domino.place_domino import place_domino
        params = place_domino((0, 0, 0.04), (0, 0, 90))
        assert params["type"] == "domino"
        assert params["position"] == [0, 0, 0.04]
        assert params["rotation_deg"] == [0, 0, 90]
        assert params["dimensions"]["height"] == 0.08

    def test_custom_dims(self):
        from kairos.skills.domino.place_domino import place_domino
        params = place_domino((0, 0, 0), (0, 0, 0), domino_dims=(0.1, 0.05, 0.01))
        assert params["dimensions"]["height"] == 0.1


class TestSetupTrigger:
    def test_default(self):
        from kairos.skills.domino.setup_trigger import setup_trigger
        t = setup_trigger()
        assert t["first_domino_index"] == 0
        assert t["method"] == "velocity"


class TestSizeGradient:
    def test_uniform(self):
        from kairos.skills.domino.size_gradient import compute_size_gradient
        assert compute_size_gradient(0, 100) == 1.0
        assert compute_size_gradient(99, 100) == 1.0

    def test_growing(self):
        from kairos.skills.domino.size_gradient import compute_size_gradient
        first = compute_size_gradient(0, 100, start_scale=0.5, end_scale=2.0)
        last = compute_size_gradient(99, 100, start_scale=0.5, end_scale=2.0)
        assert abs(first - 0.5) < 0.01
        assert abs(last - 2.0) < 0.01


class TestApplyRigidBody:
    def test_defaults(self):
        from kairos.skills.domino.apply_rigid_body import apply_rigid_body
        rb = apply_rigid_body()
        assert rb["type"] == "ACTIVE"
        assert rb["mass"] == 0.02
        assert rb["collision_shape"] == "BOX"


# ---------------------------------------------------------------------------
# Surface primitives
# ---------------------------------------------------------------------------

class TestFlatSurface:
    def test_returns_params(self):
        from kairos.skills.shared.surfaces.flat import create_flat_surface
        p = create_flat_surface((0, 0, 0), 2.0, 1.0)
        assert p["type"] == "flat_surface"
        assert p["width"] == 2.0


class TestRampSurface:
    def test_returns_params(self):
        from kairos.skills.shared.surfaces.ramp_surface import create_ramp_surface
        p = create_ramp_surface((0, 0, 0), (1, 0, 1), 0.5)
        assert p["type"] == "ramp_surface"
        assert p["slope_length"] > 0


# ---------------------------------------------------------------------------
# Environment system
# ---------------------------------------------------------------------------

class TestLightingPresets:
    def test_get_preset(self):
        from kairos.skills.shared.environment.lighting_presets import get_lighting_preset
        p = get_lighting_preset("warm_indoor")
        assert p is not None
        assert p.environment_type == "indoor"

    def test_unknown_preset(self):
        from kairos.skills.shared.environment.lighting_presets import get_lighting_preset
        assert get_lighting_preset("nonexistent") is None

    def test_find_by_theme(self):
        from kairos.skills.shared.environment.lighting_presets import find_presets_by_theme
        presets = find_presets_by_theme("dinner")
        assert len(presets) >= 1


class TestSkybox:
    def test_scan_empty_dir(self, tmp_path):
        from kairos.skills.shared.environment.skybox import scan_hdris
        entries = scan_hdris(tmp_path)
        assert entries == []

    def test_scan_hdr_file(self, tmp_path):
        from kairos.skills.shared.environment.skybox import scan_hdris
        (tmp_path / "sunset_park.hdr").touch()
        entries = scan_hdris(tmp_path)
        assert len(entries) == 1
        assert "outdoor" in entries[0].themes

    def test_find_by_theme(self, tmp_path):
        from kairos.skills.shared.environment.skybox import scan_hdris, find_hdris_by_theme
        (tmp_path / "forest_clearing.hdr").touch()
        (tmp_path / "studio_light.exr").touch()
        entries = scan_hdris(tmp_path)
        forest = find_hdris_by_theme(entries, "forest")
        assert len(forest) >= 1


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

class TestCatalogue:
    def test_round_trip(self, tmp_path):
        from kairos.skills.catalogue import AssetEntry, load_catalogue, save_catalogue

        entry = AssetEntry(
            id="test_table",
            name="Test Table",
            source="polyhaven",
            file="assets/models/table.blend",
            license="CC0",
            themes=["dining", "indoor"],
            category="furniture",
            dimensions=(1.2, 0.6, 0.75),
        )
        path = tmp_path / "catalogue.yaml"
        save_catalogue([entry], path)
        loaded = load_catalogue(path)
        assert len(loaded) == 1
        assert loaded[0].id == "test_table"
        assert loaded[0].dimensions == (1.2, 0.6, 0.75)
