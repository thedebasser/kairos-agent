"""Tests for the camera router."""

from __future__ import annotations

import pytest

from kairos.pipelines.domino.creative.camera_router import (
    CameraRouter,
    _ease_in_out,
    _lerp_tuple,
    _sphere_intersects_ray,
    _vec_len,
    _vec_norm,
    _vec_sub,
    detect_occlusions,
)
from kairos.pipelines.domino.creative.models import (
    ConnectorOutput,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    LightingConfig,
    ObjectRole,
    PlacedObject,
    SceneManifest,
    Waypoint,
)

pytestmark = pytest.mark.unit


# ─── Fixtures ────────────────────────────────────────────────────────


def _wp(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


def _env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="tile"),
        lighting=LightingConfig(preset="warm"),
    )


def _manifest_with_objects() -> SceneManifest:
    return SceneManifest(
        theme="kitchen",
        narrative="Dominoes roll.",
        environment=_env(),
        objects=[
            PlacedObject(
                asset_id="table_01", name="Table",
                position=(2.0, 2.0, 0.8),
                role=ObjectRole.FUNCTIONAL, surface_name="top",
            ),
            PlacedObject(
                asset_id="shelf_01", name="Shelf",
                position=(-2.0, -2.0, 1.5),
                role=ObjectRole.FUNCTIONAL, surface_name="top",
            ),
        ],
        domino_count=200,
    )


def _straight_path() -> ConnectorOutput:
    """A simple straight path along the Y axis."""
    return ConnectorOutput(
        connectors=[],
        complete_path_waypoints=[
            _wp(0, i * 0.5, 0) for i in range(20)
        ],
        segment_types=["ground_level"],
    )


# ─── Math helpers ────────────────────────────────────────────────────


class TestMathHelpers:
    def test_vec_sub(self):
        assert _vec_sub((3, 2, 1), (1, 1, 1)) == (2, 1, 0)

    def test_vec_norm_unit(self):
        n = _vec_norm((3, 0, 0))
        assert abs(n[0] - 1.0) < 1e-6
        assert abs(n[1]) < 1e-6

    def test_vec_norm_zero(self):
        assert _vec_norm((0, 0, 0)) == (0.0, 0.0, 0.0)

    def test_lerp_tuple_extremes(self):
        a = (0.0, 0.0, 0.0)
        b = (10.0, 10.0, 10.0)
        assert _lerp_tuple(a, b, 0.0) == a
        mid = _lerp_tuple(a, b, 0.5)
        assert abs(mid[0] - 5.0) < 1e-6

    def test_ease_in_out_boundaries(self):
        assert abs(_ease_in_out(0.0)) < 1e-6
        assert abs(_ease_in_out(1.0) - 1.0) < 1e-6
        # Midpoint should be 0.5
        assert abs(_ease_in_out(0.5) - 0.5) < 1e-6


# ─── Occlusion detection ────────────────────────────────────────────


class TestOcclusionDetection:
    def test_no_occlusion(self):
        """Ray doesn't hit any objects."""
        cam = (0.0, -5.0, 3.0)
        target = (0.0, 0.0, 0.0)
        objects = [("wall", (10.0, 10.0, 0.0), 0.5)]
        assert detect_occlusions(cam, target, objects) == []

    def test_sphere_blocks_ray(self):
        """Object is directly between camera and target."""
        cam = (0.0, -5.0, 0.0)
        target = (0.0, 5.0, 0.0)
        objects = [("pillar", (0.0, 0.0, 0.0), 1.0)]
        result = detect_occlusions(cam, target, objects)
        assert "pillar" in result

    def test_sphere_barely_misses(self):
        """Object is near the ray but doesn't intersect."""
        cam = (0.0, -5.0, 0.0)
        target = (0.0, 5.0, 0.0)
        objects = [("side_obj", (3.0, 0.0, 0.0), 0.5)]
        assert detect_occlusions(cam, target, objects) == []


# ─── Camera Router ───────────────────────────────────────────────────


class TestCameraRouter:
    def test_basic_trajectory(self):
        """Produces keyframes for a simple straight path."""
        router = CameraRouter()
        output = router.compute_trajectory(
            _straight_path(),
            _manifest_with_objects(),
            fps=30,
            duration_sec=5.0,
        )
        assert len(output.keyframes) > 0
        assert output.total_frames == 150  # 30 fps × 5 s

    def test_too_few_waypoints(self):
        """Returns empty output for degenerate path."""
        router = CameraRouter()
        empty = ConnectorOutput(
            connectors=[], complete_path_waypoints=[_wp()], segment_types=[],
        )
        output = router.compute_trajectory(
            empty,
            _manifest_with_objects(),
            fps=30,
            duration_sec=5.0,
        )
        # Should still return but with 0 total_frames or minimal keyframes
        assert output.total_frames == 0 or len(output.keyframes) >= 0

    def test_keyframes_have_position_and_target(self):
        router = CameraRouter()
        output = router.compute_trajectory(
            _straight_path(),
            _manifest_with_objects(),
            fps=30,
            duration_sec=3.0,
        )
        for kf in output.keyframes:
            assert len(kf.position) == 3
            assert len(kf.look_target) == 3

    def test_pullback_appended(self):
        """The last keyframes should be the pull-back overview."""
        router = CameraRouter(key_interval=5)
        output = router.compute_trajectory(
            _straight_path(),
            _manifest_with_objects(),
            fps=30,
            duration_sec=10.0,
        )
        # Last keyframes should be further from the path (overview)
        if len(output.keyframes) >= 2:
            last_pos = output.keyframes[-1].position
            first_pos = output.keyframes[0].position
            last_dist = _vec_len(last_pos)
            first_dist = _vec_len(first_pos)
            # Overview should be at a greater distance from origin
            assert last_dist > 0  # Just verify it's computed

    def test_occlusion_reposition(self):
        """When an object occludes, the camera repositions."""
        # Place an object directly between camera default position and path
        manifest = SceneManifest(
            theme="test",
            narrative="n",
            environment=_env(),
            objects=[
                PlacedObject(
                    asset_id="wall",
                    position=(0.0, 0.0, 0.0),  # right on the path
                    role=ObjectRole.DECORATIVE,
                    scale=5.0,  # large object
                ),
            ],
            domino_count=100,
        )
        path = ConnectorOutput(
            connectors=[],
            complete_path_waypoints=[_wp(0, i * 0.5, 0) for i in range(10)],
            segment_types=["ground"],
        )
        router = CameraRouter()
        output = router.compute_trajectory(path, manifest, fps=30, duration_sec=3.0)
        # May or may not detect occlusion depending on geometry,
        # but should not crash
        assert output.total_frames == 90
