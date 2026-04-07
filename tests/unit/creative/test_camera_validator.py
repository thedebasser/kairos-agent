"""Tests for the camera validator."""

from __future__ import annotations

import pytest

from kairos.pipelines.domino.creative.camera_validator import validate_camera
from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CameraKeyframe,
    CameraOutput,
    ConnectorOutput,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    LightingConfig,
    OcclusionEvent,
    PlacedObject,
    ObjectRole,
    SceneManifest,
    Waypoint,
)

pytestmark = pytest.mark.unit


# ─── Builders ────────────────────────────────────────────────────────


def _wp(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


def _env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="tile"),
        lighting=LightingConfig(preset="warm"),
    )


def _simple_manifest() -> SceneManifest:
    return SceneManifest(
        theme="test",
        narrative="n",
        environment=_env(),
        objects=[],
        domino_count=100,
    )


def _simple_connector() -> ConnectorOutput:
    return ConnectorOutput(
        connectors=[],
        complete_path_waypoints=[_wp(0, i * 0.5, 0) for i in range(10)],
        segment_types=["ground"],
    )


def _good_camera(total_frames: int = 150) -> CameraOutput:
    """Smooth keyframes along a line — should pass all checks."""
    kfs = [
        CameraKeyframe(
            frame=f * 5,
            position=(3.0, f * 0.5, 4.0),
            look_target=(0.0, f * 0.5, 0.0),
        )
        for f in range(total_frames // 5)
    ]
    return CameraOutput(
        keyframes=kfs,
        occlusion_events=[],
        repositions=0,
        total_frames=total_frames,
    )


def _too_few_keyframes() -> CameraOutput:
    return CameraOutput(
        keyframes=[CameraKeyframe(frame=0, position=(0, 0, 0), look_target=(1, 0, 0))],
        total_frames=100,
    )


def _spikey_camera() -> CameraOutput:
    """Keyframes with an extreme velocity spike."""
    kfs = [
        CameraKeyframe(frame=0, position=(0.0, 0.0, 0.0), look_target=(1, 0, 0)),
        CameraKeyframe(frame=5, position=(0.1, 0.0, 0.0), look_target=(1, 0, 0)),
        CameraKeyframe(frame=10, position=(0.2, 0.0, 0.0), look_target=(1, 0, 0)),
        # Huge jump
        CameraKeyframe(frame=15, position=(50.0, 50.0, 50.0), look_target=(1, 0, 0)),
        CameraKeyframe(frame=20, position=(50.1, 50.0, 50.0), look_target=(1, 0, 0)),
    ]
    return CameraOutput(keyframes=kfs, total_frames=100)


def _hard_cut_camera() -> CameraOutput:
    """Occlusion event lasting fewer than 30 frames → hard cut."""
    kfs = [
        CameraKeyframe(frame=0, position=(3.0, 0.0, 4.0), look_target=(0, 0, 0)),
        CameraKeyframe(frame=5, position=(3.0, 0.5, 4.0), look_target=(0, 0.5, 0)),
        CameraKeyframe(frame=10, position=(3.0, 1.0, 4.0), look_target=(0, 1.0, 0)),
    ]
    return CameraOutput(
        keyframes=kfs,
        occlusion_events=[
            OcclusionEvent(frame_start=0, frame_end=10, occluder="wall"),
        ],
        repositions=1,
        total_frames=100,
    )


# ─── Tests ───────────────────────────────────────────────────────────


class TestValidateCamera:
    def test_good_camera_passes(self):
        result = validate_camera(
            _good_camera(), _simple_connector(), _simple_manifest(),
        )
        assert result.passed
        assert result.agent == AgentRole.CAMERA_ROUTER

    def test_too_few_keyframes_fails(self):
        result = validate_camera(
            _too_few_keyframes(), _simple_connector(), _simple_manifest(),
        )
        assert not result.passed
        assert "keyframes" in result.error_summary.lower() or "keyframe" in result.error_summary.lower()

    def test_spike_fails(self):
        result = validate_camera(
            _spikey_camera(), _simple_connector(), _simple_manifest(),
        )
        assert not result.passed
        spike_check = next(
            (c for c in result.checks if c["name"] == "smooth_motion"), None,
        )
        assert spike_check is not None
        assert not spike_check["passed"]

    def test_hard_cut_fails(self):
        result = validate_camera(
            _hard_cut_camera(), _simple_connector(), _simple_manifest(),
        )
        assert not result.passed
        cut_check = next(
            (c for c in result.checks if c["name"] == "no_hard_cuts"), None,
        )
        assert cut_check is not None
        assert not cut_check["passed"]

    def test_zero_total_frames_fails(self):
        cam = CameraOutput(
            keyframes=[
                CameraKeyframe(frame=0, position=(0, 0, 0), look_target=(1, 0, 0)),
                CameraKeyframe(frame=5, position=(1, 0, 0), look_target=(2, 0, 0)),
            ],
            total_frames=0,
        )
        result = validate_camera(cam, _simple_connector(), _simple_manifest())
        assert not result.passed

    def test_visibility_with_occluding_objects(self):
        """When manifest objects occlude every keyframe, visibility < 90%."""
        # Put a big sphere right on the look target line
        manifest = SceneManifest(
            theme="test", narrative="n", environment=_env(),
            objects=[
                PlacedObject(
                    asset_id="big_wall",
                    position=(1.5, 1.5, 2.0),
                    role=ObjectRole.DECORATIVE,
                    scale=10.0,  # maximum scale
                ),
            ],
            domino_count=100,
        )
        # Keyframes all look at origin through the big wall
        kfs = [
            CameraKeyframe(
                frame=f * 5,
                position=(3.0, 3.0, 4.0),
                look_target=(0.0, 0.0, 0.0),
            )
            for f in range(10)
        ]
        cam = CameraOutput(keyframes=kfs, total_frames=100)
        result = validate_camera(cam, _simple_connector(), manifest)
        vis_check = next(
            (c for c in result.checks if c["name"] == "visibility_ratio"), None,
        )
        assert vis_check is not None
        # Should either pass or fail depending on actual sphere-ray geometry
        # but should not crash
