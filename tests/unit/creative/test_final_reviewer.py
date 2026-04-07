"""Tests for the final reviewer agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kairos.pipelines.domino.creative.final_reviewer import (
    FinalReviewer,
    _check_camera_coverage,
    _check_chain_coverage,
    _check_scene_coherence,
    _earliest_responsible_agent,
)
from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CameraKeyframe,
    CameraOutput,
    ConnectorOutput,
    ConnectorType,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    LightingConfig,
    ObjectRole,
    OcclusionEvent,
    PathOutput,
    PathSegment,
    PlacedObject,
    ResolvedConnector,
    ReviewIssue,
    SceneManifest,
    SegmentType,
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


def _manifest() -> SceneManifest:
    return SceneManifest(
        theme="kitchen",
        narrative="Dominoes topple.",
        environment=_env(),
        objects=[
            PlacedObject(
                asset_id="table_01", name="Table",
                position=(1.0, 2.0, 0.8),
                role=ObjectRole.FUNCTIONAL, surface_name="top",
            ),
            PlacedObject(
                asset_id="shelf_01", name="Shelf",
                position=(-1.0, -2.0, 1.5),
                role=ObjectRole.FUNCTIONAL, surface_name="top",
            ),
        ],
        domino_count=200,
    )


def _path() -> PathOutput:
    return PathOutput(
        total_length_estimate=5.0,
        segments=[
            PathSegment(
                id="s1", type=SegmentType.GROUND_LEVEL,
                waypoints=[_wp(0, 0, 0), _wp(1, 0, 0)],
                surface_ref="table_01.top",
            ),
            PathSegment(
                id="s2", type=SegmentType.HEIGHT_TRANSITION_UP,
                needs_connector=True, connector_hint="ramp",
                from_height=0.0, to_height=0.8,
                surface_ref="shelf_01.top",
            ),
            PathSegment(
                id="s3", type=SegmentType.FLAT_SURFACE,
                waypoints=[_wp(1, 0, 0.8), _wp(2, 0, 0.8)],
            ),
        ],
        domino_count=200,
    )


def _connectors() -> ConnectorOutput:
    return ConnectorOutput(
        connectors=[
            ResolvedConnector(
                id="conn_s2", for_segment="s2",
                type=ConnectorType.RAMP,
                generated_waypoints=[_wp(1, 0, 0), _wp(1, 0, 0.4), _wp(1, 0, 0.8)],
            ),
        ],
        complete_path_waypoints=[_wp(0, i * 0.3, 0) for i in range(10)],
        segment_types=["ground_level", "ramp", "flat_surface"],
    )


def _camera(total_frames: int = 150) -> CameraOutput:
    kfs = [
        CameraKeyframe(
            frame=f * 5,
            position=(3.0, f * 0.3, 4.0),
            look_target=(0.0, f * 0.3, 0.0),
        )
        for f in range(total_frames // 5)
    ]
    return CameraOutput(keyframes=kfs, total_frames=total_frames)


# ─── _check_chain_coverage ───────────────────────────────────────────


class TestCheckChainCoverage:
    def test_all_resolved(self):
        issues = _check_chain_coverage(_path(), _connectors())
        blocking = [i for i in issues if i.severity == "blocking"]
        assert len(blocking) == 0

    def test_missing_connector(self):
        """Connector for s2 is missing."""
        empty_conn = ConnectorOutput(
            connectors=[],
            complete_path_waypoints=[_wp()],
            segment_types=[],
        )
        issues = _check_chain_coverage(_path(), empty_conn)
        blocking = [i for i in issues if i.severity == "blocking"]
        assert len(blocking) >= 1
        assert any(
            i.attributed_to == AgentRole.CONNECTOR for i in blocking
        )

    def test_too_few_waypoints(self):
        """Complete path with < 4 waypoints."""
        short_conn = ConnectorOutput(
            connectors=[
                ResolvedConnector(
                    id="c1", for_segment="s2",
                    type=ConnectorType.RAMP,
                ),
            ],
            complete_path_waypoints=[_wp(), _wp(1, 0, 0)],
            segment_types=["ground"],
        )
        issues = _check_chain_coverage(_path(), short_conn)
        blocking = [i for i in issues if i.severity == "blocking"]
        assert any(i.attributed_to == AgentRole.PATH_SETTER for i in blocking)


# ─── _check_camera_coverage ──────────────────────────────────────────


class TestCheckCameraCoverage:
    def test_good_camera(self):
        issues = _check_camera_coverage(_camera())
        assert len(issues) == 0

    def test_no_camera(self):
        issues = _check_camera_coverage(None)
        assert len(issues) == 1
        assert issues[0].severity == "blocking"

    def test_too_few_keyframes(self):
        cam = CameraOutput(
            keyframes=[CameraKeyframe(frame=0, position=(0, 0, 0), look_target=(1, 0, 0))],
            total_frames=100,
        )
        issues = _check_camera_coverage(cam)
        assert any(i.severity == "blocking" for i in issues)

    def test_excessive_occlusion(self):
        """More than 10% of frames occluded."""
        cam = CameraOutput(
            keyframes=[
                CameraKeyframe(frame=f * 5, position=(3, f, 4), look_target=(0, f, 0))
                for f in range(20)
            ],
            occlusion_events=[
                OcclusionEvent(frame_start=0, frame_end=20, occluder="wall"),
            ],
            total_frames=100,
        )
        issues = _check_camera_coverage(cam)
        assert any("occlusion" in i.description.lower() for i in issues)


# ─── _check_scene_coherence ──────────────────────────────────────────


class TestCheckSceneCoherence:
    def test_all_functional_visited(self):
        """Path references all functional objects → no issues."""
        issues = _check_scene_coherence(_manifest(), _path())
        # Should be no blocking issues (could have warnings)
        blocking = [i for i in issues if i.severity == "blocking"]
        assert len(blocking) == 0

    def test_unvisited_functional_objects(self):
        """Path doesn't reference any surfaces → warning."""
        path = PathOutput(
            total_length_estimate=5.0,
            segments=[
                PathSegment(
                    id="s1", type=SegmentType.GROUND_LEVEL,
                    waypoints=[_wp(0, 0, 0), _wp(1, 0, 0)],
                    # no surface_ref
                ),
            ],
            domino_count=200,
        )
        issues = _check_scene_coherence(_manifest(), path)
        assert any(i.severity == "warning" for i in issues)
        assert any(i.attributed_to == AgentRole.PATH_SETTER for i in issues)


# ─── _earliest_responsible_agent ─────────────────────────────────────


class TestEarliestResponsibleAgent:
    def test_single_blocking_issue(self):
        issues = [
            ReviewIssue(
                description="x", attributed_to=AgentRole.CONNECTOR,
                severity="blocking",
            ),
        ]
        assert _earliest_responsible_agent(issues) == AgentRole.CONNECTOR

    def test_multiple_issues_earliest_wins(self):
        issues = [
            ReviewIssue(
                description="a", attributed_to=AgentRole.CAMERA_ROUTER,
                severity="blocking",
            ),
            ReviewIssue(
                description="b", attributed_to=AgentRole.PATH_SETTER,
                severity="blocking",
            ),
        ]
        assert _earliest_responsible_agent(issues) == AgentRole.PATH_SETTER

    def test_warnings_ignored(self):
        issues = [
            ReviewIssue(
                description="w", attributed_to=AgentRole.SET_DESIGNER,
                severity="warning",
            ),
        ]
        assert _earliest_responsible_agent(issues) is None

    def test_no_issues(self):
        assert _earliest_responsible_agent([]) is None


# ─── FinalReviewer integration ───────────────────────────────────────


class TestFinalReviewer:
    @pytest.mark.asyncio
    async def test_passes_when_all_good(self):
        reviewer = FinalReviewer()
        result = await reviewer.review(
            _manifest(), _path(), _connectors(), _camera(),
        )
        assert result.passed
        assert result.cascade_from is None

    @pytest.mark.asyncio
    async def test_fails_on_missing_connector(self):
        reviewer = FinalReviewer()
        empty_conn = ConnectorOutput(
            connectors=[],
            complete_path_waypoints=[_wp()],
            segment_types=[],
        )
        result = await reviewer.review(
            _manifest(), _path(), empty_conn, _camera(),
        )
        assert not result.passed
        assert result.cascade_from is not None

    @pytest.mark.asyncio
    async def test_fails_on_no_camera(self):
        reviewer = FinalReviewer()
        result = await reviewer.review(
            _manifest(), _path(), _connectors(), None,
        )
        assert not result.passed
        assert result.cascade_from == AgentRole.CAMERA_ROUTER

    @pytest.mark.asyncio
    async def test_vlm_review_not_called_without_frames(self):
        """No rendered_frames means no VLM call."""
        reviewer = FinalReviewer()
        with patch.object(reviewer, "_vlm_review", new_callable=AsyncMock) as mock_vlm:
            await reviewer.review(
                _manifest(), _path(), _connectors(), _camera(),
            )
            mock_vlm.assert_not_called()

    @pytest.mark.asyncio
    async def test_vlm_review_called_with_frames(self):
        """When rendered_frames provided, VLM review runs."""
        reviewer = FinalReviewer()
        with patch.object(
            reviewer, "_vlm_review",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_vlm:
            await reviewer.review(
                _manifest(), _path(), _connectors(), _camera(),
                rendered_frames=["/tmp/frame_001.png"],
            )
            mock_vlm.assert_called_once()
