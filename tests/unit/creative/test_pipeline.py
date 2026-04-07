"""Tests for the creative pipeline cascade re-run logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CameraKeyframe,
    CameraOutput,
    ConnectorOutput,
    ConnectorType,
    EnvironmentSpec,
    EnvironmentType,
    FinalReviewResult,
    GroundConfig,
    LightingConfig,
    ObjectRole,
    PathOutput,
    PathSegment,
    PlacedObject,
    ResolvedConnector,
    ReviewIssue,
    SceneManifest,
    SegmentType,
    StepValidationResult,
    Waypoint,
)
from kairos.pipelines.domino.creative.pipeline import (
    CreativePipeline,
    PER_AGENT_MAX,
    PIPELINE_TOTAL_MAX,
)

pytestmark = pytest.mark.unit


# ─── Test data builders ──────────────────────────────────────────────


def _wp(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


def _env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="tile"),
        lighting=LightingConfig(preset="warm"),
    )


def _good_manifest() -> SceneManifest:
    """Passes all scene validation checks."""
    return SceneManifest(
        theme="kitchen",
        narrative="Dominoes roll.",
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


def _bad_manifest() -> SceneManifest:
    """Fails scene validation — only 1 functional object."""
    return SceneManifest(
        theme="bad",
        narrative="n",
        environment=_env(),
        objects=[
            PlacedObject(
                asset_id="t1", position=(0, 0, 0.5),
                role=ObjectRole.FUNCTIONAL, surface_name="top",
            ),
        ],
        domino_count=100,
    )


def _good_path() -> PathOutput:
    return PathOutput(
        total_length_estimate=5.0,
        segments=[
            PathSegment(
                id="s1", type=SegmentType.GROUND_LEVEL,
                waypoints=[_wp(0, 0, 0), _wp(1, 0, 0)],
            ),
            PathSegment(
                id="s2", type=SegmentType.HEIGHT_TRANSITION_UP,
                needs_connector=True, connector_hint="ramp",
                from_height=0.0, to_height=0.8,
            ),
            PathSegment(
                id="s3", type=SegmentType.FLAT_SURFACE,
                waypoints=[_wp(1, 0, 0.8), _wp(2, 0, 0.8)],
            ),
        ],
        domino_count=200,
    )


def _bad_path() -> PathOutput:
    """Fails path validation — gradient > 15°."""
    return PathOutput(
        total_length_estimate=5.0,
        segments=[
            PathSegment(
                id="s1", type=SegmentType.GROUND_LEVEL,
                waypoints=[_wp(0, 0, 0), _wp(1, 0, 0)],
            ),
            PathSegment(
                id="s2", type=SegmentType.HEIGHT_TRANSITION_UP,
                needs_connector=True, from_height=0.0, to_height=0.8,
                gradient=25.0,
            ),
        ],
        domino_count=200,
    )


def _good_connectors() -> ConnectorOutput:
    return ConnectorOutput(
        connectors=[
            ResolvedConnector(
                id="conn_s2", for_segment="s2",
                type=ConnectorType.RAMP,
                generated_waypoints=[_wp(1, 0, 0), _wp(1, 0, 0.4), _wp(1, 0, 0.8)],
            ),
        ],
        complete_path_waypoints=[
            _wp(0, 0, 0), _wp(1, 0, 0),
            _wp(1, 0, 0), _wp(1, 0, 0.4), _wp(1, 0, 0.8),
            _wp(1, 0, 0.8), _wp(2, 0, 0.8),
        ],
        segment_types=["ground_level", "ramp", "flat_surface"],
    )


def _bad_connectors() -> ConnectorOutput:
    """Fails connector validation — missing connector for s2."""
    return ConnectorOutput(
        connectors=[],
        complete_path_waypoints=[_wp()],
        segment_types=["ground_level", "flat_surface", "flat_surface"],
    )


def _good_camera() -> CameraOutput:
    """Smooth camera output that passes all validation checks."""
    kfs = [
        CameraKeyframe(
            frame=f * 5,
            position=(3.0, f * 0.3, 4.0),
            look_target=(0.0, f * 0.3, 0.0),
        )
        for f in range(30)
    ]
    return CameraOutput(keyframes=kfs, total_frames=150)


def _mock_good_camera(pipeline: CreativePipeline) -> None:
    """Replace the camera router with a mock that returns a good camera output,
    and replace validate_camera to always pass.  Also mock final reviewer to pass."""
    pipeline._camera_router.compute_trajectory = MagicMock(return_value=_good_camera())
    pipeline._final_reviewer.review = AsyncMock(
        return_value=FinalReviewResult(passed=True, summary="All good"),
    )


# ─── Happy path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_success_first_attempt(tmp_path):
    """All agents pass on first attempt → success immediately."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert result.success
    assert result.manifest is not None
    assert result.path is not None
    assert result.connectors is not None
    assert result.history.total_pipeline_attempts == 4  # one per agent (incl. camera router)


# ─── Cascade: connector failure restarts only connector ──────────────


@pytest.mark.asyncio
async def test_connector_fail_retries_only_connector(tmp_path):
    """Connector fails once, then succeeds — only connector re-runs."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(
        side_effect=[_bad_connectors(), _good_connectors()],
    )
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert result.success
    # set_designer: 1 call, path_setter: 1 call, connector: 2 calls
    assert pipeline._set_designer.design_scene.call_count == 1
    assert pipeline._path_setter.plan_path.call_count == 1
    assert pipeline._connector.resolve_connectors.call_count == 2


# ─── Cascade: path setter failure restarts path + connector ──────────


@pytest.mark.asyncio
async def test_path_fail_cascades_to_path_and_connector(tmp_path):
    """Path setter fails, then succeeds — re-runs path + connector."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(
        side_effect=[_bad_path(), _good_path()],
    )
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert result.success
    assert pipeline._set_designer.design_scene.call_count == 1
    assert pipeline._path_setter.plan_path.call_count == 2
    assert pipeline._connector.resolve_connectors.call_count == 1  # only after path passes


# ─── Cascade: set designer failure restarts all ──────────────────────


@pytest.mark.asyncio
async def test_scene_fail_cascades_all(tmp_path):
    """Scene fails, then all three re-run."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(
        side_effect=[_bad_manifest(), _good_manifest()],
    )
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert result.success
    assert pipeline._set_designer.design_scene.call_count == 2
    # path_setter + connector only run after second (good) manifest
    assert pipeline._path_setter.plan_path.call_count == 1
    assert pipeline._connector.resolve_connectors.call_count == 1


# ─── Exhaustion (per-agent limit) ───────────────────────────────────


@pytest.mark.asyncio
async def test_per_agent_exhaustion(tmp_path):
    """Agent reaches per_agent_max → pipeline stops, success=False."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    # Set designer always fails
    pipeline._set_designer.design_scene = AsyncMock(return_value=_bad_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert not result.success
    assert pipeline._set_designer.design_scene.call_count == PER_AGENT_MAX
    # final_review.yaml should exist
    assert (tmp_path / "sessions" / "test" / "final_review.yaml").exists()


# ─── Session artefacts written ───────────────────────────────────────


@pytest.mark.asyncio
async def test_session_artefacts_created(tmp_path):
    """Successful run writes manifest.yaml and result.yaml."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="art")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    await pipeline.run(domino_count=200)

    root = tmp_path / "sessions" / "art"
    assert (root / "manifest.yaml").exists()
    assert (root / "result.yaml").exists()
    # Attempt directories
    assert (root / "agents" / "set_designer" / "attempt_1" / "output.yaml").exists()
    assert (root / "agents" / "path_setter" / "attempt_1" / "validation.yaml").exists()
    assert (root / "agents" / "connector" / "attempt_1" / "summary.md").exists()


# ─── Camera Router step in pipeline ─────────────────────────────────


def _bad_camera() -> CameraOutput:
    """Only 1 keyframe — fails has_keyframes check."""
    return CameraOutput(
        keyframes=[CameraKeyframe(frame=0, position=(0, 0, 0), look_target=(1, 0, 0))],
        total_frames=100,
    )


@pytest.mark.asyncio
async def test_camera_fail_retries_only_camera(tmp_path):
    """Camera router fails once, then succeeds — only camera re-runs."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    # Camera: fail once (1 keyframe), then good
    pipeline._camera_router.compute_trajectory = MagicMock(
        side_effect=[_bad_camera(), _good_camera()],
    )
    pipeline._final_reviewer.review = AsyncMock(
        return_value=FinalReviewResult(passed=True, summary="All good"),
    )

    result = await pipeline.run(domino_count=200)

    assert result.success
    assert result.camera is not None
    # Set designer, path, connector each called once
    assert pipeline._set_designer.design_scene.call_count == 1
    assert pipeline._path_setter.plan_path.call_count == 1
    assert pipeline._connector.resolve_connectors.call_count == 1


@pytest.mark.asyncio
async def test_pipeline_success_includes_camera_and_review(tmp_path):
    """Successful run populates camera and final_review on the result."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    _mock_good_camera(pipeline)

    result = await pipeline.run(domino_count=200)

    assert result.success
    assert result.camera is not None
    assert result.final_review is not None
    assert result.final_review.passed


# ─── Final reviewer cascade ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_final_reviewer_fails_cascades_from_attributed_agent(tmp_path):
    """Final reviewer failure cascades from the attributed agent."""
    pipeline = CreativePipeline(assets=[], output_dir=tmp_path, session_id="test")

    pipeline._set_designer.design_scene = AsyncMock(return_value=_good_manifest())
    pipeline._path_setter.plan_path = AsyncMock(return_value=_good_path())
    pipeline._connector.resolve_connectors = AsyncMock(return_value=_good_connectors())
    pipeline._camera_router.compute_trajectory = MagicMock(return_value=_good_camera())

    # First call: final reviewer fails, attributed to CONNECTOR
    # Second call: final reviewer passes
    call_count = {"n": 0}

    async def _mock_review(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return FinalReviewResult(
                passed=False,
                issues=[
                    ReviewIssue(
                        description="Chain break detected",
                        attributed_to=AgentRole.CONNECTOR,
                        severity="blocking",
                    ),
                ],
                cascade_from=AgentRole.CONNECTOR,
                summary="[connector] Chain break detected",
            )
        return FinalReviewResult(passed=True, summary="All good")

    pipeline._final_reviewer.review = _mock_review

    result = await pipeline.run(domino_count=200)

    assert result.success
    # Connector re-ran (cascade from CONNECTOR = [CONNECTOR, CAMERA_ROUTER])
    assert pipeline._connector.resolve_connectors.call_count == 2
    # Set designer and path setter didn't re-run
    assert pipeline._set_designer.design_scene.call_count == 1
    assert pipeline._path_setter.plan_path.call_count == 1

