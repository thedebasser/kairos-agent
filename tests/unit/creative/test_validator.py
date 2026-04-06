"""Tests for per-step validation logic."""

from __future__ import annotations

import pytest

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    ConnectorOutput,
    ConnectorType,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    LightingConfig,
    ObjectRole,
    PathOutput,
    PathSegment,
    PlacedObject,
    ResolvedConnector,
    SceneManifest,
    SegmentType,
    Waypoint,
)
from kairos.pipelines.domino.creative.validator import (
    validate_connectors,
    validate_path,
    validate_scene,
)

pytestmark = pytest.mark.unit


# ─── Fixtures ────────────────────────────────────────────────────────


def _env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="tile"),
        lighting=LightingConfig(preset="bright"),
    )


def _functional_obj(
    asset_id: str = "table_01",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 0.8,
    surface: str = "top",
) -> PlacedObject:
    return PlacedObject(
        asset_id=asset_id,
        name="Table",
        position=(x, y, z),
        role=ObjectRole.FUNCTIONAL,
        surface_name=surface,
    )


def _deco_obj(asset_id: str = "vase_01") -> PlacedObject:
    return PlacedObject(
        asset_id=asset_id,
        name="Vase",
        position=(0.0, 0.0, 0.0),
        role=ObjectRole.DECORATIVE,
    )


def _valid_manifest() -> SceneManifest:
    return SceneManifest(
        theme="kitchen",
        narrative="Dominoes cascade across the kitchen.",
        environment=_env(),
        objects=[
            _functional_obj("table_01", 1.0, 2.0, 0.8),
            _functional_obj("shelf_01", -1.0, -2.0, 1.5),
            _deco_obj(),
        ],
        domino_count=200,
    )


def _wp(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


def _valid_path() -> PathOutput:
    return PathOutput(
        total_length_estimate=5.0,
        segments=[
            PathSegment(
                id="s1",
                type=SegmentType.GROUND_LEVEL,
                waypoints=[_wp(0, 0, 0), _wp(1, 0, 0)],
            ),
            PathSegment(
                id="s2",
                type=SegmentType.HEIGHT_TRANSITION_UP,
                needs_connector=True,
                connector_hint="ramp",
                from_height=0.0,
                to_height=0.8,
            ),
            PathSegment(
                id="s3",
                type=SegmentType.FLAT_SURFACE,
                surface_ref="table_01.top",
                waypoints=[_wp(1, 0, 0.8), _wp(2, 0, 0.8)],
            ),
        ],
        domino_count=200,
    )


def _valid_connectors(path: PathOutput) -> ConnectorOutput:
    return ConnectorOutput(
        connectors=[
            ResolvedConnector(
                id="conn_s2",
                for_segment="s2",
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


# ─── Scene Validation ───────────────────────────────────────────────


class TestValidateScene:
    def test_valid_scene_passes(self):
        result = validate_scene(_valid_manifest())
        assert result.passed
        assert result.agent == AgentRole.SET_DESIGNER

    def test_too_few_functional_objects(self):
        m = SceneManifest(
            theme="t",
            narrative="n",
            environment=_env(),
            objects=[_functional_obj("t1"), _deco_obj()],  # only 1 functional
            domino_count=100,
        )
        result = validate_scene(m)
        assert not result.passed
        assert "functional" in result.error_summary.lower()

    def test_functional_without_surface_fails(self):
        bad_obj = PlacedObject(
            asset_id="no_surface",
            position=(1.0, 1.0, 0.5),
            role=ObjectRole.FUNCTIONAL,
            surface_name=None,  # missing!
        )
        m = SceneManifest(
            theme="t",
            narrative="n",
            environment=_env(),
            objects=[
                _functional_obj("a", 0, 0, 0.5),
                bad_obj,
            ],
            domino_count=100,
        )
        result = validate_scene(m)
        assert not result.passed
        assert "no_surface" in result.error_summary

    def test_out_of_bounds(self):
        far_obj = _functional_obj("far", x=10.0, y=0.0, z=0.5)
        m = SceneManifest(
            theme="t",
            narrative="n",
            environment=_env(),
            objects=[
                _functional_obj("a", 0, 0, 0.5),
                far_obj,
            ],
            domino_count=100,
        )
        result = validate_scene(m)
        assert not result.passed
        assert "far" in result.error_summary

    def test_overlapping_objects(self):
        m = SceneManifest(
            theme="t",
            narrative="n",
            environment=_env(),
            objects=[
                _functional_obj("a", 1.0, 1.0, 0.5),
                _functional_obj("b", 1.0, 1.0, 1.0),  # same x,y position
            ],
            domino_count=100,
        )
        result = validate_scene(m)
        assert not result.passed
        assert "overlap" in result.error_summary.lower()

    def test_same_height_fails(self):
        m = SceneManifest(
            theme="t",
            narrative="n",
            environment=_env(),
            objects=[
                _functional_obj("a", 0, 0, 0.8),
                _functional_obj("b", 2, 2, 0.8),  # same height
            ],
            domino_count=100,
        )
        result = validate_scene(m)
        assert not result.passed
        assert "height" in result.error_summary.lower()


# ─── Path Validation ────────────────────────────────────────────────


class TestValidatePath:
    def test_valid_path_passes(self):
        result = validate_path(_valid_path(), _valid_manifest())
        assert result.passed
        assert result.agent == AgentRole.PATH_SETTER

    def test_too_few_segments(self):
        po = PathOutput(
            total_length_estimate=1.0,
            segments=[
                PathSegment(id="s1", type=SegmentType.GROUND_LEVEL, waypoints=[_wp()]),
            ],
            domino_count=100,
        )
        result = validate_path(po, _valid_manifest())
        assert not result.passed
        assert "segment" in result.error_summary.lower()

    def test_no_transitions(self):
        po = PathOutput(
            total_length_estimate=3.0,
            segments=[
                PathSegment(id="s1", type=SegmentType.GROUND_LEVEL, waypoints=[_wp(), _wp(1, 0, 0)]),
                PathSegment(id="s2", type=SegmentType.FLAT_SURFACE, waypoints=[_wp(1, 0, 0), _wp(2, 0, 0)]),
            ],
            domino_count=100,
        )
        result = validate_path(po, _valid_manifest())
        assert not result.passed
        assert "transition" in result.error_summary.lower()

    def test_steep_gradient_fails(self):
        po = PathOutput(
            total_length_estimate=5.0,
            segments=[
                PathSegment(id="s1", type=SegmentType.GROUND_LEVEL, waypoints=[_wp(), _wp(1, 0, 0)]),
                PathSegment(
                    id="s2",
                    type=SegmentType.HEIGHT_TRANSITION_UP,
                    needs_connector=True,
                    from_height=0.0,
                    to_height=0.8,
                    gradient=25.0,  # too steep
                ),
                PathSegment(id="s3", type=SegmentType.FLAT_SURFACE, waypoints=[_wp(1, 0, 0.8)]),
            ],
            domino_count=100,
        )
        result = validate_path(po, _valid_manifest())
        assert not result.passed
        assert "gradient" in result.error_summary.lower()

    def test_zero_domino_count(self):
        po = PathOutput(
            total_length_estimate=5.0,
            segments=[
                PathSegment(id="s1", type=SegmentType.GROUND_LEVEL, waypoints=[_wp()]),
                PathSegment(id="s2", type=SegmentType.HEIGHT_TRANSITION_UP, needs_connector=True, from_height=0.0, to_height=0.5),
            ],
            domino_count=0,
        )
        result = validate_path(po, _valid_manifest())
        assert not result.passed
        assert "domino" in result.error_summary.lower()


# ─── Connector Validation ───────────────────────────────────────────


class TestValidateConnectors:
    def test_valid_connectors_passes(self):
        path = _valid_path()
        conn = _valid_connectors(path)
        result = validate_connectors(conn, path)
        assert result.passed
        assert result.agent == AgentRole.CONNECTOR

    def test_missing_connector_fails(self):
        path = _valid_path()
        conn = ConnectorOutput(
            connectors=[],  # missing the connector for s2
            complete_path_waypoints=[_wp()],
            segment_types=["ground_level", "flat_surface", "flat_surface"],
        )
        result = validate_connectors(conn, path)
        assert not result.passed
        assert "unresolved" in result.error_summary.lower()

    def test_empty_waypoints_fails(self):
        path = _valid_path()
        conn = ConnectorOutput(
            connectors=[
                ResolvedConnector(
                    id="conn_s2",
                    for_segment="s2",
                    type=ConnectorType.RAMP,
                    generated_waypoints=[],  # empty!
                ),
            ],
            complete_path_waypoints=[_wp()],
            segment_types=["ground_level", "ramp", "flat_surface"],
        )
        result = validate_connectors(conn, path)
        assert not result.passed
        assert "waypoint" in result.error_summary.lower()

    def test_empty_complete_path_fails(self):
        path = _valid_path()
        conn = ConnectorOutput(
            connectors=[
                ResolvedConnector(
                    id="conn_s2",
                    for_segment="s2",
                    type=ConnectorType.RAMP,
                    generated_waypoints=[_wp()],
                ),
            ],
            complete_path_waypoints=[],  # empty!
            segment_types=["ground_level", "ramp", "flat_surface"],
        )
        result = validate_connectors(conn, path)
        assert not result.passed
        assert "no waypoints" in result.error_summary.lower()

    def test_segment_types_mismatch(self):
        path = _valid_path()
        conn = ConnectorOutput(
            connectors=[
                ResolvedConnector(
                    id="conn_s2",
                    for_segment="s2",
                    type=ConnectorType.RAMP,
                    generated_waypoints=[_wp(), _wp()],
                ),
            ],
            complete_path_waypoints=[_wp()],
            segment_types=["ground_level"],  # should be 3, not 1
        )
        result = validate_connectors(conn, path)
        assert not result.passed
        assert "segment_types" in result.error_summary.lower()
