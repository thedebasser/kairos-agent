"""Tests for creative pipeline Pydantic models."""

from __future__ import annotations

import pytest

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    AttemptRecord,
    CalibrationRef,
    ConnectorOutput,
    ConnectorType,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    IterationHistory,
    LightingConfig,
    ObjectRole,
    PathOutput,
    PathSegment,
    PlacedObject,
    ResolvedConnector,
    SceneManifest,
    SegmentType,
    StepValidationResult,
    Waypoint,
)

pytestmark = pytest.mark.unit


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="wood_floor"),
        lighting=LightingConfig(preset="warm"),
    )


def _make_object(
    asset_id: str = "table_01",
    role: ObjectRole = ObjectRole.FUNCTIONAL,
    z: float = 0.8,
) -> PlacedObject:
    return PlacedObject(
        asset_id=asset_id,
        name="Table",
        position=(1.0, 2.0, z),
        role=role,
        surface_name="top" if role == ObjectRole.FUNCTIONAL else None,
    )


def _make_manifest() -> SceneManifest:
    return SceneManifest(
        theme="modern_kitchen",
        narrative="Dominoes travel across kitchen counters.",
        environment=_make_env(),
        objects=[
            _make_object("table_01", ObjectRole.FUNCTIONAL, 0.8),
            _make_object("shelf_01", ObjectRole.FUNCTIONAL, 1.5),
            _make_object("vase_01", ObjectRole.DECORATIVE, 0.0),
        ],
        domino_count=200,
    )


def _make_waypoint(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


# ─── Scene Manifest Tests ───────────────────────────────────────────


class TestSceneManifest:
    def test_frozen_immutable(self):
        m = _make_manifest()
        with pytest.raises(Exception):
            m.theme = "other"  # type: ignore[misc]

    def test_domino_count_bounds(self):
        with pytest.raises(Exception):
            SceneManifest(
                theme="t",
                narrative="n",
                environment=_make_env(),
                domino_count=10,  # below 50 minimum
            )

    def test_defaults(self):
        m = SceneManifest(
            theme="t", narrative="n", environment=_make_env(),
        )
        assert m.domino_count == 300
        assert m.objects == []


# ─── Path Models Tests ───────────────────────────────────────────────


class TestPathModels:
    def test_segment_frozen(self):
        seg = PathSegment(id="s1", type=SegmentType.FLAT_SURFACE)
        with pytest.raises(Exception):
            seg.id = "s2"  # type: ignore[misc]

    def test_path_output_creation(self):
        seg = PathSegment(
            id="s1",
            type=SegmentType.FLAT_SURFACE,
            waypoints=[_make_waypoint(0, 0, 0), _make_waypoint(1, 0, 0)],
        )
        po = PathOutput(
            total_length_estimate=5.0,
            segments=[seg],
            domino_count=200,
        )
        assert po.domino_count == 200
        assert len(po.segments) == 1


# ─── Connector Models Tests ─────────────────────────────────────────


class TestConnectorModels:
    def test_calibration_ref_defaults(self):
        ref = CalibrationRef()
        assert ref.source == "baseline"
        assert ref.match_confidence == 0.0

    def test_resolved_connector(self):
        rc = ResolvedConnector(
            id="conn_s2",
            for_segment="s2",
            type=ConnectorType.RAMP,
            generated_waypoints=[_make_waypoint(0, 0, 0)],
        )
        assert rc.type == ConnectorType.RAMP

    def test_connector_output_creation(self):
        co = ConnectorOutput(
            connectors=[],
            complete_path_waypoints=[_make_waypoint()],
            segment_types=["flat"],
        )
        assert len(co.complete_path_waypoints) == 1


# ─── Validation Result Tests ────────────────────────────────────────


class TestStepValidationResult:
    def test_passed(self):
        r = StepValidationResult(
            agent=AgentRole.SET_DESIGNER, passed=True, checks=[]
        )
        assert r.passed

    def test_failed_with_summary(self):
        r = StepValidationResult(
            agent=AgentRole.PATH_SETTER,
            passed=False,
            error_summary="Gradient too steep",
        )
        assert not r.passed
        assert "Gradient" in r.error_summary


# ─── Iteration History Tests ────────────────────────────────────────


class TestIterationHistory:
    def test_add_and_count(self):
        h = IterationHistory()
        h.add(AttemptRecord(agent=AgentRole.SET_DESIGNER, attempt=1, passed=True))
        h.add(AttemptRecord(agent=AgentRole.PATH_SETTER, attempt=1, passed=False, error_summary="bad"))
        assert h.total_pipeline_attempts == 2
        assert len(h.for_agent(AgentRole.SET_DESIGNER)) == 1

    def test_latest_for_agent(self):
        h = IterationHistory()
        h.add(AttemptRecord(agent=AgentRole.CONNECTOR, attempt=1, passed=False, error_summary="e1"))
        h.add(AttemptRecord(agent=AgentRole.CONNECTOR, attempt=2, passed=True))
        latest = h.latest_for_agent(AgentRole.CONNECTOR)
        assert latest is not None
        assert latest.attempt == 2
        assert latest.passed

    def test_format_feedback_first_attempt(self):
        h = IterationHistory()
        assert h.format_feedback(AgentRole.SET_DESIGNER) == "This is your first attempt."

    def test_format_feedback_with_history(self):
        h = IterationHistory()
        h.add(AttemptRecord(agent=AgentRole.SET_DESIGNER, attempt=1, passed=False, error_summary="overlap"))
        fb = h.format_feedback(AgentRole.SET_DESIGNER)
        assert "FAILED" in fb
        assert "overlap" in fb
