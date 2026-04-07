"""Tests for the placement-calibration integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kairos.pipelines.domino.creative.models import (
    CalibrationRef,
    ConnectorOutput,
    ConnectorType,
    PathOutput,
    PathSegment,
    ResolvedConnector,
    SegmentType,
    Waypoint,
)
from kairos.pipelines.domino.creative.placement_calibration import (
    store_run_corrections,
)

pytestmark = pytest.mark.unit


# ─── Builders ────────────────────────────────────────────────────────


def _wp(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Waypoint:
    return Waypoint(x=x, y=y, z=z)


def _path() -> PathOutput:
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
                calibration=CalibrationRef(
                    source="chromadb",
                    spacing_correction=0.02,
                    friction_correction=-0.01,
                    trigger_correction=0.05,
                ),
            ),
        ],
        complete_path_waypoints=[_wp(0, 0, 0), _wp(1, 0, 0), _wp(1, 0, 0.8)],
        segment_types=["ground_level", "ramp"],
    )


# ─── Tests ───────────────────────────────────────────────────────────


class TestStoreRunCorrections:
    @patch("kairos.config.get_settings")
    def test_disabled_returns_zero(self, mock_settings):
        settings = MagicMock()
        settings.calibration_enabled = False
        mock_settings.return_value = settings

        result = store_run_corrections(_path(), _connectors())
        assert result == 0

    @patch("kairos.config.get_settings")
    def test_stores_corrections_when_enabled(self, mock_settings):
        settings = MagicMock()
        settings.calibration_enabled = True
        mock_settings.return_value = settings

        mock_kb_instance = MagicMock()
        with patch(
            "kairos.calibration.knowledge_base.KnowledgeBase",
            return_value=mock_kb_instance,
        ):
            result = store_run_corrections(_path(), _connectors())

        assert result == 1
        mock_kb_instance.store.assert_called_once()

    @patch("kairos.config.get_settings")
    def test_multiple_connectors(self, mock_settings):
        settings = MagicMock()
        settings.calibration_enabled = True
        mock_settings.return_value = settings

        two_conn = ConnectorOutput(
            connectors=[
                ResolvedConnector(
                    id="c1", for_segment="s2",
                    type=ConnectorType.RAMP,
                ),
                ResolvedConnector(
                    id="c2", for_segment="s3",
                    type=ConnectorType.SPIRAL_RAMP,
                ),
            ],
            complete_path_waypoints=[_wp()],
            segment_types=["ramp", "spiral"],
        )
        path = PathOutput(
            total_length_estimate=5.0,
            segments=[
                PathSegment(
                    id="s2", type=SegmentType.HEIGHT_TRANSITION_UP,
                    needs_connector=True, from_height=0.0, to_height=0.5,
                ),
                PathSegment(
                    id="s3", type=SegmentType.HEIGHT_TRANSITION_UP,
                    needs_connector=True, from_height=0.5, to_height=1.5,
                ),
            ],
            domino_count=200,
        )

        mock_kb_instance = MagicMock()
        with patch(
            "kairos.calibration.knowledge_base.KnowledgeBase",
            return_value=mock_kb_instance,
        ):
            result = store_run_corrections(path, two_conn)

        assert result == 2
        assert mock_kb_instance.store.call_count == 2

    @patch("kairos.config.get_settings")
    def test_handles_kb_exception_gracefully(self, mock_settings):
        """If KnowledgeBase import fails, returns 0 without crashing."""
        settings = MagicMock()
        settings.calibration_enabled = True
        mock_settings.return_value = settings

        with patch(
            "kairos.calibration.knowledge_base.KnowledgeBase",
            side_effect=ImportError("no chroma"),
        ):
            result = store_run_corrections(_path(), _connectors())

        assert result == 0
