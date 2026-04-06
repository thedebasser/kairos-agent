"""Connector Agent — fills height transitions with skill library connectors.

Receives the PathOutput from the Path Setter. For each segment flagged
with needs_connector=True, selects an appropriate connector primitive,
looks up calibration data from ChromaDB, and produces a ConnectorOutput
with the fully-connected domino path.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CalibrationRef,
    ConnectorOutput,
    ConnectorType,
    IterationHistory,
    PathOutput,
    PathSegment,
    ResolvedConnector,
    SegmentType,
    Waypoint,
)
from kairos.skills.contracts import ConnectorResult, Vector3
from kairos.skills.shared.connectors import (
    create_plank_bridge,
    create_platform,
    create_ramp,
    create_spiral_ramp,
    create_staircase,
)

logger = logging.getLogger(__name__)

# Map ConnectorType enum to skill library functions
_CONNECTOR_FUNCTIONS = {
    ConnectorType.RAMP: create_ramp,
    ConnectorType.SPIRAL_RAMP: create_spiral_ramp,
    ConnectorType.STAIRCASE: create_staircase,
    ConnectorType.PLATFORM: create_platform,
    ConnectorType.PLANK_BRIDGE: create_plank_bridge,
}


def _select_connector_type(segment: PathSegment) -> ConnectorType:
    """Select an appropriate connector type based on the segment's hint and height delta."""
    # Honour the hint from the Path Setter if it's a valid type
    if segment.connector_hint:
        try:
            return ConnectorType(segment.connector_hint)
        except ValueError:
            pass

    height_delta = abs(segment.to_height - segment.from_height)

    if height_delta < 0.05:
        return ConnectorType.PLATFORM
    elif height_delta < 0.5:
        return ConnectorType.RAMP
    elif height_delta < 1.5:
        return ConnectorType.STAIRCASE
    else:
        return ConnectorType.SPIRAL_RAMP


def _segment_start_end(segment: PathSegment) -> tuple[Vector3, Vector3]:
    """Extract start and end positions from a segment's waypoints or heights."""
    if segment.waypoints and len(segment.waypoints) >= 2:
        wp0 = segment.waypoints[0]
        wp1 = segment.waypoints[-1]
        return (
            Vector3(wp0.x, wp0.y, wp0.z),
            Vector3(wp1.x, wp1.y, wp1.z),
        )
    # Fallback: construct from heights (assume Y-forward movement)
    return (
        Vector3(0.0, 0.0, segment.from_height),
        Vector3(0.0, 1.0, segment.to_height),
    )


def _call_connector(
    connector_type: ConnectorType,
    start: Vector3,
    end: Vector3,
    params: dict[str, Any],
) -> ConnectorResult:
    """Call the skill library connector function."""
    fn = _CONNECTOR_FUNCTIONS[connector_type]

    # Each connector accepts start + end; pass additional params as kwargs
    # Only pass kwargs the function actually accepts
    import inspect
    sig = inspect.signature(fn)
    valid_kwargs = {
        k: v for k, v in params.items()
        if k in sig.parameters
    }
    return fn(start, end, **valid_kwargs)


def _lookup_calibration(
    connector_type: ConnectorType,
    height_delta: float,
) -> CalibrationRef:
    """Look up calibration corrections from ChromaDB for a connector scenario.

    Returns baseline defaults if calibration is unavailable.
    """
    from kairos.config import get_settings
    settings = get_settings()

    if not settings.calibration_enabled:
        return CalibrationRef(source="baseline")

    try:
        from kairos.calibration.knowledge_base import KnowledgeBase
        from kairos.calibration.models import (
            PathDescriptor,
            PathType,
            ScenarioDescriptor,
        )

        # Map connector type to the closest calibrated path type
        type_map = {
            ConnectorType.RAMP: PathType.STRAIGHT,
            ConnectorType.SPIRAL_RAMP: PathType.SPIRAL,
            ConnectorType.STAIRCASE: PathType.CASCADE,
            ConnectorType.PLATFORM: PathType.STRAIGHT,
            ConnectorType.PLANK_BRIDGE: PathType.STRAIGHT,
        }
        path_type = type_map.get(connector_type, PathType.STRAIGHT)

        scenario = ScenarioDescriptor(
            path=PathDescriptor(type=path_type),
            domino_count=int(height_delta * 100),  # rough estimate
        )
        kb = KnowledgeBase()
        corrections = kb.lookup_starting_params(scenario)
        if corrections:
            return CalibrationRef(
                source="chromadb",
                match_confidence=0.8,
                spacing_correction=corrections.spacing_correction,
                friction_correction=corrections.friction_correction,
                trigger_correction=corrections.trigger_force_correction,
            )
    except Exception:
        logger.debug("Calibration lookup failed, using baseline", exc_info=True)

    return CalibrationRef(source="baseline")


def _waypoints_to_model(waypoints: list[Vector3]) -> list[Waypoint]:
    """Convert skill library Vector3 waypoints to Pydantic Waypoint models."""
    return [Waypoint(x=w.x, y=w.y, z=w.z) for w in waypoints]


class ConnectorAgent:
    """Fills height transitions with skill library connector primitives.

    For each segment flagged as needing a connector:
    1. Select connector type (from hint or height delta heuristic)
    2. Look up calibration data from ChromaDB
    3. Call the skill library function to generate waypoints
    4. Assemble the complete connected path
    """

    async def resolve_connectors(
        self,
        path_output: PathOutput,
        *,
        history: IterationHistory | None = None,
    ) -> ConnectorOutput:
        """Resolve all connectors and produce the fully-connected path.

        Args:
            path_output: Segmented path from the Path Setter.
            history: Optional iteration history for feedback on retries.

        Returns:
            A frozen ConnectorOutput with resolved connectors and the
            complete merged waypoint list.
        """
        resolved: list[ResolvedConnector] = []
        all_waypoints: list[Waypoint] = []
        segment_types: list[str] = []

        for segment in path_output.segments:
            if segment.needs_connector:
                connector = self._resolve_single(segment)
                resolved.append(connector)
                all_waypoints.extend(connector.generated_waypoints)
                segment_types.append(connector.type.value)
            elif segment.waypoints:
                all_waypoints.extend(segment.waypoints)
                segment_types.append(segment.type.value)

        logger.info(
            "[connector] Resolved %d connectors, %d total waypoints",
            len(resolved),
            len(all_waypoints),
        )

        return ConnectorOutput(
            connectors=resolved,
            complete_path_waypoints=all_waypoints,
            segment_types=segment_types,
        )

    def _resolve_single(self, segment: PathSegment) -> ResolvedConnector:
        """Resolve a single connector for a transition segment."""
        ctype = _select_connector_type(segment)
        start, end = _segment_start_end(segment)
        height_delta = abs(segment.to_height - segment.from_height)

        # Calibration lookup
        calibration = _lookup_calibration(ctype, height_delta)

        # Build params from segment metadata + calibration
        params: dict[str, Any] = {}
        if segment.available_footprint != (0.0, 0.0):
            params["width"] = segment.available_footprint[0]

        # Apply calibration corrections to spacing if available
        if calibration.spacing_correction != 0.0:
            params["point_spacing"] = max(
                0.03, 0.05 + calibration.spacing_correction
            )

        # Call skill library
        result = _call_connector(ctype, start, end, params)
        generated_waypoints = _waypoints_to_model(result.waypoints)

        return ResolvedConnector(
            id=f"conn_{segment.id}",
            for_segment=segment.id,
            type=ctype,
            params=params,
            calibration=calibration,
            generated_waypoints=generated_waypoints,
        )
