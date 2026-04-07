"""Placement-calibration integration — stores corrections after successful runs.

After a successful pipeline run, this module stores the placement
corrections derived from the run into ChromaDB so future runs can
look them up. This closes the calibration feedback loop:

    lookup (before placement) → run → store (after success)
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.pipelines.domino.creative.models import (
    ConnectorOutput,
    ConnectorType,
    PathOutput,
)

logger = logging.getLogger(__name__)


def store_run_corrections(
    path_output: PathOutput,
    connector_output: ConnectorOutput,
) -> int:
    """Store calibration data from a successful run into ChromaDB.

    Returns the number of corrections stored.
    """
    from kairos.config import get_settings
    settings = get_settings()

    if not settings.calibration_enabled:
        return 0

    stored = 0
    try:
        from kairos.calibration.knowledge_base import KnowledgeBase
        from kairos.calibration.models import (
            CalibrationEntry,
            CorrectionFactors,
            PathDescriptor,
            PathType,
            ScenarioDescriptor,
        )

        type_map: dict[str, PathType] = {
            ConnectorType.RAMP.value: PathType.STRAIGHT,
            ConnectorType.SPIRAL_RAMP.value: PathType.SPIRAL,
            ConnectorType.STAIRCASE.value: PathType.CASCADE,
            ConnectorType.PLATFORM.value: PathType.STRAIGHT,
            ConnectorType.PLANK_BRIDGE.value: PathType.STRAIGHT,
        }

        kb = KnowledgeBase()

        for connector in connector_output.connectors:
            path_type = type_map.get(connector.type.value, PathType.STRAIGHT)

            # Find the matching path segment for height delta
            seg = next(
                (s for s in path_output.segments if s.id == connector.for_segment),
                None,
            )
            height_delta = (
                abs(seg.to_height - seg.from_height) if seg else 0.5
            )

            scenario = ScenarioDescriptor(
                path=PathDescriptor(type=path_type),
                domino_count=int(height_delta * 100),
            )

            cal = connector.calibration
            corrections = CorrectionFactors(
                spacing_ratio=1.0 + cal.spacing_correction,
                domino_friction=1.0 + cal.friction_correction,
                trigger_impulse=1.0 + cal.trigger_correction,
            )

            entry = CalibrationEntry(
                scenario=scenario,
                corrections=corrections,
                confidence=0.5,
                iteration_count=1,
            )

            kb.store(entry)
            stored += 1
            logger.debug(
                "[placement_calibration] Stored calibration for %s (height_delta=%.2f)",
                connector.type.value,
                height_delta,
            )

    except Exception:
        logger.debug(
            "[placement_calibration] Failed to store corrections",
            exc_info=True,
        )

    if stored:
        logger.info("[placement_calibration] Stored %d corrections", stored)
    return stored
