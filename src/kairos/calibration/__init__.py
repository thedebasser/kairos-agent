"""Kairos Agent — Calibration Learning System.

Three-environment system: Sandbox → Quality Gate → Knowledge Base.

Usage::

    from kairos.calibration import KnowledgeBase, run_calibration, ScenarioDescriptor

Feature-gated via ``settings.calibration_enabled``.
"""

from kairos.calibration.knowledge_base import KnowledgeBase
from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CalibrationEntry,
    CalibrationMatch,
    CalibrationSession,
    CalibrationStatus,
    CorrectionFactors,
    FailureMode,
    PathDescriptor,
    PathType,
    QualityGateResult,
    ScenarioDescriptor,
    SizeProfile,
    SizeProfileDescriptor,
    SurfaceDescriptor,
    SurfaceType,
)
from kairos.calibration.sandbox import run_calibration

__all__ = [
    "BASELINE_PHYSICS",
    "CalibrationEntry",
    "CalibrationMatch",
    "CalibrationSession",
    "CalibrationStatus",
    "CorrectionFactors",
    "FailureMode",
    "KnowledgeBase",
    "PathDescriptor",
    "PathType",
    "QualityGateResult",
    "ScenarioDescriptor",
    "SizeProfile",
    "SizeProfileDescriptor",
    "SurfaceDescriptor",
    "SurfaceType",
    "run_calibration",
]
