"""Kairos Agent — Calibration System Data Models.

Pydantic models for the three-environment calibration learning system:
Sandbox (experimentation) → Quality Gate → Knowledge Base (ChromaDB).

Calibration parameters are stored as correction factors (multipliers)
relative to the research-locked baseline, making them composable across
dimensions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class CalibrationStatus(str, Enum):
    """Status of a calibration session."""

    RUNNING = "running"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    PROMOTED = "promoted"


class FailureMode(str, Enum):
    """Categorised failure modes from simulation validation."""

    CHAIN_BREAK = "chain_break"
    DIRECTION_ERROR = "direction_error"
    CLIPPING = "clipping"
    EXPLOSION = "explosion"
    FLOATING = "floating"
    TRIGGER_MISS = "trigger_miss"
    INCOMPLETE_PROPAGATION = "incomplete_propagation"
    SCRIPT_CRASH = "script_crash"
    UNKNOWN = "unknown"


class PathType(str, Enum):
    """Supported path types for domino layouts."""

    STRAIGHT = "straight"
    ARC = "arc"
    S_CURVE = "s_curve"
    SPIRAL = "spiral"
    CASCADE = "cascade"
    BRANCHING = "branching"
    WORD_SPELL = "word_spell"
    CUSTOM_SPLINE = "custom_spline"


class SurfaceType(str, Enum):
    """Surface type the dominoes sit on."""

    FLAT = "flat"
    RAMP = "ramp"
    STAIRCASE = "staircase"
    UNEVEN = "uneven"


class SizeProfile(str, Enum):
    """Size variation profile across the chain."""

    UNIFORM = "uniform"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    ALTERNATING = "alternating"


# =============================================================================
# Scenario Descriptor
# =============================================================================


class PathDescriptor(BaseModel):
    """Describes the geometric path dominoes follow."""

    type: PathType = Field(description="Path shape archetype")
    radius: float | None = Field(default=None, description="Arc radius in meters (arc/spiral)")
    arc_degrees: float | None = Field(default=None, description="Arc sweep in degrees")
    amplitude: float | None = Field(default=None, description="S-curve amplitude in BU")
    cycles: float | None = Field(default=None, description="S-curve oscillation count")
    spiral_turns: float | None = Field(default=None, description="Spiral rotation count")
    branch_count: int | None = Field(default=None, description="Number of branches")


class SurfaceDescriptor(BaseModel):
    """Describes the surface dominoes sit on."""

    type: SurfaceType = Field(default=SurfaceType.FLAT)
    tilt_degrees: float = Field(default=0.0, description="Incline angle (0 = flat)")


class SizeProfileDescriptor(BaseModel):
    """Describes how domino sizes vary across the chain."""

    type: SizeProfile = Field(default=SizeProfile.UNIFORM)
    start_height: float = Field(default=0.4, description="First domino height (m)")
    end_height: float = Field(default=0.4, description="Last domino height (m)")
    growth_factor: float = Field(default=1.0, description="Size multiplier per step (1.0 = uniform)")


class ScenarioDescriptor(BaseModel):
    """Full structured description of a domino scenario.

    This is both the input to the calibration loop AND the search query
    for ChromaDB.  Every field maps to either a metadata filter or
    contributes to the semantic embedding.
    """

    path: PathDescriptor = Field(description="Path geometry")
    surface: SurfaceDescriptor = Field(default_factory=SurfaceDescriptor)
    size_profile: SizeProfileDescriptor = Field(default_factory=SizeProfileDescriptor)
    domino_count: int = Field(ge=10, le=1000, description="Number of dominoes")
    material: str = Field(default="wood", description="Domino material (affects friction/mass)")

    def to_natural_language(self) -> str:
        """Convert to natural language for ChromaDB embedding."""
        parts = [
            f"Scenario: {self.domino_count} dominoes following a "
            f"{self.path.type.value} path on a {self.surface.type.value} surface.",
            f"Path type: {self.path.type.value}.",
        ]
        if self.path.radius is not None:
            parts.append(f"Radius: {self.path.radius}m.")
        if self.path.arc_degrees is not None:
            parts.append(f"Arc degrees: {self.path.arc_degrees}.")
        if self.path.amplitude is not None:
            parts.append(f"Amplitude: {self.path.amplitude} BU.")
        if self.path.cycles is not None:
            parts.append(f"Cycles: {self.path.cycles}.")
        if self.path.spiral_turns is not None:
            parts.append(f"Spiral turns: {self.path.spiral_turns}.")
        if self.path.branch_count is not None:
            parts.append(f"Branch count: {self.path.branch_count}.")

        parts.append(
            f"Surface: {self.surface.type.value}"
            + (f", tilt {self.surface.tilt_degrees}°." if self.surface.tilt_degrees else ".")
        )
        parts.append(
            f"Size profile: {self.size_profile.type.value}."
            + (
                f" Height {self.size_profile.start_height}m → {self.size_profile.end_height}m."
                if self.size_profile.type != SizeProfile.UNIFORM
                else f" Domino height: {self.size_profile.start_height}m."
            )
        )
        parts.append(f"Domino count: {self.domino_count}.")
        return " ".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        """Convert to flat metadata dict for ChromaDB filtering."""
        meta: dict[str, Any] = {
            "path_type": self.path.type.value,
            "surface_type": self.surface.type.value,
            "surface_tilt_degrees": self.surface.tilt_degrees,
            "size_profile": self.size_profile.type.value,
            "domino_count": self.domino_count,
            "domino_height": self.size_profile.start_height,
            "material": self.material,
        }
        if self.path.amplitude is not None:
            meta["path_amplitude"] = self.path.amplitude
        if self.path.cycles is not None:
            meta["path_cycles"] = self.path.cycles
        if self.path.spiral_turns is not None:
            meta["spiral_turns"] = self.path.spiral_turns
        if self.path.branch_count is not None:
            meta["branch_count"] = self.path.branch_count
        return meta


# =============================================================================
# Calibration Parameters (Correction Factors)
# =============================================================================


# Research-locked baseline values — the "1.0" reference point for all
# correction factors.  These match the overrides in idea_agent.py.
BASELINE_PHYSICS: dict[str, float] = {
    "spacing_ratio": 0.35,
    "domino_mass": 0.3,
    "domino_friction": 0.6,
    "domino_bounce": 0.1,
    "ground_friction": 0.8,
    "trigger_impulse": 1.5,
    "trigger_tilt_degrees": 8.0,
    "substeps_per_frame": 20,
    "solver_iterations": 20,
}


class CorrectionFactors(BaseModel):
    """Multipliers relative to the research-locked baseline.

    A value of 1.0 means "use the baseline as-is".
    A value of 0.92 means "8% tighter/lower than baseline".
    A value of 1.15 means "15% higher than baseline".

    This makes calibrations composable: multiply correction factors from
    different dimensions to get a combined starting point.
    """

    spacing_ratio: float = Field(default=1.0, description="Multiplier for spacing_ratio baseline")
    domino_mass: float = Field(default=1.0, description="Multiplier for mass baseline")
    domino_friction: float = Field(default=1.0, description="Multiplier for domino friction")
    domino_bounce: float = Field(default=1.0, description="Multiplier for bounce")
    ground_friction: float = Field(default=1.0, description="Multiplier for ground friction")
    trigger_impulse: float = Field(default=1.0, description="Multiplier for trigger impulse")
    trigger_tilt_degrees: float = Field(default=1.0, description="Multiplier for trigger tilt")
    substeps_per_frame: float = Field(default=1.0, description="Multiplier for substeps")
    solver_iterations: float = Field(default=1.0, description="Multiplier for solver iterations")
    curve_inner_spacing_factor: float = Field(
        default=1.0,
        description="Additional spacing factor for inner edges of curves",
    )
    notes: str = Field(default="", description="Human-readable notes about this calibration")

    def apply_to_baseline(self) -> dict[str, float]:
        """Apply correction factors to the baseline, returning absolute values."""
        result: dict[str, float] = {}
        for param, baseline_val in BASELINE_PHYSICS.items():
            factor = getattr(self, param, 1.0)
            result[param] = round(baseline_val * factor, 6)
        result["curve_inner_spacing_factor"] = self.curve_inner_spacing_factor
        return result

    @classmethod
    def from_absolute(cls, params: dict[str, float]) -> CorrectionFactors:
        """Compute correction factors from absolute parameter values."""
        factors: dict[str, float] = {}
        for param, baseline_val in BASELINE_PHYSICS.items():
            if param in params and baseline_val != 0:
                factors[param] = round(params[param] / baseline_val, 6)
        if "curve_inner_spacing_factor" in params:
            factors["curve_inner_spacing_factor"] = params["curve_inner_spacing_factor"]
        return cls(**factors)


# =============================================================================
# Calibration Entry (what gets stored in ChromaDB)
# =============================================================================


class CalibrationEntry(BaseModel):
    """A single calibration stored in the knowledge base.

    Contains the scenario descriptor, the proven correction factors,
    confidence metadata, and provenance information.
    """

    calibration_id: UUID = Field(default_factory=uuid4)
    scenario: ScenarioDescriptor
    corrections: CorrectionFactors
    confidence: float = Field(ge=0.0, le=1.0, description="Trustworthiness score")
    iteration_count: int = Field(description="How many iterations to converge")
    perceptual_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="VLM perceptual validation score",
    )
    date_calibrated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    blender_version: str = Field(default="", description="Blender version used for calibration")
    calibration_type: str = Field(
        default="resolved",
        description="'resolved' for proven calibrations, 'unresolved' for negative examples",
    )

    def to_chromadb_document(self) -> str:
        """Generate the text document for ChromaDB embedding."""
        return self.scenario.to_natural_language()

    def to_chromadb_metadata(self) -> dict[str, Any]:
        """Generate flat metadata for ChromaDB storage."""
        meta = self.scenario.to_metadata()
        meta.update({
            "calibration_id": str(self.calibration_id),
            "confidence": self.confidence,
            "iteration_count": self.iteration_count,
            "perceptual_score": self.perceptual_score,
            "date_calibrated": self.date_calibrated.isoformat(),
            "blender_version": self.blender_version,
            "calibration_type": self.calibration_type,
            # Store corrections as JSON string in metadata
            "corrections_json": self.corrections.model_dump_json(),
        })
        return meta


# =============================================================================
# Calibration Session (sandbox iteration tracking)
# =============================================================================


class IterationRecord(BaseModel):
    """Record of a single calibration iteration."""

    iteration: int
    params_used: dict[str, Any] = Field(default_factory=dict)
    validation_passed: bool = False
    completion_ratio: float = 0.0
    failure_modes: list[FailureMode] = Field(default_factory=list)
    failure_details: str = ""
    correction_applied: dict[str, Any] = Field(default_factory=dict)


class CalibrationSession(BaseModel):
    """Full record of a calibration sandbox session."""

    session_id: UUID = Field(default_factory=uuid4)
    scenario: ScenarioDescriptor
    status: CalibrationStatus = CalibrationStatus.RUNNING
    max_iterations: int = Field(default=10)
    iterations: list[IterationRecord] = Field(default_factory=list)
    starting_params: dict[str, float] = Field(
        default_factory=dict,
        description="Initial parameters (from ChromaDB lookup or baseline)",
    )
    final_corrections: CorrectionFactors | None = None
    confidence: float = 0.0
    blender_version: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def last_iteration(self) -> IterationRecord | None:
        return self.iterations[-1] if self.iterations else None


# =============================================================================
# Quality Gate Result
# =============================================================================


class QualityGateResult(BaseModel):
    """Result of passing a calibration through the quality gate."""

    passed: bool = False
    chain_completion: float = Field(default=0.0, description="Fraction of dominoes toppled")
    physics_anomalies: int = Field(default=0, description="Count of anomalous events")
    perceptual_score: float = Field(default=0.0, description="VLM confidence score")
    confidence: float = Field(default=0.0, description="Computed confidence score")
    iteration_count: int = 0
    param_deviation_from_baseline: float = Field(
        default=0.0,
        description="Max deviation of any correction factor from 1.0",
    )
    requires_human_review: bool = False
    human_review_reason: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Calibration Lookup Result
# =============================================================================


class CalibrationMatch(BaseModel):
    """A single match from ChromaDB lookup."""

    calibration_id: str
    corrections: CorrectionFactors
    similarity: float = Field(description="Cosine similarity (0-1)")
    dimensional_overlap: float = Field(description="Structural match score (0-1)")
    combined_score: float = Field(description="Weighted combination of similarity + overlap")
    confidence: float
    matching_dimensions: list[str] = Field(default_factory=list)
