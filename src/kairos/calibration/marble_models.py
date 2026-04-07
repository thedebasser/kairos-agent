"""Marble run calibration models.

Extends the calibration system with marble-specific scenario descriptors
and correction factors. Marble calibrations are stored in the same
ChromaDB collection as domino calibrations, but with content_type="marble_run"
metadata for filtering.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarbleTrackType(str, Enum):
    """Track geometry types for marble runs."""

    STRAIGHT = "straight"
    CURVED = "curved"
    SPIRAL_DESCENT = "spiral_descent"
    FUNNEL_TO_TRACK = "funnel_to_track"
    LOOP = "loop"
    DROP = "drop"
    MIXED = "mixed"


class MarbleTrackSurface(str, Enum):
    """Surface material of the track."""

    SMOOTH_PLASTIC = "smooth_plastic"
    ROUGH_WOOD = "rough_wood"
    METAL = "metal"
    GLASS = "glass"


class MarbleMaterial(str, Enum):
    """Marble material (affects mass and friction)."""

    GLASS = "glass"
    METAL = "metal"
    CLAY = "clay"


# Research-locked baseline values for marble physics
MARBLE_BASELINE_PHYSICS: dict[str, float] = {
    "marble_mass": 0.028,
    "marble_friction": 0.6,
    "marble_bounce": 0.4,
    "track_friction": 0.5,
    "gravity_scale": 1.0,
    "substeps_per_frame": 20,
    "solver_iterations": 20,
    "linear_damping": 0.04,
    "angular_damping": 0.1,
}


class MarbleTrackDescriptor(BaseModel):
    """Describes the track geometry for a marble run scenario."""

    type: MarbleTrackType = Field(description="Track geometry type")
    piece_count: int = Field(ge=1, le=50, default=10, description="Number of track pieces")
    total_length_m: float = Field(default=2.0, description="Total track length in metres")
    total_height_drop_m: float = Field(default=1.0, description="Total height drop in metres")
    has_loop: bool = Field(default=False)
    has_funnel: bool = Field(default=False)
    min_curve_radius_m: float | None = Field(default=None, description="Tightest curve radius")


class MarbleScenarioDescriptor(BaseModel):
    """Full structured description of a marble run scenario.

    Used as both the calibration loop input and ChromaDB search query.
    """

    track: MarbleTrackDescriptor = Field(description="Track geometry")
    track_surface: MarbleTrackSurface = Field(default=MarbleTrackSurface.SMOOTH_PLASTIC)
    marble_material: MarbleMaterial = Field(default=MarbleMaterial.GLASS)
    marble_count: int = Field(ge=1, le=80, default=5)
    marble_radius_m: float = Field(default=0.04)

    def to_natural_language(self) -> str:
        """Convert to natural language for ChromaDB embedding."""
        parts = [
            f"Marble run scenario: {self.marble_count} {self.marble_material.value} "
            f"marbles on a {self.track.type.value} track.",
            f"Track: {self.track.piece_count} pieces, "
            f"{self.track.total_length_m}m long, "
            f"{self.track.total_height_drop_m}m height drop.",
            f"Surface: {self.track_surface.value}.",
            f"Marble radius: {self.marble_radius_m}m.",
        ]
        if self.track.has_loop:
            parts.append("Track includes a loop.")
        if self.track.has_funnel:
            parts.append("Track includes a funnel entry.")
        if self.track.min_curve_radius_m is not None:
            parts.append(f"Tightest curve radius: {self.track.min_curve_radius_m}m.")
        return " ".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        """Convert to flat metadata dict for ChromaDB filtering."""
        meta: dict[str, Any] = {
            "content_type": "marble_run",
            "track_type": self.track.type.value,
            "piece_count": self.track.piece_count,
            "total_length_m": self.track.total_length_m,
            "total_height_drop_m": self.track.total_height_drop_m,
            "has_loop": self.track.has_loop,
            "has_funnel": self.track.has_funnel,
            "track_surface": self.track_surface.value,
            "marble_material": self.marble_material.value,
            "marble_count": self.marble_count,
            "marble_radius_m": self.marble_radius_m,
        }
        if self.track.min_curve_radius_m is not None:
            meta["min_curve_radius_m"] = self.track.min_curve_radius_m
        return meta


class MarbleCorrectionFactors(BaseModel):
    """Correction factors for marble run calibration.

    Multipliers relative to MARBLE_BASELINE_PHYSICS.
    """

    marble_mass: float = Field(default=1.0)
    marble_friction: float = Field(default=1.0)
    marble_bounce: float = Field(default=1.0)
    track_friction: float = Field(default=1.0)
    gravity_scale: float = Field(default=1.0)
    substeps_per_frame: float = Field(default=1.0)
    solver_iterations: float = Field(default=1.0)
    linear_damping: float = Field(default=1.0)
    angular_damping: float = Field(default=1.0)
    notes: str = Field(default="")

    def apply_to_baseline(self) -> dict[str, float]:
        """Apply correction factors to the marble baseline."""
        result: dict[str, float] = {}
        for param, baseline_val in MARBLE_BASELINE_PHYSICS.items():
            factor = getattr(self, param, 1.0)
            result[param] = round(baseline_val * factor, 6)
        return result


# =============================================================================
# Bootstrap scenarios for marble run calibration
# =============================================================================

MARBLE_BOOTSTRAP_SCENARIOS: list[MarbleScenarioDescriptor] = [
    # 1. Simple straight descent
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.STRAIGHT,
            piece_count=3,
            total_length_m=1.5,
            total_height_drop_m=0.5,
        ),
    ),
    # 2. Long straight descent
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.STRAIGHT,
            piece_count=8,
            total_length_m=4.0,
            total_height_drop_m=1.5,
        ),
    ),
    # 3. Gentle curves
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.CURVED,
            piece_count=6,
            total_length_m=2.5,
            total_height_drop_m=0.8,
            min_curve_radius_m=0.5,
        ),
    ),
    # 4. Tight curves
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.CURVED,
            piece_count=6,
            total_length_m=2.0,
            total_height_drop_m=0.8,
            min_curve_radius_m=0.15,
        ),
    ),
    # 5. Spiral descent
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.SPIRAL_DESCENT,
            piece_count=10,
            total_length_m=3.0,
            total_height_drop_m=2.0,
            min_curve_radius_m=0.25,
        ),
    ),
    # 6. Funnel to track
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.FUNNEL_TO_TRACK,
            piece_count=5,
            total_length_m=2.0,
            total_height_drop_m=0.6,
            has_funnel=True,
        ),
    ),
    # 7. Track with loop
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.LOOP,
            piece_count=8,
            total_length_m=3.0,
            total_height_drop_m=1.5,
            has_loop=True,
        ),
    ),
    # 8. Drop segments
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.DROP,
            piece_count=6,
            total_length_m=2.0,
            total_height_drop_m=1.8,
        ),
    ),
    # 9. Mixed course (all piece types)
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.MIXED,
            piece_count=10,
            total_length_m=4.0,
            total_height_drop_m=2.0,
            has_loop=True,
            has_funnel=True,
            min_curve_radius_m=0.2,
        ),
    ),
    # 10. Multi-marble race (higher marble count)
    MarbleScenarioDescriptor(
        track=MarbleTrackDescriptor(
            type=MarbleTrackType.MIXED,
            piece_count=12,
            total_length_m=5.0,
            total_height_drop_m=2.5,
            has_funnel=True,
            min_curve_radius_m=0.3,
        ),
        marble_count=10,
    ),
]
