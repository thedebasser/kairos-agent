"""Marble funnel / ramp simulation configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kairos.pipelines.physics.configs.base import BaseSimulationConfig


class RampSegment(BaseModel):
    """A single ramp/funnel segment definition."""

    x_start: float = Field(description="Start X position")
    y_start: float = Field(description="Start Y position")
    x_end: float = Field(description="End X position")
    y_end: float = Field(description="End Y position")
    thickness: float = Field(default=8.0, description="Ramp thickness in pixels")


class FunnelDef(BaseModel):
    """Funnel geometry definition."""

    center_x: float = Field(description="Center X of funnel mouth")
    top_y: float = Field(description="Top Y of funnel")
    mouth_width: float = Field(default=300.0, description="Width of funnel mouth")
    neck_width: float = Field(default=40.0, description="Width of funnel neck")
    height: float = Field(default=200.0, description="Funnel height")


class MarbleFunnelConfig(BaseSimulationConfig):
    """Configuration for a marble funnel / ramp simulation.

    The LLM controls creative choices (ramp layout, funnel positions,
    marble counts, colour palette) while physics parameters are locked.
    """

    # Marble geometry
    marble_radius_min: int = Field(default=10, ge=6, le=18, description="Minimum marble radius")
    marble_radius_max: int = Field(default=20, ge=14, le=30, description="Maximum marble radius")
    marble_count: int = Field(default=60, ge=20, le=120, description="Total marbles to spawn")

    # Spawn timing
    spawn_rate: float = Field(
        default=3.0,
        ge=1.0,
        le=8.0,
        description="Marbles spawned per second",
    )
    spawn_start_sec: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="When marble spawning begins",
    )
    spawn_x_min: float = Field(default=350.0, description="Leftmost spawn X position")
    spawn_x_max: float = Field(default=730.0, description="Rightmost spawn X position")
    spawn_y: float = Field(default=50.0, description="Spawn Y position (top of screen)")

    # Layout
    layout_type: Literal["zig_zag", "spiral", "cascade", "funnel_only"] = Field(
        default="zig_zag",
        description="Ramp layout pattern",
    )
    ramp_count: int = Field(default=6, ge=3, le=12, description="Number of ramp segments")
    ramps: list[RampSegment] = Field(
        default_factory=list,
        description="Custom ramp segments (if empty, auto-generated from layout_type)",
    )
    funnels: list[FunnelDef] = Field(
        default_factory=list,
        description="Funnel definitions (if empty, auto-generated)",
    )

    # Collection bin at bottom
    bin_type: Literal["open", "divided", "single"] = Field(
        default="divided",
        description="Collection bin type at the bottom",
    )
    bin_count: int = Field(default=3, ge=1, le=5, description="Number of collection bins")

    # Physics — locked to validated values
    marble_elasticity: float = Field(
        default=0.5,
        ge=0.3,
        le=0.7,
        description="Marble bounce elasticity",
    )
    marble_friction: float = Field(
        default=0.4,
        ge=0.2,
        le=0.7,
        description="Marble surface friction",
    )
    marble_mass: float = Field(default=3.0, description="Marble mass")
    ramp_elasticity: float = Field(default=0.2, description="Ramp surface elasticity")
    ramp_friction: float = Field(default=0.7, description="Ramp surface friction")
    substeps: int = Field(default=3, ge=2, le=5, description="Physics substeps per frame")
