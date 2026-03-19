"""Ball pit simulation configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from kairos.pipelines.physics.configs.base import BaseSimulationConfig


class BallPitConfig(BaseSimulationConfig):
    """Configuration for a ball pit simulation.

    The LLM controls creative choices (container shape, ball counts,
    colour palette, climax event) while physics parameters are locked.
    """

    # Ball geometry
    ball_radius_min: int = Field(default=15, ge=8, le=25, description="Minimum ball radius")
    ball_radius_max: int = Field(default=35, ge=20, le=50, description="Maximum ball radius")
    ball_count: int = Field(default=200, ge=50, le=400, description="Total balls to drop")

    # Spawn timing
    drop_rate_phase1: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Balls per second during slow intro phase",
    )
    drop_rate_phase2: float = Field(
        default=10.0,
        ge=5.0,
        le=20.0,
        description="Balls per second during fast fill phase",
    )
    phase2_start_sec: float = Field(
        default=15.0,
        ge=5.0,
        le=30.0,
        description="Second at which phase 2 begins",
    )

    # Container
    container_type: Literal["box", "v_shape", "funnel", "rounded"] = Field(
        default="box",
        description="Container shape type",
    )
    container_width_ratio: float = Field(
        default=0.7,
        ge=0.4,
        le=0.95,
        description="Container width as ratio of screen width",
    )
    container_height_ratio: float = Field(
        default=0.4,
        ge=0.2,
        le=0.6,
        description="Container height as ratio of screen height",
    )

    # Climax event
    climax_type: Literal["gate_drop", "big_ball", "shake", "none"] = Field(
        default="gate_drop",
        description="Climax event type",
    )
    climax_time_sec: float = Field(
        default=50.0,
        ge=30.0,
        le=60.0,
        description="When the climax event triggers",
    )

    # Physics — locked to validated values
    ball_elasticity: float = Field(
        default=0.7,
        ge=0.5,
        le=0.9,
        description="Ball bounce elasticity (0.7–0.85 for satisfying bounces)",
    )
    ball_friction: float = Field(
        default=0.3,
        ge=0.1,
        le=0.6,
        description="Ball surface friction",
    )
    ball_mass_min: float = Field(default=1.0, description="Mass for smallest balls")
    ball_mass_max: float = Field(default=5.0, description="Mass for largest balls")
    wall_elasticity: float = Field(default=0.3, description="Wall elasticity")
    wall_friction: float = Field(default=0.8, description="Wall friction")
    substeps: int = Field(default=3, ge=2, le=5, description="Physics substeps per frame")
