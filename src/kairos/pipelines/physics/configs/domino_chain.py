"""Domino chain simulation configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from kairos.pipelines.physics.configs.base import BaseSimulationConfig


class DominoChainConfig(BaseSimulationConfig):
    """Configuration for a domino chain simulation.

    The LLM fills in creative choices (path type, domino count, colours)
    while physics parameters are locked to research-validated values.
    """

    # Domino geometry
    domino_width: int = Field(default=10, description="Domino width in pixels")
    domino_height: int = Field(default=60, description="Domino height in pixels")
    domino_count: int = Field(default=80, ge=30, le=150, description="Number of dominoes")

    # Spacing — locked to 0.4× height (research-validated sweet spot)
    spacing_ratio: float = Field(
        default=0.4,
        ge=0.3,
        le=0.5,
        description="Spacing as fraction of domino height (0.3–0.5)",
    )

    # Path layout
    path_type: Literal["straight", "s_curve", "arc"] = Field(
        default="s_curve",
        description="Path shape for domino placement",
    )
    path_amplitude: float = Field(
        default=150.0,
        ge=0.0,
        le=300.0,
        description="Horizontal amplitude for s_curve/arc in pixels",
    )
    path_cycles: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Number of S-curve oscillation cycles",
    )

    # Trigger
    trigger_time_sec: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Time in seconds when the first domino is pushed",
    )
    trigger_impulse: float = Field(
        default=200.0,
        ge=100.0,
        le=400.0,
        description="Horizontal impulse applied to first domino top edge",
    )

    # Physics — locked to research-validated values
    domino_mass: float = Field(default=10.0, description="Domino mass")
    domino_elasticity: float = Field(
        default=0.0,
        ge=0.0,
        le=0.1,
        description="Domino shape elasticity (0.0 = fully inelastic, research recommends 0.0)",
    )
    domino_friction: float = Field(
        default=0.5,
        ge=0.3,
        le=0.8,
        description="Domino surface friction",
    )
    floor_elasticity: float = Field(
        default=0.0,
        ge=0.0,
        le=0.1,
        description="Floor elasticity (0.0 recommended)",
    )
    floor_friction: float = Field(
        default=0.9,
        ge=0.7,
        le=1.0,
        description="Floor friction (high prevents backward sliding on impact)",
    )
    substeps: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Physics substeps per frame (3 recommended for thin bodies)",
    )
