"""Block destruction simulation configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kairos.pipelines.physics.configs.base import BaseSimulationConfig


class StructureLayer(BaseModel):
    """A single layer of the structure to be destroyed."""

    block_count: int = Field(default=5, ge=1, le=12, description="Blocks in this row")
    block_width: float = Field(default=80.0, description="Block width in pixels")
    block_height: float = Field(default=40.0, description="Block height in pixels")
    offset_x: float = Field(default=0.0, description="Horizontal offset for staggering")
    color_index: int = Field(default=0, description="Index into the palette for this layer")


class DestructionConfig(BaseSimulationConfig):
    """Configuration for a block destruction simulation.

    The LLM controls creative choices (structure layout, projectile type,
    colour palette) while physics parameters are locked.
    """

    # Structure definition
    structure_type: Literal["tower", "pyramid", "wall", "bridge", "castle"] = Field(
        default="tower",
        description="Structure shape type",
    )
    layers: list[StructureLayer] = Field(
        default_factory=list,
        description="Structure layers from bottom to top (if empty, auto-generated)",
    )
    structure_center_x: float = Field(
        default=540.0,
        description="Center X position of the structure",
    )
    structure_base_y_offset: int = Field(
        default=100,
        description="Pixels above the floor for structure base",
    )
    default_block_width: float = Field(default=80.0, description="Default block width")
    default_block_height: float = Field(default=40.0, description="Default block height")
    default_rows: int = Field(default=10, ge=3, le=20, description="Default row count if layers empty")
    default_cols: int = Field(default=5, ge=2, le=10, description="Default column count if layers empty")

    # Pre-settle (let structure settle under gravity before projectile)
    settle_time_sec: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Seconds to let structure settle before filming",
    )

    # Projectile
    projectile_type: Literal["ball", "wrecking_ball", "explosion", "beam"] = Field(
        default="ball",
        description="Type of destructive force",
    )
    projectile_radius: float = Field(
        default=40.0,
        ge=20.0,
        le=80.0,
        description="Projectile radius in pixels",
    )
    projectile_mass: float = Field(
        default=50.0,
        ge=20.0,
        le=200.0,
        description="Projectile mass (heavy = more destruction)",
    )
    launch_time_sec: float = Field(
        default=5.0,
        ge=2.0,
        le=15.0,
        description="When the projectile is launched",
    )
    launch_velocity_x: float = Field(
        default=800.0,
        description="Projectile horizontal velocity (negative = from right)",
    )
    launch_velocity_y: float = Field(
        default=-200.0,
        description="Projectile vertical velocity (negative = upward arc)",
    )
    launch_origin_x: float = Field(
        default=0.0,
        description="Projectile spawn X position",
    )
    launch_origin_y: float = Field(
        default=1400.0,
        description="Projectile spawn Y position",
    )

    # Physics — locked to validated values
    block_mass: float = Field(default=2.0, description="Block mass")
    block_elasticity: float = Field(
        default=0.1,
        ge=0.0,
        le=0.3,
        description="Block elasticity (low for realistic rubble)",
    )
    block_friction: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Block friction (high for stable stacking)",
    )
    floor_elasticity: float = Field(default=0.0, description="Floor elasticity")
    floor_friction: float = Field(default=0.9, description="Floor friction")
    substeps: int = Field(default=3, ge=2, le=5, description="Physics substeps per frame")
