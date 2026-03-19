"""Base simulation configuration shared by all categories."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaseSimulationConfig(BaseModel):
    """Fields common to every simulation category.

    All templates read these base fields for video output and physics globals.
    Category-specific subclasses add their own fields.
    """

    # Video output
    width: int = Field(default=1080, description="Video width in pixels")
    height: int = Field(default=1920, description="Video height in pixels")
    fps: int = Field(default=30, description="Frames per second")
    duration_sec: int = Field(default=65, ge=62, le=68, description="Total video duration")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Colours (RGB tuples)
    background_color: list[int] = Field(
        default=[26, 26, 46],
        min_length=3,
        max_length=3,
        description="Background RGB colour",
    )
    palette: list[list[int]] = Field(
        description="Colour palette as list of [R, G, B] values",
    )

    # Global physics
    gravity_y: float = Field(default=900.0, description="Downward gravity (positive = down)")
    floor_y_offset: int = Field(
        default=100,
        description="Floor position as pixels from the bottom of the screen",
    )
