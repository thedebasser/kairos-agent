"""Pydantic data models for the Blender domino run pipeline.

These are the LLM-structured outputs and internal state models
used exclusively by the domino pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from kairos.models.contracts import (
    AudioBrief,
    DominoArchetype,
    EnergyLevel,
)


class DominoCourseConfig(BaseModel, frozen=True):
    """Full configuration for generating one domino run in Blender.

    Produced by the idea agent (LLM) and consumed by the Blender scripts.
    The LLM only controls creative parameters (path shape, count, colours,
    finale object). Physics values are research-locked for reliable toppling.
    """

    seed: int = Field(description="Random seed for deterministic generation")
    archetype: DominoArchetype = Field(description="Path archetype to use")
    title: str = Field(description="Short descriptive title for the concept")
    visual_brief: str = Field(
        description="2-3 sentence description of what the viewer sees",
    )

    # Path layout
    domino_count: int = Field(
        ge=50, le=1000, default=300,
        description="Number of dominoes to place along the path",
    )
    path_amplitude: float = Field(
        ge=0.0, le=5.0, default=1.0,
        description="Horizontal amplitude for curved paths (Blender units)",
    )
    path_cycles: float = Field(
        ge=0.5, le=4.0, default=1.0,
        description="Number of S-curve oscillation cycles",
    )
    spiral_turns: float = Field(
        ge=1.0, le=5.0, default=2.0,
        description="Number of spiral rotations (spiral archetype only)",
    )
    branch_count: int = Field(
        ge=2, le=5, default=3,
        description="Number of branches (branching archetype only)",
    )

    # Domino geometry (locked to validated values)
    domino_width: float = Field(default=0.08, description="Domino width (m)")
    domino_height: float = Field(default=0.4, description="Domino height (m)")
    domino_depth: float = Field(default=0.06, description="Domino depth / thickness in tipping direction (m)")

    # Spacing — locked to 0.35× height (validated: 1.4 BU at WORLD_SCALE=10
    # for h=4.0 BU dominos with d=0.6 BU depth → 0.8 BU gap).
    spacing_ratio: float = Field(
        default=0.35, ge=0.2, le=0.6,
        description="Spacing between dominoes as fraction of domino height",
    )

    # Trigger
    trigger_frame: int = Field(
        default=30, ge=5, le=150,
        description="Frame at which the trigger animation fires",
    )
    trigger_impulse: float = Field(
        default=1.5, ge=0.5, le=5.0,
        description="Force applied to tip the first domino",
    )
    trigger_tilt_degrees: float = Field(
        default=8.0, ge=3.0, le=20.0,
        description="Tilt angle for first domino to initiate chain",
    )

    # Finale object
    finale_type: str = Field(
        default="none",
        description="Finale object at end of path: none, tower, ramp, ball",
    )

    # Visual
    palette: str = Field(default="rainbow", description="Colour palette name")
    camera_style: str = Field(default="tracking", description="Camera style")
    lighting_preset: str = Field(default="studio", description="Lighting preset")

    # Timing
    duration_sec: int = Field(default=65, ge=62, le=68)
    fps: int = Field(default=30, ge=24, le=30)

    # Physics — locked to research-validated values
    domino_mass: float = Field(default=0.3, description="Domino mass (kg)")
    domino_friction: float = Field(default=0.6, ge=0.4, le=0.9)
    domino_bounce: float = Field(default=0.1, ge=0.0, le=0.3)
    ground_friction: float = Field(default=0.8, ge=0.6, le=1.0)
    substeps_per_frame: int = Field(default=20, ge=10, le=30)
    solver_iterations: int = Field(default=20, ge=10, le=60)

    hook_text: str = Field(
        default="",
        description="Hook caption text (<=6 words) shown at 0-2s",
        max_length=50,
    )
    audio_brief: AudioBrief = Field(
        default_factory=lambda: AudioBrief(mood=["satisfying", "building"]),
    )

    novelty_score: float = Field(ge=0.0, le=10.0, default=7.0)
    feasibility_score: float = Field(ge=0.0, le=10.0, default=9.0)

    def to_blender_config(self) -> dict[str, Any]:
        """Convert to the flat dict expected by Blender scripts."""
        return {
            "seed": self.seed,
            "archetype": self.archetype.value,
            "domino_count": self.domino_count,
            "path_amplitude": self.path_amplitude,
            "path_cycles": self.path_cycles,
            "spiral_turns": self.spiral_turns,
            "branch_count": self.branch_count,
            "domino_width": self.domino_width,
            "domino_height": self.domino_height,
            "domino_depth": self.domino_depth,
            "spacing_ratio": self.spacing_ratio,
            "trigger_frame": self.trigger_frame,
            "trigger_impulse": self.trigger_impulse,
            "trigger_tilt_degrees": self.trigger_tilt_degrees,
            "finale_type": self.finale_type,
            "palette": self.palette,
            "camera_style": self.camera_style,
            "lighting_preset": self.lighting_preset,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "domino_mass": self.domino_mass,
            "domino_friction": self.domino_friction,
            "domino_bounce": self.domino_bounce,
            "ground_friction": self.ground_friction,
            "substeps_per_frame": self.substeps_per_frame,
            "solver_iterations": self.solver_iterations,
        }


class CourseGenerationResult(BaseModel, frozen=True):
    """Result from Blender domino course generation."""

    seed: int
    archetype: str
    domino_count: int
    path_length: float = 0.0
    course_center: list[float] = Field(default_factory=list)
    course_bounds_min: list[float] = Field(default_factory=list)
    course_bounds_max: list[float] = Field(default_factory=list)
    duration_frames: int = 1950
    fps: int = 30
    camera_style: str = ""
    lighting_preset: str = ""
    palette: str = ""
    blend_file: str = ""


class CourseValidationResult(BaseModel, frozen=True):
    """Result from Blender domino course validation."""

    passed: bool
    summary: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)


class SmokeTestResult(BaseModel, frozen=True):
    """Result from the physics smoke test."""

    passed: bool
    reason: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    fallen_count: int = 0
    total_count: int = 0
    completion_ratio: float = 0.0


class BakeRenderResult(BaseModel, frozen=True):
    """Result from baking + rendering."""

    output_path: str
    preset: str = ""
    bake_time_sec: float = 0.0
    render_time_sec: float = 0.0
    total_frames: int = 0
    resolution: str = ""
    fps: int = 30
