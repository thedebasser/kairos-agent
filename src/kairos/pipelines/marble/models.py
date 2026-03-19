"""Pydantic data models for the Blender marble course pipeline.

These are the LLM-structured outputs and internal state models
used exclusively by the marble pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from kairos.models.contracts import (
    AudioBrief,
    EnergyLevel,
    MarbleArchetype,
)


class MarbleCourseConfig(BaseModel, frozen=True):
    """Full configuration for generating one marble course in Blender.

    Produced by the idea agent (LLM) and consumed by the Blender scripts.
    """

    seed: int = Field(description="Random seed for deterministic generation")
    archetype: MarbleArchetype = Field(description="Course archetype to use")
    title: str = Field(description="Short descriptive title for the concept")
    visual_brief: str = Field(
        description="2-3 sentence description of what the viewer sees",
    )
    marble_count: int = Field(ge=3, le=80, default=5, description="Number of marbles to spawn")
    marble_radius: float = Field(ge=0.02, le=0.08, default=0.04)
    marble_bounce: float = Field(ge=0.1, le=0.8, default=0.4)
    marble_friction: float = Field(ge=0.3, le=0.9, default=0.6)
    marble_mass: float = Field(ge=0.3, le=3.0, default=1.0)
    palette: str = Field(default="rainbow", description="Colour palette name")
    camera_style: str = Field(default="marble_follow")
    lighting_preset: str = Field(default="studio")
    duration_sec: int = Field(default=65, ge=62, le=68)
    fps: int = Field(default=30, ge=24, le=30)
    substeps_per_frame: int = Field(default=20, ge=5, le=30)
    solver_iterations: int = Field(default=20, ge=10, le=60)

    # Optional module overrides (if None, archetype grammar generates)
    module_sequence: list[dict[str, Any]] | None = Field(
        default=None,
        description="Explicit module sequence override. If null, generated from archetype.",
    )

    hook_text: str = Field(
        default="",
        description="Hook caption text (≤6 words) shown at 0-2s",
        max_length=50,
    )
    audio_brief: AudioBrief = Field(
        default_factory=lambda: AudioBrief(mood=["ambient", "satisfying"]),
    )

    novelty_score: float = Field(ge=0.0, le=10.0, default=7.0)
    feasibility_score: float = Field(ge=0.0, le=10.0, default=8.0)

    def to_blender_config(self) -> dict[str, Any]:
        """Convert to the flat dict expected by Blender scripts.

        Always sets module_sequence=None so the archetype grammar
        generates the course structure.  The LLM often invents
        creative module types that don't match our MODULE_BUILDERS.
        """
        return {
            "seed": self.seed,
            "archetype": self.archetype.value,
            "marble_count": self.marble_count,
            "marble_radius": self.marble_radius,
            "marble_bounce": self.marble_bounce,
            "marble_friction": self.marble_friction,
            "marble_mass": self.marble_mass,
            "palette": self.palette,
            "camera_style": self.camera_style,
            "lighting_preset": self.lighting_preset,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "substeps_per_frame": self.substeps_per_frame,
            "solver_iterations": self.solver_iterations,
            "module_sequence": None,  # always use archetype grammar
        }


class CourseGenerationResult(BaseModel, frozen=True):
    """Result from Blender course generation."""

    seed: int
    archetype: str
    module_count: int
    module_names: list[str] = Field(default_factory=list)
    marble_count: int
    course_center: list[float] = Field(default_factory=list)
    course_bounds_min: list[float] = Field(default_factory=list)
    course_bounds_max: list[float] = Field(default_factory=list)
    duration_frames: int
    fps: int
    camera_style: str = ""
    lighting_preset: str = ""
    palette: str = ""
    blend_file: str = ""


class CourseValidationResult(BaseModel, frozen=True):
    """Result from Blender course validation."""

    passed: bool
    summary: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)


class SmokeTestResult(BaseModel, frozen=True):
    """Result from the physics smoke test."""

    passed: bool
    reason: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    marble_final_positions: list[dict[str, Any]] = Field(default_factory=list)


class BakeRenderResult(BaseModel, frozen=True):
    """Result from baking + rendering."""

    output_path: str
    preset: str = ""
    bake_time_sec: float = 0.0
    render_time_sec: float = 0.0
    total_frames: int = 0
    resolution: str = ""
    fps: int = 30
