"""Pydantic models for the creative pipeline.

Defines the data contracts exchanged between Set Designer, Path Setter,
and Connector Agent.  All output models are frozen for immutability.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────


class EnvironmentType(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class SegmentType(str, Enum):
    """Types of path segment between waypoints."""

    FLAT_SURFACE = "flat_surface"
    HEIGHT_TRANSITION_UP = "height_transition_up"
    HEIGHT_TRANSITION_DOWN = "height_transition_down"
    GROUND_LEVEL = "ground_level"


class ConnectorType(str, Enum):
    """Available connector primitives from the skill library."""

    RAMP = "ramp"
    SPIRAL_RAMP = "spiral_ramp"
    STAIRCASE = "staircase"
    PLATFORM = "platform"
    PLANK_BRIDGE = "plank_bridge"


class ObjectRole(str, Enum):
    """Role of a placed object in the scene."""

    FUNCTIONAL = "functional"  # Course travels on/through this object
    DECORATIVE = "decorative"  # Visual filler, no physics interaction


class AgentRole(str, Enum):
    """Identifies which creative agent produced an output."""

    SET_DESIGNER = "set_designer"
    PATH_SETTER = "path_setter"
    CONNECTOR = "connector"
    CAMERA_ROUTER = "camera_router"
    FINAL_REVIEWER = "final_reviewer"


# ─── Scene Manifest (Set Designer output) ────────────────────────────


class GroundConfig(BaseModel, frozen=True):
    texture: str = Field(description="Ground texture name or path")
    material_params: dict[str, Any] = Field(default_factory=dict)


class LightingConfig(BaseModel, frozen=True):
    preset: str = Field(description="Lighting preset name")
    params: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel, frozen=True):
    type: EnvironmentType = Field(description="Indoor or outdoor scene")
    ground: GroundConfig
    lighting: LightingConfig
    hdri: str | None = Field(
        default=None, description="HDRI name for outdoor scenes"
    )


class PlacedObject(BaseModel, frozen=True):
    """A single object placed in the scene by the Set Designer."""

    asset_id: str = Field(description="ID from the asset catalogue")
    name: str = Field(default="", description="Human-readable name")
    position: tuple[float, float, float] = Field(
        description="World position (x, y, z)"
    )
    rotation: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Euler rotation in degrees"
    )
    scale: float = Field(default=1.0, ge=0.1, le=10.0)
    role: ObjectRole = Field(default=ObjectRole.DECORATIVE)
    surface_name: str | None = Field(
        default=None,
        description="If functional, which surface the course uses",
    )


class SceneManifest(BaseModel, frozen=True):
    """Complete scene specification produced by the Set Designer.

    Consumed by the Path Setter to route domino paths through and
    around the placed objects.
    """

    theme: str = Field(description="Scene theme (e.g. 'modern_kitchen')")
    narrative: str = Field(
        description="1-2 sentence visual story the viewer experiences"
    )
    environment: EnvironmentSpec
    objects: list[PlacedObject] = Field(
        default_factory=list,
        description="Objects placed in the scene",
    )
    domino_count: int = Field(
        default=300, ge=50, le=1000,
        description="Target number of dominoes",
    )


# ─── Path Output (Path Setter output) ────────────────────────────────


class Waypoint(BaseModel, frozen=True):
    x: float
    y: float
    z: float


class PathSegment(BaseModel, frozen=True):
    """A segment of the domino path between surface transitions."""

    id: str = Field(description="Unique segment identifier")
    type: SegmentType
    surface_ref: str | None = Field(
        default=None,
        description="Reference to the surface (asset_id.surface_name)",
    )
    waypoints: list[Waypoint] = Field(default_factory=list)
    from_height: float = Field(default=0.0)
    to_height: float = Field(default=0.0)
    needs_connector: bool = Field(
        default=False,
        description="True if this segment is a height transition requiring a connector",
    )
    connector_hint: str = Field(
        default="",
        description="Suggested connector type for this transition",
    )
    available_footprint: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Available space (width, depth) for the connector",
    )
    gradient: float = Field(
        default=0.0,
        description="Gradient angle in degrees",
    )


class PathOutput(BaseModel, frozen=True):
    """Complete path specification produced by the Path Setter.

    Consumed by the Connector Agent to fill height transitions.
    """

    total_length_estimate: float = Field(
        description="Estimated total path length in metres"
    )
    segments: list[PathSegment] = Field(description="Ordered path segments")
    domino_count: int = Field(description="Target domino count from manifest")


# ─── Connector Output (Connector Agent output) ───────────────────────


class CalibrationRef(BaseModel, frozen=True):
    """Reference to a ChromaDB calibration used for a connector."""

    source: str = Field(
        default="baseline",
        description="Where the calibration came from (baseline, chromadb, composite)",
    )
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    spacing_correction: float = Field(default=0.0)
    friction_correction: float = Field(default=0.0)
    trigger_correction: float = Field(default=0.0)


class ResolvedConnector(BaseModel, frozen=True):
    """A connector chosen and parameterised by the Connector Agent."""

    id: str = Field(description="Unique connector identifier")
    for_segment: str = Field(description="ID of the PathSegment this fills")
    type: ConnectorType
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to the skill library primitive",
    )
    calibration: CalibrationRef = Field(default_factory=CalibrationRef)
    generated_waypoints: list[Waypoint] = Field(
        default_factory=list,
        description="Waypoints generated by the connector primitive",
    )


class ConnectorOutput(BaseModel, frozen=True):
    """Complete connector specification produced by the Connector Agent.

    This plus the PathOutput forms the fully-connected domino path
    ready for Blender scene generation.
    """

    connectors: list[ResolvedConnector] = Field(
        default_factory=list,
        description="Resolved connectors for all height transitions",
    )
    complete_path_waypoints: list[Waypoint] = Field(
        default_factory=list,
        description="Full flattened waypoint list (path + connectors merged)",
    )
    segment_types: list[str] = Field(
        default_factory=list,
        description="Type label for each segment in the complete path",
    )


# ─── Validation Result ───────────────────────────────────────────────


class StepValidationResult(BaseModel, frozen=True):
    """Structured result from per-step validation."""

    agent: AgentRole = Field(description="Which agent's output was validated")
    passed: bool
    checks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Individual validation checks with pass/fail and messages",
    )
    error_summary: str = Field(
        default="", description="Human-readable summary of failures"
    )


# ─── Iteration Feedback ─────────────────────────────────────────────


class AttemptRecord(BaseModel, frozen=True):
    """Record of a single agent attempt within a pipeline iteration."""

    agent: AgentRole
    attempt: int
    passed: bool
    error_summary: str = ""
    params_snapshot: dict[str, Any] = Field(default_factory=dict)


class IterationHistory(BaseModel):
    """Tracks all attempts across the creative pipeline for feedback formatting."""

    attempts: list[AttemptRecord] = Field(default_factory=list)
    total_pipeline_attempts: int = 0

    def add(self, record: AttemptRecord) -> None:
        self.attempts.append(record)
        self.total_pipeline_attempts += 1

    def for_agent(self, agent: AgentRole) -> list[AttemptRecord]:
        return [a for a in self.attempts if a.agent == agent]

    def latest_for_agent(self, agent: AgentRole) -> AttemptRecord | None:
        records = self.for_agent(agent)
        return records[-1] if records else None

    def format_feedback(self, agent: AgentRole) -> str:
        """Format iteration history as human-readable feedback for an agent."""
        records = self.for_agent(agent)
        if not records:
            return "This is your first attempt."

        lines = [f"Previous attempts ({len(records)} total):"]
        for rec in records:
            status = "PASSED" if rec.passed else "FAILED"
            lines.append(f"  Attempt {rec.attempt}: {status}")
            if rec.error_summary:
                lines.append(f"    Issues: {rec.error_summary}")
        lines.append(
            f"\nTotal pipeline attempts so far: {self.total_pipeline_attempts}"
        )
        return "\n".join(lines)


# ─── Camera Models ───────────────────────────────────────────────────


class CameraKeyframe(BaseModel, frozen=True):
    """A single camera keyframe in the tracking trajectory."""

    frame: int
    position: tuple[float, float, float]
    look_target: tuple[float, float, float]


class OcclusionEvent(BaseModel, frozen=True):
    """A detected occlusion between camera and wavefront."""

    frame_start: int
    frame_end: int
    occluder: str = Field(description="Name/id of the occluding object")
    severity: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="0=minor partial, 1=fully blocked",
    )


class CameraOutput(BaseModel, frozen=True):
    """Camera trajectory produced by the Camera Router.

    Contains keyframes for tracking + repositioning + pull-back,
    plus any occlusion events that were resolved.
    """

    keyframes: list[CameraKeyframe] = Field(default_factory=list)
    occlusion_events: list[OcclusionEvent] = Field(default_factory=list)
    repositions: int = Field(default=0, description="Number of occlusion repositions")
    follow_distance: float = Field(default=3.5)
    camera_height: float = Field(default=4.5)
    total_frames: int = Field(default=0)


class CameraValidationResult(BaseModel, frozen=True):
    """Result of camera trajectory validation."""

    visibility_ratio: float = Field(
        description="Fraction of frames where wavefront is visible (0-1)",
    )
    max_velocity_spike: float = Field(
        default=0.0,
        description="Maximum velocity relative to average (>2.0 is bad)",
    )
    smooth_motion: bool = Field(default=True)
    occlusion_frames: int = Field(default=0)
    passed: bool = Field(default=True)
    issues: list[str] = Field(default_factory=list)


class ReviewIssue(BaseModel, frozen=True):
    """A single issue identified by the Final Reviewer."""

    description: str
    attributed_to: AgentRole
    reason: str = ""
    suggested_fix: str = ""
    severity: str = Field(default="blocking", description="blocking | warning")


class FinalReviewResult(BaseModel, frozen=True):
    """Final Reviewer assessment of a complete rendered run."""

    passed: bool
    issues: list[ReviewIssue] = Field(default_factory=list)
    cascade_from: AgentRole | None = Field(
        default=None,
        description="If failed, which agent to cascade re-run from",
    )
    summary: str = Field(default="")
