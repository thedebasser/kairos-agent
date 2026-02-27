"""Kairos Agent — Pydantic Data Contracts.

All data contracts used across the pipeline. These are the shared interfaces
between agents — validated via Instructor for LLM outputs and used as the
canonical data shapes throughout the system.

Every model here is immutable (frozen=True) to prevent accidental mutation
during pipeline execution.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class PipelineStatus(str, Enum):
    """Pipeline run status."""

    RUNNING = "running"
    IDEA_PHASE = "idea_phase"
    SIMULATION_PHASE = "simulation_phase"
    EDITING_PHASE = "editing_phase"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IdeaStatus(str, Enum):
    """Video idea status."""

    PENDING = "pending"
    IN_PRODUCTION = "in_production"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    CANCELLED = "cancelled"


class OutputStatus(str, Enum):
    """Output video status."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"


class ReviewAction(str, Enum):
    """Human review action codes."""

    APPROVED = "approved"
    BAD_CONCEPT = "bad_concept"
    BAD_SIMULATION = "bad_simulation"
    BAD_EDIT = "bad_edit"
    REQUEST_REEDIT = "request_reedit"


class PublishStatus(str, Enum):
    """Publish queue status."""

    QUEUED = "queued"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRunStatus(str, Enum):
    """Agent execution status."""

    SUCCESS = "success"
    FAILED = "failed"
    RETRIED = "retried"
    ESCALATED = "escalated"


class CaptionType(str, Enum):
    """Caption type for video overlay."""

    HOOK = "hook"
    RULE = "rule"
    TENSION = "tension"
    PAYOFF = "payoff"


class EnergyLevel(str, Enum):
    """Energy curve descriptor for audio/visual matching."""

    LOW = "low"
    BUILDING = "building"
    HIGH = "high"
    CLIMAX = "climax"


class ScenarioCategory(str, Enum):
    """POC scenario categories for Oddly Satisfying Physics pipeline."""

    BALL_PIT = "ball_pit"
    MARBLE_FUNNEL = "marble_funnel"
    DOMINO_CHAIN = "domino_chain"
    DESTRUCTION = "destruction"


# =============================================================================
# Core Data Contracts
# =============================================================================


class AudioBrief(BaseModel, frozen=True):
    """Audio requirements extracted from concept brief."""

    mood: list[str] = Field(description="Mood tags for music selection, e.g. ['upbeat', 'energetic']")
    tempo_bpm_min: int = Field(default=90, description="Minimum BPM for track selection")
    tempo_bpm_max: int = Field(default=140, description="Maximum BPM for track selection")
    energy_curve: EnergyLevel = Field(
        default=EnergyLevel.BUILDING,
        description="Overall energy progression of the video",
    )


class SimulationRequirements(BaseModel, frozen=True):
    """Technical requirements for the simulation engine."""

    body_count_initial: int = Field(description="Starting number of physics bodies")
    body_count_max: int = Field(description="Maximum physics bodies at any point")
    interaction_type: str = Field(description="Primary physics interaction type")
    colour_palette: list[str] = Field(
        default_factory=lambda: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        description="Hex colour codes for physics bodies",
    )
    background_colour: str = Field(default="#1a1a2e", description="Background hex colour")
    special_effects: list[str] = Field(
        default_factory=list,
        description="Optional visual effects: trails, glow, particles",
    )


class ConceptBrief(BaseModel, frozen=True):
    """Output of the Idea Agent — full concept for production.

    Validated via Instructor from LLM output.
    """

    concept_id: UUID = Field(default_factory=uuid4)
    pipeline: str = Field(default="physics")
    category: ScenarioCategory
    title: str = Field(description="Short descriptive title for the concept")
    visual_brief: str = Field(
        description="2-3 sentence description of what the viewer sees",
    )
    simulation_requirements: SimulationRequirements
    audio_brief: AudioBrief
    hook_text: str = Field(
        description="Hook caption text (≤6 words) shown at 0-2s",
        max_length=50,
    )

    @field_validator("hook_text")
    @classmethod
    def hook_text_max_words(cls, v: str) -> str:
        """Hook text must be ≤6 words."""
        word_count = len(v.split())
        if word_count > 6:
            msg = f"Hook text must be ≤6 words, got {word_count}"
            raise ValueError(msg)
        return v

    novelty_score: float = Field(
        ge=0.0,
        le=10.0,
        description="How novel this concept is vs existing content (0-10)",
    )
    feasibility_score: float = Field(
        ge=0.0,
        le=10.0,
        description="How feasible to implement in the simulation engine (0-10)",
    )
    target_duration_sec: int = Field(default=65, ge=62, le=68)
    seed: int | None = Field(
        default=None,
        description="Random seed for deterministic reproduction",
    )


class SimulationStats(BaseModel, frozen=True):
    """Statistics from a simulation execution."""

    duration_sec: float
    peak_body_count: int
    avg_fps: float
    min_fps: float
    payoff_timestamp_sec: float = Field(
        description="Timestamp where the 'climax' or payoff begins",
    )
    total_frames: int
    file_size_bytes: int


class SimulationResult(BaseModel, frozen=True):
    """Result of executing simulation code in the sandbox."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    output_files: list[str] = Field(default_factory=list)
    stats: SimulationStats | None = None
    execution_time_sec: float = 0.0


class ValidationCheck(BaseModel, frozen=True):
    """Single validation check result."""

    name: str
    passed: bool
    message: str = ""
    value: Any = None  # noqa: ANN401
    threshold: Any = None  # noqa: ANN401


class ValidationResult(BaseModel, frozen=True):
    """Aggregated result of all validation checks on a simulation output."""

    passed: bool = Field(description="True if ALL mandatory checks passed")
    checks: list[ValidationCheck] = Field(default_factory=list)
    tier1_passed: bool = Field(
        default=False,
        description="All programmatic (Tier 1) checks passed",
    )
    tier2_passed: bool | None = Field(
        default=None,
        description="AI-assisted (Tier 2) checks passed (None if not run)",
    )

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    @property
    def summary(self) -> str:
        """Human-readable summary of validation results."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        return f"{passed}/{total} checks passed"


class Caption(BaseModel, frozen=True):
    """Single caption overlay for the video."""

    caption_type: CaptionType
    text: str = Field(max_length=50, description="Caption text (≤6 words)")
    start_sec: float = Field(ge=0.0, description="Start time in seconds")
    end_sec: float = Field(ge=0.0, description="End time in seconds")


class CaptionSet(BaseModel, frozen=True):
    """Set of captions for a video. POC: hook only."""

    captions: list[Caption] = Field(
        default_factory=list,
        description="Ordered list of captions",
    )

    @property
    def hook(self) -> Caption | None:
        """Get the hook caption (first caption of type HOOK)."""
        return next((c for c in self.captions if c.caption_type == CaptionType.HOOK), None)


class MusicTrackMetadata(BaseModel, frozen=True):
    """Metadata for a music track from the curated library."""

    track_id: str
    filename: str
    source: str = "pixabay"
    pixabay_id: int | None = None
    artist: str = ""
    license: str = "pixabay_content_license"
    duration_sec: float
    bpm: int
    mood: list[str]
    energy_curve: str
    genre: str = ""
    contentid_status: str = "unknown"
    last_used_at: datetime | None = None
    use_count: int = 0


class VideoOutput(BaseModel, frozen=True):
    """Final assembled video ready for review."""

    output_id: UUID = Field(default_factory=uuid4)
    pipeline_run_id: UUID
    simulation_id: UUID
    final_video_path: str
    captions: CaptionSet
    music_track: MusicTrackMetadata | None = None
    title: str
    description: str = ""
    stats: SimulationStats | None = None
    validation: ValidationResult | None = None
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Pipeline State (LangGraph)
# =============================================================================


class PipelineState(BaseModel):
    """LangGraph state object — mutable during pipeline execution.

    NOT frozen — LangGraph nodes mutate state as the pipeline progresses.
    """

    pipeline_run_id: UUID = Field(default_factory=uuid4)
    pipeline: str = "physics"
    status: PipelineStatus = PipelineStatus.RUNNING

    # Idea phase
    concept: ConceptBrief | None = None
    concept_attempts: int = 0

    # Simulation phase
    simulation_code: str = ""
    simulation_result: SimulationResult | None = None
    simulation_stats: SimulationStats | None = None
    validation_result: ValidationResult | None = None
    simulation_iteration: int = 0
    raw_video_path: str = ""

    # Video editing phase
    captions: CaptionSet | None = None
    music_track: MusicTrackMetadata | None = None
    final_video_path: str = ""
    video_output: VideoOutput | None = None

    # Review phase
    review_action: ReviewAction | None = None
    review_feedback: str = ""

    # Tracking
    total_cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# Agent Run Tracking
# =============================================================================


class AgentRunRecord(BaseModel, frozen=True):
    """Record of a single agent execution (LLM call or significant operation)."""

    run_id: UUID = Field(default_factory=uuid4)
    pipeline_run_id: UUID
    idea_id: UUID | None = None
    agent_name: str = Field(
        description="'idea_agent', 'simulation_agent', 'video_editor_agent'",
    )
    step_name: str = Field(
        description="'concept_developer', 'frame_inspector', 'caption_writer', etc.",
    )
    model_used: str = Field(
        description="'claude-sonnet-4-6', 'ollama/mistral:7b', 'programmatic', etc.",
    )
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: AgentRunStatus = AgentRunStatus.SUCCESS
    error_message: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
