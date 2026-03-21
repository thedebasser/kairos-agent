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


class MarbleArchetype(str, Enum):
    """Archetype categories for the Blender marble course pipeline."""

    FUNNEL_RACE = "funnel_race"
    RACE_LANE = "race_lane"
    PEG_MAZE = "peg_maze"


class DominoArchetype(str, Enum):
    """Archetype categories for the Blender domino run pipeline."""

    SPIRAL = "spiral"
    S_CURVE = "s_curve"
    BRANCHING = "branching"
    WORD_SPELL = "word_spell"
    CASCADE = "cascade"


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


# ---------------------------------------------------------------------------
# Structured validation feedback (AI review §2)
# ---------------------------------------------------------------------------


class FailedCheck(BaseModel, frozen=True):
    """One failed validation check enriched with quantitative delta."""

    check_name: str
    actual: float | None = None
    target_min: float | None = None
    target_max: float | None = None
    delta: float | None = None
    suggested_fix: str = ""
    historical_note: str = ""


class PastFix(BaseModel, frozen=True):
    """A fix from a previous run that resolved the same check."""

    category: str
    check_name: str
    parameter_changed: str
    old_value: str
    new_value: str
    worked: bool = True


class ValidationFeedback(BaseModel, frozen=True):
    """Structured feedback for the LLM adjustment step (AI review §2)."""

    failed_checks: list[FailedCheck] = Field(default_factory=list)
    suggested_parameter_changes: dict[str, str] = Field(default_factory=dict)
    similar_past_fixes: list[PastFix] = Field(default_factory=list)
    iteration: int = 1
    max_iterations: int = 5
    urgency: str = Field(
        default="minor_tweak",
        description="'minor_tweak' | 'significant_change' | 'fundamental_rethink'",
    )
    iteration_history_summary: str = Field(
        default="",
        description="Summary of what was already tried in earlier iterations.",
    )

    def to_prompt_text(self) -> str:
        """Render this feedback as a prompt section for the LLM."""
        lines: list[str] = [f"### Structured Validation Feedback (iteration {self.iteration}/{self.max_iterations})"]
        lines.append(f"**Urgency:** {self.urgency}")
        if self.failed_checks:
            lines.append("\n**Failed checks:**")
            for fc in self.failed_checks:
                parts = [f"- **{fc.check_name}**: actual={fc.actual}"]
                if fc.target_min is not None or fc.target_max is not None:
                    parts.append(f"  target=[{fc.target_min}, {fc.target_max}]")
                if fc.delta is not None:
                    parts.append(f"  delta={fc.delta:+.2f}")
                if fc.suggested_fix:
                    parts.append(f"  → {fc.suggested_fix}")
                if fc.historical_note:
                    parts.append(f"  (historical: {fc.historical_note})")
                lines.append(" ".join(parts))
        if self.suggested_parameter_changes:
            lines.append("\n**Suggested parameter changes:**")
            for param, suggestion in self.suggested_parameter_changes.items():
                lines.append(f"- `{param}`: {suggestion}")
        if self.similar_past_fixes:
            lines.append("\n**Fixes that worked in similar past runs:**")
            for pf in self.similar_past_fixes[:3]:
                lines.append(f"- {pf.parameter_changed}: {pf.old_value} → {pf.new_value} ({pf.category})")
        if self.iteration_history_summary:
            lines.append(f"\n**Already tried this run:**\n{self.iteration_history_summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category knowledge (AI review §6)
# ---------------------------------------------------------------------------


class CategoryKnowledge(BaseModel):
    """Accumulated knowledge about what works for a category.

    Stored as JSONB in ``category_stats.knowledge`` and injected into
    generation prompts for returning categories.
    """

    category: str = ""
    parameter_ranges: dict[str, list[float]] = Field(
        default_factory=dict,
        description="param_name → [min_that_worked, max_that_worked]",
    )
    common_failure_modes: list[str] = Field(default_factory=list)
    avg_iterations_to_pass: float = 0.0
    successful_code_patterns: list[str] = Field(
        default_factory=list,
        description="Short descriptions of approaches that worked",
    )
    failed_code_patterns: list[str] = Field(
        default_factory=list,
        description="Approaches that consistently fail",
    )
    best_duration_setting: float = 0.0
    total_examples: int = 0

    def to_prompt_text(self) -> str:
        """Render as a prompt section for injection into code-generation prompts."""
        if self.total_examples == 0:
            return ""
        lines = [f"### What we know about '{self.category}' (from {self.total_examples} past runs)"]
        if self.best_duration_setting:
            lines.append(f"- Best SIMULATION_TIME setting: {self.best_duration_setting}")
        if self.avg_iterations_to_pass:
            lines.append(f"- Average iterations to pass validation: {self.avg_iterations_to_pass:.1f}")
        if self.parameter_ranges:
            lines.append("- Known good parameter ranges:")
            for param, rng in self.parameter_ranges.items():
                lines.append(f"  - {param}: {rng[0]} – {rng[1]}")
        if self.common_failure_modes:
            lines.append("- Common failure modes to avoid:")
            for fm in self.common_failure_modes[:5]:
                lines.append(f"  - {fm}")
        if self.successful_code_patterns:
            lines.append("- Approaches that work:")
            for sp in self.successful_code_patterns[:3]:
                lines.append(f"  - {sp}")
        return "\n".join(lines)


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


# =============================================================================
# Review Data Contracts
# =============================================================================


class ReviewIssueSeverity(str, Enum):
    """Severity level for a review issue."""

    CRITICAL = "critical"    # auto-reject
    MAJOR = "major"          # likely reject unless only finding
    MINOR = "minor"          # warning, does not cause rejection
    INFO = "info"            # informational note


class ReviewIssue(BaseModel, frozen=True):
    """Single issue found during video or audio review."""

    category: str = Field(description="Issue category, e.g. 'broken_physics', 'bad_framing', 'audio_artifact'")
    severity: ReviewIssueSeverity = Field(default=ReviewIssueSeverity.MAJOR)
    description: str = Field(description="Human-readable description of the issue")
    timestamp_sec: float | None = Field(default=None, description="Approximate timestamp in the video/audio where the issue occurs")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model confidence in this finding (0-1)")


class VideoReviewResult(BaseModel, frozen=True):
    """Structured result from the Video Review Agent."""

    passed: bool = Field(description="True if the video passed review with no critical/major issues")
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall reviewer confidence (0-1). Low confidence triggers escalation.")
    issues: list[ReviewIssue] = Field(default_factory=list, description="List of identified issues")
    model_used: str = Field(default="", description="Model alias that produced this review")
    escalated: bool = Field(default=False, description="Whether this review was escalated to the heavier model")
    pre_check_scores: dict[str, float] = Field(default_factory=dict, description="Optional pre-check scores, e.g. {'dover_technical': 0.7, 'aesthetic': 5.2}")

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        critical = sum(1 for i in self.issues if i.severity == ReviewIssueSeverity.CRITICAL)
        major = sum(1 for i in self.issues if i.severity == ReviewIssueSeverity.MAJOR)
        return f"{'PASS' if self.passed else 'FAIL'} | {critical} critical, {major} major issues | confidence={self.overall_confidence:.2f}"


class LoudnessMetrics(BaseModel, frozen=True):
    """FFmpeg ebur128 loudness measurement results."""

    integrated_lufs: float = Field(description="Integrated loudness in LUFS")
    true_peak_dbtp: float = Field(description="True peak in dBTP")
    loudness_range_lu: float = Field(default=0.0, description="Loudness range in LU")
    passed: bool = Field(default=True, description="Whether loudness is within acceptable range")
    details: str = Field(default="", description="Additional measurement details")


class AudioReviewResult(BaseModel, frozen=True):
    """Structured result from the Audio Review Agent."""

    passed: bool = Field(description="True if the audio passed review")
    issues: list[ReviewIssue] = Field(default_factory=list, description="List of identified issues")
    model_used: str = Field(default="", description="Model alias or 'specialist_stack'")
    loudness: LoudnessMetrics | None = Field(default=None, description="FFmpeg loudness measurement (always populated)")
    tts_wer: float | None = Field(default=None, description="TTS word error rate (0-1) if transcript check was run")
    dnsmos_scores: dict[str, float] | None = Field(default=None, description="DNSMOS P.835 scores: sig, bak, ovrl (1-5 MOS)")

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        critical = sum(1 for i in self.issues if i.severity == ReviewIssueSeverity.CRITICAL)
        major = sum(1 for i in self.issues if i.severity == ReviewIssueSeverity.MAJOR)
        lufs_str = f" | {self.loudness.integrated_lufs:.1f} LUFS" if self.loudness else ""
        return f"{'PASS' if self.passed else 'FAIL'} | {critical} critical, {major} major issues{lufs_str}"


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


class IdeaAgentInput(BaseModel, frozen=True):
    """Narrow input DTO for the Idea Agent (Finding 2.2).

    Only exposes the fields that ``generate_concept`` actually reads,
    rather than passing the entire 24-field ``PipelineState``.
    """

    pipeline: str = "physics"


class SimulationLoopResult(BaseModel):
    """Narrow output DTO from the simulation iteration loop (Finding 2.2).

    Contains only the fields that ``run_loop`` writes.  The graph node
    is responsible for mapping this back into the ``PipelineGraphState``.
    """

    simulation_code: str = ""
    simulation_result: SimulationResult | None = None
    simulation_stats: SimulationStats | None = None
    validation_result: ValidationResult | None = None
    simulation_iteration: int = 0
    raw_video_path: str = ""
    errors: list[str] = Field(default_factory=list)


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
    theme_name: str = ""

    # Video editing phase
    captions: CaptionSet | None = None
    music_track: MusicTrackMetadata | None = None
    final_video_path: str = ""
    video_output: VideoOutput | None = None

    # Review phase — automated
    video_review_result: VideoReviewResult | None = None
    audio_review_result: AudioReviewResult | None = None
    video_review_attempts: int = 0
    audio_review_attempts: int = 0

    # Review phase — human
    review_action: ReviewAction | None = None
    review_feedback: str = ""

    # Output versioning — tracks review-triggered re-renders (§1)
    output_version: int = 1

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
