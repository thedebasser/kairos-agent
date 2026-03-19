"""Kairos Agent — Abstract Base Classes for Agents.

These ABCs define the contract that every pipeline's agents must implement.
mypy strict mode + these ABCs make it structurally impossible to add a pipeline
without implementing the required methods.

Agent logic is plain Python — LangGraph calls these methods but the agents
have no dependency on LangGraph abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from kairos.models.contracts import (
    AudioReviewResult,
    CaptionSet,
    ConceptBrief,
    IdeaAgentInput,
    MusicTrackMetadata,
    PipelineState,
    SimulationLoopResult,
    SimulationResult,
    SimulationStats,
    ValidationResult,
    VideoOutput,
    VideoReviewResult,
)


class BaseIdeaAgent(ABC):
    """Abstract Idea Agent — generates concepts for video production.

    Subagents:
    - Inventory Analyst (SQL queries)
    - Category Selector (local LLM, rule-based)
    - Concept Developer (cloud LLM, creative)
    - Trend Scout (optional, periodic)
    """

    @abstractmethod
    async def generate_concept(self, input: IdeaAgentInput) -> ConceptBrief:
        """Generate a single production-ready concept.

        Must respect category rotation rules:
        - Hard block: no repeat of previous category
        - Soft block: deprioritise categories >30% of last 30 days
        - Boost: categories with <5 total videos
        - Streak break: force switch after 3 consecutive same-category

        Args:
            input: Narrow DTO with only the fields needed (e.g. pipeline name).
        """
        ...

    @abstractmethod
    async def get_category_stats(self, pipeline: str) -> dict[str, int]:
        """Get current category distribution for rotation logic."""
        ...


class BaseSimulationAgent(ABC):
    """Abstract Simulation Agent — writes, iterates, validates, and renders.

    Loop: Write code → Execute in sandbox → Validate → Adjust → Re-run
    Max iterations: 5 (configurable via pipeline_config)
    """

    @abstractmethod
    async def generate_simulation(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate initial simulation code from concept brief.

        Returns the complete Python source code for the simulation.
        """
        ...

    @abstractmethod
    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute simulation code in the Docker sandbox.

        Returns execution result including stdout, stderr, and output files.
        """
        ...

    @abstractmethod
    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run all validation checks (Tier 1 programmatic + Tier 2 AI-assisted).

        Tier 1 (mandatory): duration, FPS, frame count, resolution, motion, colour
        Tier 2 (optional): Moondream2 frame inspection
        """
        ...

    @abstractmethod
    async def adjust_parameters(
        self,
        code: str,
        validation_result: ValidationResult,
        iteration: int,
    ) -> str:
        """Adjust simulation parameters based on validation failures.

        Uses local LLM (Mistral 7B) for mechanical parameter edits.
        Escalates to Claude Sonnet for complex debugging.
        """
        ...

    @abstractmethod
    async def get_simulation_stats(self, video_path: str) -> SimulationStats:
        """Extract statistics from a rendered simulation video."""
        ...

    @abstractmethod
    async def run_loop(
        self,
        concept: ConceptBrief,
    ) -> SimulationLoopResult:
        """Run the full simulation iteration loop.

        generate → execute → validate → adjust → repeat
        Returns a narrow ``SimulationLoopResult`` containing only the
        fields this step produces (Finding 2.2).
        """
        ...


class BaseVideoEditorAgent(ABC):
    """Abstract Video Editor Agent — assembles final video.

    No voiceover. Music and minimal captions only.
    FFmpeg for all composition.
    """

    @abstractmethod
    async def select_music(
        self,
        concept: ConceptBrief,
        stats: SimulationStats,
    ) -> MusicTrackMetadata:
        """Select a music track from the curated library.

        Programmatic — no LLM. Tag/mood-based filtering.
        """
        ...

    @abstractmethod
    async def generate_captions(
        self,
        concept: ConceptBrief,
        *,
        theme_name: str = "",
    ) -> CaptionSet:
        """Generate captions for the video.

        POC: Hook caption only (0-2s, ≤6 words).
        Uses Claude Sonnet for hook quality.
        """
        ...

    @abstractmethod
    async def generate_title(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate video title.

        Uses local LLM (Llama 3.1 8B).
        """
        ...

    @abstractmethod
    async def compose_video(
        self,
        raw_video_path: str,
        music: MusicTrackMetadata,
        captions: CaptionSet,
        concept: ConceptBrief,
    ) -> VideoOutput:
        """FFmpeg assembly: raw video + music + captions + watermark.

        Output: 9:16, correct codec, 62-68s duration.
        Audio: music at -18dB, fade out last 3s.
        Caption: Inter Bold, white with black stroke, lower third.
        """
        ...


class BaseVideoReviewAgent(ABC):
    """Abstract Video Review Agent — inspects rendered clips for quality issues.

    Reviews the final composed video (post-editing) and produces a structured
    pass/fail finding. Supports escalation to a heavier model for uncertain clips.

    Checks include:
    - Broken physics simulation (clipping, unrealistic trajectories)
    - Objects stopping unexpectedly or flying off screen
    - Bad framing / camera angle
    - Caption/text overlay placement issues
    - Overall visual polish
    """

    @abstractmethod
    async def review_video(
        self,
        video_path: str,
        concept: ConceptBrief | None = None,
    ) -> VideoReviewResult:
        """Review a rendered video clip for quality issues.

        Uses the default VLM model (e.g. Qwen3-VL-8B). If confidence
        is below the escalation threshold, the implementation should
        automatically re-review with the escalation model.

        Args:
            video_path: Path to the final assembled MP4 video.
            concept: Optional concept brief for context-aware review.

        Returns:
            Structured review result with pass/fail and issue list.
        """
        ...

    @abstractmethod
    async def run_pre_checks(
        self,
        video_path: str,
    ) -> dict[str, float]:
        """Run optional lightweight pre-check tools (DOVER, aesthetic scoring).

        Returns a dict of score names to values. Empty dict if pre-checks
        are disabled. Pre-check failures may short-circuit the VLM review.

        Args:
            video_path: Path to the video file.

        Returns:
            Score dict, e.g. {'dover_technical': 0.7, 'aesthetic': 5.2}.
        """
        ...


class BaseAudioReviewAgent(ABC):
    """Abstract Audio Review Agent — inspects final composed audio for quality issues.

    Reviews the audio track of the final video (music + TTS + SFX mix).
    FFmpeg loudness analysis always runs regardless of which model/stack
    is selected.

    Checks include:
    - Background static / noise artifacts
    - Unexpected sounds
    - TTS accuracy (word error rate)
    - Theme/vibe match
    - Volume level consistency
    - LUFS loudness compliance
    """

    @abstractmethod
    async def review_audio(
        self,
        audio_path: str,
        expected_transcript: str = "",
    ) -> AudioReviewResult:
        """Review composed audio for quality issues.

        Always runs FFmpeg loudness analysis. Primary review model
        is configurable (omni-modal LLM, specialist stack, etc.).

        Args:
            audio_path: Path to the audio file or final video (audio extracted via FFmpeg).
            expected_transcript: Expected TTS transcript for WER check.

        Returns:
            Structured review result with pass/fail, issues, and loudness metrics.
        """
        ...

    @abstractmethod
    async def measure_loudness(
        self,
        audio_path: str,
    ) -> dict[str, float]:
        """Run FFmpeg ebur128 loudness measurement.

        This always runs regardless of which primary reviewer is active.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dict with 'integrated_lufs', 'true_peak_dbtp', 'loudness_range_lu'.
        """
        ...


class BasePipelineAdapter(ABC):
    """Abstract Pipeline Adapter — engine-specific logic.

    Each pipeline (physics, beamng, marble, etc.) implements this adapter.
    The adapter provides engine-specific agents and configuration.
    """

    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Unique pipeline identifier (e.g., 'physics', 'beamng')."""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Simulation engine name (e.g., 'pygame_pymunk', 'beamng')."""
        ...

    @property
    @abstractmethod
    def categories(self) -> list[str]:
        """Available scenario categories for this pipeline."""
        ...

    @abstractmethod
    def get_idea_agent(self) -> BaseIdeaAgent:
        """Return the Idea Agent configured for this pipeline."""
        ...

    @abstractmethod
    def get_simulation_agent(self) -> BaseSimulationAgent:
        """Return the Simulation Agent configured for this pipeline."""
        ...

    @abstractmethod
    def get_video_editor_agent(self) -> BaseVideoEditorAgent:
        """Return the Video Editor Agent configured for this pipeline."""
        ...

    @abstractmethod
    def get_video_review_agent(self) -> BaseVideoReviewAgent:
        """Return the Video Review Agent configured for this pipeline."""
        ...

    @abstractmethod
    def get_audio_review_agent(self) -> BaseAudioReviewAgent:
        """Return the Audio Review Agent configured for this pipeline."""
        ...

    @abstractmethod
    def get_sandbox_dockerfile(self) -> str:
        """Return path to the sandbox Dockerfile for this pipeline's engine."""
        ...

    @abstractmethod
    def get_prompt_template(self, category: str) -> str:
        """Return the simulation prompt template for a given category."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if all pipeline dependencies are available."""
        ...
