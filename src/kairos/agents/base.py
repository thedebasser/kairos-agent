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
    CaptionSet,
    ConceptBrief,
    MusicTrackMetadata,
    PipelineState,
    SimulationResult,
    SimulationStats,
    ValidationResult,
    VideoOutput,
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
    async def generate_concept(self, state: PipelineState) -> ConceptBrief:
        """Generate a single production-ready concept.

        Must respect category rotation rules:
        - Hard block: no repeat of previous category
        - Soft block: deprioritise categories >30% of last 30 days
        - Boost: categories with <5 total videos
        - Streak break: force switch after 3 consecutive same-category
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
        state: PipelineState,
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
        state: PipelineState,
    ) -> PipelineState:
        """Run the full simulation iteration loop.

        generate → execute → validate → adjust → repeat
        Returns updated state with simulation result or escalation flag.
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
