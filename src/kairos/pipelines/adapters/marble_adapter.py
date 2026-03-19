"""Marble Pipeline Adapter.

Implements PipelineAdapter for the Blender marble course pipeline.
Registered as "marble" via the @register_pipeline decorator.
"""

from __future__ import annotations

from pathlib import Path

from kairos.pipelines.contracts import (
    AudioReviewAgent,
    IdeaAgent,
    PipelineAdapter,
    SimulationAgent,
    VideoEditorAgent,
    VideoReviewAgent,
)
from kairos.schemas.contracts import MarbleArchetype
from kairos.orchestrator.registry import register_pipeline
from kairos.engines.blender.executor import find_blender


@register_pipeline("marble")
class MarblePipelineAdapter(PipelineAdapter):
    """Pipeline adapter for Blender marble course simulations.

    Pipeline 2 — "Marble Courses"
    Engine: Blender 5.x rigid body physics
    Archetypes: funnel_race, race_lane, peg_maze
    """

    @property
    def pipeline_name(self) -> str:
        return "marble"

    @property
    def engine_name(self) -> str:
        return "blender"

    @property
    def categories(self) -> list[str]:
        return [a.value for a in MarbleArchetype]

    def get_idea_agent(self) -> IdeaAgent:
        """Return the marble idea agent."""
        from kairos.pipelines.marble.idea_agent import MarbleIdeaAgent

        return MarbleIdeaAgent()

    def get_simulation_agent(self) -> SimulationAgent:
        """Return the marble simulation agent."""
        from kairos.pipelines.marble.simulation_agent import MarbleSimulationAgent

        return MarbleSimulationAgent()

    def get_video_editor_agent(self) -> VideoEditorAgent:
        """Return the marble video editor agent."""
        from kairos.pipelines.marble.video_editor_agent import MarbleVideoEditorAgent

        return MarbleVideoEditorAgent()

    def get_video_review_agent(self) -> VideoReviewAgent:
        """Return the shared video review agent."""
        from kairos.ai.review.video_review_agent import VideoReviewAgent

        return VideoReviewAgent()

    def get_audio_review_agent(self) -> AudioReviewAgent:
        """Return the shared audio review agent."""
        from kairos.ai.review.audio_review_agent import AudioReviewAgent

        return AudioReviewAgent()

    def get_sandbox_dockerfile(self) -> str:
        """No Docker sandbox needed — Blender runs natively."""
        return ""

    def get_prompt_template(self, category: str) -> str:
        """Marble pipeline uses inline prompts, not template files."""
        return f"Marble course generation for archetype: {category}"

    async def health_check(self) -> bool:
        """Check if Blender is available on the system."""
        blender = find_blender()
        if blender is None:
            return False
        # Also check that blend/scripts/ directory exists
        scripts_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "blend" / "scripts"
        return scripts_dir.exists()
