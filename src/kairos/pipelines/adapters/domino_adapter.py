"""Domino Pipeline Adapter.

Implements PipelineAdapter for the Blender domino run pipeline.
Registered as "domino" via the @register_pipeline decorator.
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
from kairos.schemas.contracts import DominoArchetype
from kairos.orchestrator.registry import register_pipeline
from kairos.engines.blender.executor import find_blender


@register_pipeline("domino")
class DominoPipelineAdapter(PipelineAdapter):
    """Pipeline adapter for Blender domino run simulations.

    Pipeline 3 — "Domino Runs"
    Engine: Blender 5.x rigid body physics
    Archetypes: spiral, s_curve, branching, word_spell, cascade
    """

    @property
    def pipeline_name(self) -> str:
        return "domino"

    @property
    def engine_name(self) -> str:
        return "blender"

    @property
    def categories(self) -> list[str]:
        return [a.value for a in DominoArchetype]

    def get_idea_agent(self) -> IdeaAgent:
        """Return the domino idea agent."""
        from kairos.pipelines.domino.idea_agent import DominoIdeaAgent

        return DominoIdeaAgent()

    def get_simulation_agent(self) -> SimulationAgent:
        """Return the domino simulation agent."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent

        return DominoSimulationAgent()

    def get_video_editor_agent(self) -> VideoEditorAgent:
        """Return the domino video editor agent."""
        from kairos.pipelines.domino.video_editor_agent import DominoVideoEditorAgent

        return DominoVideoEditorAgent()

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
        """Domino pipeline uses inline prompts, not template files."""
        return f"Domino run generation for archetype: {category}"

    async def health_check(self) -> bool:
        """Check if Blender is available on the system."""
        blender = find_blender()
        if blender is None:
            return False
        scripts_dir = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "blend" / "scripts"
        )
        return scripts_dir.exists() and (scripts_dir / "generate_domino_course.py").exists()
