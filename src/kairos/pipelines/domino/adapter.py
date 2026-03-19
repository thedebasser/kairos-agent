"""Domino Pipeline Adapter.

Implements BasePipelineAdapter for the Blender domino run pipeline.
Registered as "domino" via the @register_pipeline decorator.
"""

from __future__ import annotations

from pathlib import Path

from kairos.agents.base import (
    BaseAudioReviewAgent,
    BaseIdeaAgent,
    BasePipelineAdapter,
    BaseSimulationAgent,
    BaseVideoEditorAgent,
    BaseVideoReviewAgent,
)
from kairos.models.contracts import DominoArchetype
from kairos.pipeline.registry import register_pipeline
from kairos.pipelines.marble.blender_executor import find_blender


@register_pipeline("domino")
class DominoPipelineAdapter(BasePipelineAdapter):
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

    def get_idea_agent(self) -> BaseIdeaAgent:
        """Return the domino idea agent."""
        from kairos.pipelines.domino.idea_agent import DominoIdeaAgent

        return DominoIdeaAgent()

    def get_simulation_agent(self) -> BaseSimulationAgent:
        """Return the domino simulation agent."""
        from kairos.pipelines.domino.simulation_agent import DominoSimulationAgent

        return DominoSimulationAgent()

    def get_video_editor_agent(self) -> BaseVideoEditorAgent:
        """Return the domino video editor agent."""
        from kairos.pipelines.domino.video_editor_agent import DominoVideoEditorAgent

        return DominoVideoEditorAgent()

    def get_video_review_agent(self) -> BaseVideoReviewAgent:
        """Return the shared video review agent."""
        from kairos.services.video_review import VideoReviewAgent

        return VideoReviewAgent()

    def get_audio_review_agent(self) -> BaseAudioReviewAgent:
        """Return the shared audio review agent."""
        from kairos.services.audio_review import AudioReviewAgent

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
