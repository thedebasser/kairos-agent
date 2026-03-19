"""Physics Pipeline Adapter.

Implements PipelineAdapter for "Oddly Satisfying Physics" simulations.
Uses Pygame 2.6 + Pymunk 6.8 as the simulation engine.
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
from kairos.schemas.contracts import ScenarioCategory
from kairos.orchestrator.registry import register_pipeline


@register_pipeline("physics")
class PhysicsPipelineAdapter(PipelineAdapter):
    """Pipeline adapter for physics simulations.

    Pipeline 1 from the implementation document — "Oddly Satisfying Physics".
    Engine: Pygame 2.6 + Pymunk 6.8
    Categories: ball_pit, marble_funnel, domino_chain, destruction
    """

    @property
    def pipeline_name(self) -> str:
        return "physics"

    @property
    def engine_name(self) -> str:
        return "pygame_pymunk"

    @property
    def categories(self) -> list[str]:
        return [c.value for c in ScenarioCategory]

    def get_idea_agent(self) -> IdeaAgent:
        """Return the physics idea agent."""
        from kairos.pipelines.physics.idea_agent import PhysicsIdeaAgent

        return PhysicsIdeaAgent()

    def get_simulation_agent(self) -> SimulationAgent:
        """Return the physics simulation agent."""
        from kairos.pipelines.physics.simulation_agent import PhysicsSimulationAgent

        return PhysicsSimulationAgent()

    def get_video_editor_agent(self) -> VideoEditorAgent:
        """Return the physics video editor agent."""
        from kairos.pipelines.physics.video_editor_agent import PhysicsVideoEditorAgent

        return PhysicsVideoEditorAgent()

    def get_video_review_agent(self) -> VideoReviewAgent:
        """Return the shared video review agent."""
        from kairos.ai.review.video_review_agent import VideoReviewAgent

        return VideoReviewAgent()

    def get_audio_review_agent(self) -> AudioReviewAgent:
        """Return the shared audio review agent."""
        from kairos.ai.review.audio_review_agent import AudioReviewAgent

        return AudioReviewAgent()

    def get_sandbox_dockerfile(self) -> str:
        """Return path to the sandbox Dockerfile for physics simulations."""
        return str(Path("sandbox/Dockerfile"))

    def get_prompt_template(self, category: str) -> str:
        """Load a prompt template for a given scenario category.

        TODO: Implement prompt template loading in Step 2.
        """
        templates_dir = Path(__file__).parent / "prompts"
        template_path = templates_dir / f"{category}.txt"
        if template_path.exists():
            return template_path.read_text()
        raise FileNotFoundError(f"Prompt template not found: {category}")

    async def health_check(self) -> bool:
        """Check if all physics pipeline dependencies are available."""
        dockerfile = Path(self.get_sandbox_dockerfile())
        return dockerfile.exists()
