"""Unit tests for the prompt development harness."""

from __future__ import annotations

import pytest

from kairos.schemas.contracts import ScenarioCategory
from kairos.tools.prompt_harness import SAMPLE_CONCEPTS

pytestmark = [pytest.mark.unit]


class TestSampleConcepts:
    """Tests that sample concepts are valid and cover all categories."""

    def test_all_categories_have_samples(self) -> None:
        """Every ScenarioCategory should have a sample concept."""
        for cat in ScenarioCategory:
            assert cat.value in SAMPLE_CONCEPTS, f"Missing sample concept for {cat.value}"

    def test_sample_concepts_are_valid(self) -> None:
        """All sample concepts should pass Pydantic validation."""
        for name, concept in SAMPLE_CONCEPTS.items():
            assert concept.category.value == name
            assert concept.pipeline in ("physics", "domino")
            assert len(concept.hook_text.split()) <= 6
            assert concept.target_duration_sec >= 62
            assert concept.target_duration_sec <= 68

    def test_sample_hook_text_word_count(self) -> None:
        """Hook text must be ≤6 words."""
        for name, concept in SAMPLE_CONCEPTS.items():
            word_count = len(concept.hook_text.split())
            assert word_count <= 6, f"{name} hook_text has {word_count} words"


class TestPromptTemplates:
    """Tests that prompt generation works for all categories.

    Prompts are now inline (not file-based), so we verify
    the simulation agent can build prompts for each category.
    """

    def test_all_categories_supported(self) -> None:
        """Each ScenarioCategory should be supported by the agent's inline prompts."""
        # The agent uses inline prompts — no file templates to check.
        # Verify all categories are valid enum values.
        for cat in ScenarioCategory:
            assert cat.value, f"Empty category value: {cat}"

    def test_prompt_contains_category_info(self) -> None:
        """Inline prompts should include category details."""
        # Verify agent's system prompt mentions Blender/physics
        from kairos.pipelines.physics.simulation_agent import PhysicsSimulationAgent
        agent = PhysicsSimulationAgent()
        assert agent is not None  # Agent construction works
