"""Unit tests for the prompt development harness."""

from __future__ import annotations

import pytest

from kairos.models.contracts import ScenarioCategory
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
            assert concept.pipeline == "physics"
            assert len(concept.hook_text.split()) <= 6
            assert concept.target_duration_sec >= 62
            assert concept.target_duration_sec <= 68

    def test_sample_hook_text_word_count(self) -> None:
        """Hook text must be ≤6 words."""
        for name, concept in SAMPLE_CONCEPTS.items():
            word_count = len(concept.hook_text.split())
            assert word_count <= 6, f"{name} hook_text has {word_count} words"


class TestPromptTemplates:
    """Tests that prompt templates exist for all categories."""

    def test_all_categories_have_templates(self) -> None:
        """Each category should have a .txt prompt template."""
        from pathlib import Path

        prompts_dir = Path(__file__).resolve().parents[3] / "src" / "kairos" / "pipelines" / "physics" / "prompts"
        for cat in ScenarioCategory:
            template = prompts_dir / f"{cat.value}.txt"
            assert template.exists(), f"Missing prompt template: {template}"

    def test_templates_contain_placeholders(self) -> None:
        """Templates should contain variable placeholders."""
        from pathlib import Path

        prompts_dir = Path(__file__).resolve().parents[3] / "src" / "kairos" / "pipelines" / "physics" / "prompts"
        for cat in ScenarioCategory:
            template = (prompts_dir / f"{cat.value}.txt").read_text()
            assert "{{ title }}" in template, f"{cat.value} missing title placeholder"
            assert "{{ target_duration_sec }}" in template, f"{cat.value} missing duration placeholder"
            assert "{{ body_count_initial }}" in template, f"{cat.value} missing body_count placeholder"
            assert "simulation.mp4" in template, f"{cat.value} missing output path"
