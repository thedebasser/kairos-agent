"""Unit tests for kairos.ai.prompts.lineage."""

from __future__ import annotations

from kairos.ai.prompts.lineage import compute_lineage, compute_template_hash


class TestTemplateHash:
    """Tests for compute_template_hash."""

    def test_deterministic(self):
        h1 = compute_template_hash("Hello {{ name }}")
        h2 = compute_template_hash("Hello {{ name }}")
        assert h1 == h2
        assert len(h1) == 12

    def test_different_content_different_hash(self):
        h1 = compute_template_hash("Hello {{ name }}")
        h2 = compute_template_hash("Goodbye {{ name }}")
        assert h1 != h2


class TestComputeLineage:
    """Tests for compute_lineage."""

    def test_basic_lineage(self):
        lineage = compute_lineage(
            template_name="concept_developer",
            raw_template="You are a {{ role }}",
            version=1,
            pipeline="domino",
            step="concept",
        )
        assert lineage["template_name"] == "concept_developer"
        assert lineage["template_version"] == 1
        assert lineage["pipeline"] == "domino"
        assert len(lineage["template_hash"]) == 12
        assert lineage["step"] == "concept"

    def test_lineage_without_variables(self):
        lineage = compute_lineage(
            template_name="test",
            raw_template="static prompt",
            version=2,
        )
        assert lineage["step"] == ""
        assert lineage["pipeline"] == ""
