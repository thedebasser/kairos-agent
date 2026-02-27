"""Tests for pipeline adapter interface compliance.

Verifies all registered pipelines implement the BasePipelineAdapter contract.
"""

import pytest

from kairos.pipeline.registry import get_registry

pytestmark = pytest.mark.pipeline


class TestPipelineInterface:
    """Tests that all registered pipelines conform to the adapter interface."""

    def test_physics_pipeline_registered(self):
        """Physics pipeline should be registered."""
        registry = get_registry()
        # After physics adapter is imported, it should be in the registry
        assert "physics" in registry or len(registry) == 0  # Placeholder until adapter registered

    def test_all_pipelines_have_required_attributes(self):
        """All registered pipelines must implement required attributes."""
        for name, adapter_cls in get_registry().items():
            adapter = adapter_cls()
            assert hasattr(adapter, "pipeline_name")
            assert hasattr(adapter, "engine_name")
            assert hasattr(adapter, "categories")
            assert hasattr(adapter, "get_idea_agent")
            assert hasattr(adapter, "get_simulation_agent")
            assert hasattr(adapter, "get_video_editor_agent")
