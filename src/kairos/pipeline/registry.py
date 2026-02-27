"""Kairos Agent — Pipeline Registry.

Central registry for all pipeline adapters. Adding a new pipeline = registering
a new adapter. The registry enables automatic discovery by tests (e.g.,
test_pipeline_interface.py auto-discovers and tests all registered pipelines).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kairos.agents.base import BasePipelineAdapter

logger = logging.getLogger(__name__)

# Global registry: pipeline_name → adapter class
_REGISTRY: dict[str, type[BasePipelineAdapter]] = {}
_discovered = False


def register_pipeline(pipeline_name: str) -> type:
    """Decorator to register a pipeline adapter class.

    Usage:
        @register_pipeline("physics")
        class PhysicsPipelineAdapter(BasePipelineAdapter):
            ...
    """

    def decorator(cls: type[BasePipelineAdapter]) -> type[BasePipelineAdapter]:
        if pipeline_name in _REGISTRY:
            msg = (
                f"Pipeline '{pipeline_name}' is already registered "
                f"by {_REGISTRY[pipeline_name].__name__}"
            )
            raise ValueError(msg)
        _REGISTRY[pipeline_name] = cls
        return cls

    return decorator


def _discover_pipelines() -> None:
    """Auto-import all sub-packages of ``kairos.pipelines`` so that
    ``@register_pipeline`` decorators execute and populate the registry."""
    global _discovered  # noqa: PLW0603
    if _discovered:
        return
    _discovered = True
    try:
        import kairos.pipelines as _pkg

        for importer, modname, ispkg in pkgutil.walk_packages(
            _pkg.__path__, prefix=_pkg.__name__ + "."
        ):
            try:
                importlib.import_module(modname)
            except Exception:
                logger.debug("Skipping pipeline module %s", modname, exc_info=True)
    except Exception:
        logger.debug("Pipeline auto-discovery failed", exc_info=True)


def get_pipeline(pipeline_name: str) -> BasePipelineAdapter:
    """Instantiate and return a pipeline adapter by name.

    Raises:
        KeyError: If no pipeline adapter is registered under this name.
    """
    _discover_pipelines()
    if pipeline_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        msg = f"Pipeline '{pipeline_name}' not found. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[pipeline_name]()


def list_pipelines() -> list[str]:
    """Return sorted list of all registered pipeline names."""
    _discover_pipelines()
    return sorted(_REGISTRY.keys())


def get_registry() -> dict[str, type[BasePipelineAdapter]]:
    """Return the full pipeline registry (for testing/inspection)."""
    _discover_pipelines()
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Clear the pipeline registry (for testing only)."""
    _REGISTRY.clear()
