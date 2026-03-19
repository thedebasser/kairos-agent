"""Template loader for the config-based simulation pipeline.

Loads fixed template scripts and injects LLM-generated JSON configs
into them, producing a complete runnable script for the sandbox.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Placeholder string in template files that gets replaced with actual config
_CONFIG_PLACEHOLDER = "__CONFIG_PLACEHOLDER__"

# Map category name → template filename
_TEMPLATE_FILES: dict[str, str] = {
    "ball_pit": "ball_pit.py",
    "domino_chain": "domino_chain.py",
    "destruction": "destruction.py",
    "marble_funnel": "marble_funnel.py",
}


@lru_cache(maxsize=8)
def _load_template_raw(category: str) -> str:
    """Load the raw template file content (cached)."""
    filename = _TEMPLATE_FILES.get(category)
    if filename is None:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid: {', '.join(_TEMPLATE_FILES)}"
        )
    path = _TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_simulation_script(category: str, config: dict) -> str:
    """Build a runnable simulation script from template + config.

    Loads the fixed template for *category*, replaces the config
    placeholder with the serialised JSON config dict, and returns
    the complete script ready for sandbox execution.

    Args:
        category: Scenario category (e.g. ``"domino_chain"``).
        config: The validated configuration dict from the LLM.

    Returns:
        Complete Python script as a string.

    Raises:
        ValueError: If *category* is not recognised.
        FileNotFoundError: If the template file is missing.
    """
    template = _load_template_raw(category)

    # Serialise config as a Python dict literal (compact JSON)
    config_json = json.dumps(config, indent=2)

    if _CONFIG_PLACEHOLDER not in template:
        raise RuntimeError(
            f"Template for '{category}' is missing the "
            f"{_CONFIG_PLACEHOLDER} marker"
        )

    script = template.replace(_CONFIG_PLACEHOLDER, config_json)

    logger.info(
        "Built simulation script for '%s' (%d chars, %d config keys)",
        category,
        len(script),
        len(config),
    )
    return script


def get_available_categories() -> list[str]:
    """Return the list of categories with available templates."""
    return list(_TEMPLATE_FILES.keys())
