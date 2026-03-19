"""Prompt builder for the physics simulation pipeline.

Assembles prompts from shared fragments and category-specific content,
eliminating duplication across the 4 scenario types.

All public functions return ``RenderedPrompt`` objects that carry version
metadata from YAML front-matter headers (Finding 3.2).

Usage::

    from kairos.pipelines.physics.prompts.builder import (
        build_simulation_prompt,
        load_system_prompt,
        build_user_prompt,
    )

    # System/user prompts for other agents
    rp = load_system_prompt("concept_developer")
    print(rp.version, rp.text[:80])

    # Simulation code generation (full assembled prompt)
    rp = build_simulation_prompt("ball_pit", {"title": "...", ...})
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from kairos.services.prompt_registry import PromptRegistry, RenderedPrompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent

# Singleton registry scoped to the physics prompts directory.
_registry = PromptRegistry(_PROMPTS_DIR)

# Per-category config merged into template variables automatically.
CATEGORY_CONFIG: dict[str, dict[str, str]] = {
    "ball_pit": {"category_label": "ball pit", "gravity_y": "900"},
    "domino_chain": {"category_label": "domino chain", "gravity_y": "981"},
    "destruction": {"category_label": "destruction", "gravity_y": "900"},
    "marble_funnel": {"category_label": "marble funnel", "gravity_y": "900"},
}

# Ordered list of shared fragments that precede category-specific content.
_SHARED_SECTIONS: list[str] = [
    "_shared/role.txt",
    "_shared/concept_details.txt",
    "_shared/coordinates.txt",
    "_shared/technical_reqs.txt",
]


@lru_cache(maxsize=64)
def _load_fragment(relative_path: str) -> str:
    """Load a prompt fragment file, cached for performance."""
    path = _PROMPTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt fragment not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _render(template: str, variables: dict[str, str]) -> str:
    """Replace ``{{ key }}`` placeholders with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{ " + key + " }}", str(value))
        result = result.replace("{{" + key + "}}", str(value))
    return result


def build_simulation_prompt(
    category: str,
    variables: dict[str, str],
) -> RenderedPrompt:
    """Assemble the full simulation code-generation user prompt.

    Combines shared fragments (role, concept details, coordinates,
    technical requirements) with category-specific content (physics
    parameters, energy curve, scene construction, working example,
    colour psychology, quality checklist).

    The returned ``RenderedPrompt.version`` is the *maximum* version found
    across all constituent fragments, giving callers a single version
    number that bumps whenever any fragment is updated.

    Args:
        category: Scenario category key (e.g. ``"ball_pit"``).
        variables: Template variables to substitute (title, visual_brief, …).

    Returns:
        ``RenderedPrompt`` with the fully assembled and rendered text.

    Raises:
        ValueError: If *category* is not recognised.
    """
    config = CATEGORY_CONFIG.get(category)
    if config is None:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid: {', '.join(CATEGORY_CONFIG)}"
        )

    merged = {**config, **variables}

    # Render each fragment through the registry to capture version info.
    fragment_names = list(_SHARED_SECTIONS) + [f"categories/{category}.txt"]
    rendered_parts: list[RenderedPrompt] = [
        _registry.render(name, merged) for name in fragment_names
    ]

    text = "\n\n".join(rp.text for rp in rendered_parts)
    composite_version = max(rp.version for rp in rendered_parts)

    return RenderedPrompt(
        text=text,
        template_name=f"simulation/{category}",
        version=composite_version,
        description=f"Composite simulation prompt for {category}",
    )


def load_system_prompt(
    step: str,
    variables: dict[str, str] | None = None,
) -> RenderedPrompt:
    """Load a system prompt from the ``system/`` directory.

    Args:
        step: Step name (e.g. ``"simulation_codegen"``, ``"concept_developer"``).
        variables: Optional template variables to substitute.

    Returns:
        ``RenderedPrompt`` with the system prompt text and version metadata.
    """
    return _registry.render(f"system/{step}.txt", variables)


def build_user_prompt(
    step: str,
    variables: dict[str, str],
) -> RenderedPrompt:
    """Load and render a user prompt template from the ``user/`` directory.

    Args:
        step: Step name (e.g. ``"concept_developer"``, ``"caption_writer"``).
        variables: Template variables to substitute.

    Returns:
        ``RenderedPrompt`` with the user prompt text and version metadata.
    """
    return _registry.render(f"user/{step}.txt", variables)
