"""Kairos Agent -- Domino Prompt Builder.

Loads domino prompt templates from ``.txt`` files via the shared
``PromptRegistry`` instead of inline strings.  Mirrors the physics
pipeline's builder pattern.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from kairos.ai.prompts.registry import PromptRegistry, RenderedPrompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _get_registry() -> PromptRegistry:
    return PromptRegistry(_PROMPTS_DIR)


def render_concept_system(*, rulebook: str = "") -> RenderedPrompt:
    """Render the domino concept developer system prompt.

    Appends the domino rulebook if available.
    """
    rp = _get_registry().render("system/concept_developer.txt")
    if rulebook:
        # Append rulebook as additional context
        return RenderedPrompt(
            text=rp.text + "\n" + rulebook,
            template_name=rp.template_name,
            version=rp.version,
            description=rp.description,
        )
    return rp


def render_concept_user(
    *,
    archetype: str,
    palette: str,
) -> RenderedPrompt:
    """Render the domino concept developer user prompt.

    Phase 4: ``schema`` param removed — Instructor injects it via response_model.
    """
    return _get_registry().render(
        "user/concept_developer.txt",
        variables={
            "archetype": archetype,
            "palette": palette,
        },
    )
