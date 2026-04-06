"""Set Designer Agent — creative scene composition.

Receives the asset catalogue, environment presets, and theme guidance.
Produces a SceneManifest describing the scene layout, object placements,
and visual narrative that the Path Setter will route through.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm
from kairos.pipelines.domino.creative.models import (
    AgentRole,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    IterationHistory,
    LightingConfig,
    ObjectRole,
    PlacedObject,
    SceneManifest,
)
from kairos.skills.catalogue import AssetEntry

logger = logging.getLogger(__name__)

# ─── Themes ──────────────────────────────────────────────────────────

SCENE_THEMES = [
    "modern_kitchen",
    "cozy_living_room",
    "industrial_warehouse",
    "outdoor_garden",
    "rooftop_terrace",
    "library_study",
    "childrens_playroom",
    "japanese_zen",
    "retro_arcade",
    "art_gallery",
]


def _format_catalogue_for_prompt(assets: list[AssetEntry], theme: str) -> str:
    """Format the asset catalogue subset as a concise string for the LLM."""
    lines = []
    for a in assets:
        surfaces = ""
        if a.surfaces:
            surface_strs = [
                f"{s.name} (h={s.local_height:.2f}m, {s.surface_type})"
                for s in a.surfaces
            ]
            surfaces = f"  surfaces: {', '.join(surface_strs)}"
        dims = f"{a.dimensions[0]:.1f}×{a.dimensions[1]:.1f}×{a.dimensions[2]:.1f}m"
        lines.append(
            f"- id={a.id}  name={a.name}  category={a.category}  "
            f"dims={dims}  themes={a.themes}{surfaces}"
        )
    return "\n".join(lines) if lines else "(no assets available)"


def _build_system_prompt() -> str:
    return (
        "You are the Set Designer for a domino run video. Your job is to compose "
        "an interesting 3D scene where dominoes will topple through furniture and "
        "objects at different heights.\n\n"
        "RULES:\n"
        "1. Place 3-8 objects from the catalogue. At least 2 must be FUNCTIONAL "
        "(role=functional) with navigable surfaces at different elevations.\n"
        "2. Decorative objects add visual interest but dominoes don't interact.\n"
        "3. Objects must not overlap — leave clearance for the domino path between them.\n"
        "4. Functional objects need their surface_name set to a valid surface from the catalogue.\n"
        "5. Keep all positions within a 10×10m area centered at origin.\n"
        "6. Write a short narrative (1-2 sentences) describing the visual story.\n"
        "7. Choose environment settings (indoor/outdoor, ground texture, lighting) "
        "that match the theme.\n\n"
        "Respond ONLY with valid JSON matching the SceneManifest schema."
    )


def _build_user_prompt(
    theme: str,
    catalogue_text: str,
    domino_count: int,
    feedback: str,
) -> str:
    lines = [
        f"Theme: {theme}",
        f"Target domino count: {domino_count}",
        "",
        "Available assets:",
        catalogue_text,
        "",
    ]
    if feedback and feedback != "This is your first attempt.":
        lines.extend(["Previous attempt feedback:", feedback, ""])
    lines.append(
        "Design a scene for this theme. Place functional objects at "
        "varied heights so the domino path has interesting elevation changes."
    )
    return "\n".join(lines)


class SetDesignerAgent:
    """Composes a scene by selecting and placing assets from the catalogue.

    The LLM chooses objects, positions, and roles to create an interesting
    environment for the domino path to travel through.
    """

    def __init__(
        self,
        assets: list[AssetEntry],
        *,
        force_theme: str | None = None,
    ) -> None:
        self._assets = assets
        self._force_theme = force_theme

    async def design_scene(
        self,
        *,
        domino_count: int = 300,
        history: IterationHistory | None = None,
    ) -> SceneManifest:
        """Generate a SceneManifest by calling the LLM.

        Args:
            domino_count: Target number of dominoes.
            history: Optional iteration history for feedback on retries.

        Returns:
            A frozen SceneManifest describing the full scene.
        """
        theme = self._force_theme or random.choice(SCENE_THEMES)  # noqa: S311

        # Filter assets by theme (fallback to all if too few match)
        themed = [a for a in self._assets if theme in a.themes]
        catalogue = themed if len(themed) >= 3 else self._assets

        catalogue_text = _format_catalogue_for_prompt(catalogue, theme)
        feedback = (
            history.format_feedback(AgentRole.SET_DESIGNER)
            if history
            else "This is your first attempt."
        )

        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {
                "role": "user",
                "content": _build_user_prompt(
                    theme, catalogue_text, domino_count, feedback,
                ),
            },
        ]

        step_cfg = _get_step_config_safe("concept_developer")
        model = _resolve_model(step_cfg)

        manifest = await call_llm(
            model=model,
            messages=messages,
            response_model=SceneManifest,
            cache_step="set_designer",
        )

        logger.info(
            "[set_designer] Scene designed: theme=%s, %d objects, narrative='%s'",
            manifest.theme,
            len(manifest.objects),
            manifest.narrative[:80],
        )
        return manifest


# ─── Helpers ─────────────────────────────────────────────────────────


def _get_step_config_safe(step_name: str):
    """Get step config without raising on missing config."""
    try:
        return get_step_config(step_name)
    except Exception:
        return None


def _resolve_model(step_cfg) -> str:
    """Resolve the LLM model to use, with safe fallback."""
    if step_cfg:
        try:
            return step_cfg.resolve_model()
        except Exception:
            pass
    return "concept-developer"
