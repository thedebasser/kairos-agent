"""Path Setter Agent — routes the domino path through the scene.

Receives the SceneManifest from Set Designer. Analyses surfaces,
plans a route through functional objects at different elevations,
and produces a PathOutput with segments and transition flags.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm
from kairos.pipelines.domino.creative.models import (
    AgentRole,
    ConnectorType,
    IterationHistory,
    PathOutput,
    PathSegment,
    SceneManifest,
    SegmentType,
    Waypoint,
)

logger = logging.getLogger(__name__)


def _format_scene_for_prompt(manifest: SceneManifest) -> str:
    """Format scene manifest as concise text for the LLM."""
    lines = [
        f"Theme: {manifest.theme}",
        f"Narrative: {manifest.narrative}",
        f"Target domino count: {manifest.domino_count}",
        f"Environment: {manifest.environment.type.value}",
        "",
        "Placed objects:",
    ]
    for obj in manifest.objects:
        pos = f"({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f})"
        surface_info = ""
        if obj.surface_name:
            surface_info = f"  surface={obj.surface_name}"
        lines.append(
            f"  - {obj.name} (id={obj.asset_id}, role={obj.role.value}, "
            f"pos={pos}, scale={obj.scale:.1f}){surface_info}"
        )
    return "\n".join(lines)


def _build_system_prompt() -> str:
    return (
        "You are the Path Setter for a domino run video. Your job is to route "
        "a domino path through the scene, travelling across functional objects "
        "at different heights to create an exciting visual journey.\n\n"
        "RULES:\n"
        "1. Start from ground level and route through functional objects' surfaces.\n"
        "2. Create segments: FLAT_SURFACE (dominoes on a surface), "
        "HEIGHT_TRANSITION_UP/DOWN (needs a connector), GROUND_LEVEL (floor).\n"
        "3. Flag height transitions with needs_connector=true and suggest a "
        "connector type: ramp, spiral_ramp, staircase, platform, plank_bridge.\n"
        "4. connector_hint should match the height delta: ramp (<0.5m), "
        "staircase (0.3-1.5m), spiral_ramp (>1m), platform (same height), "
        "plank_bridge (horizontal gap).\n"
        "5. Provide waypoints for flat/ground segments (the path shape).\n"
        "6. Estimate the total path length.\n"
        "7. available_footprint is the (width, depth) in metres available for "
        "each connector based on surrounding object clearance.\n"
        "8. gradient should stay under 15° for reliable domino toppling.\n\n"
        "Respond ONLY with valid JSON matching the PathOutput schema."
    )


def _build_user_prompt(
    scene_text: str,
    feedback: str,
) -> str:
    lines = [scene_text, ""]
    if feedback and feedback != "This is your first attempt.":
        lines.extend(["Previous attempt feedback:", feedback, ""])
    lines.append(
        "Route a domino path through this scene. The path should travel "
        "across at least 2 functional surfaces at different heights, with "
        "appropriate connectors flagged for each height transition."
    )
    return "\n".join(lines)


class PathSetterAgent:
    """Plans a domino path route through the scene's functional objects.

    Analyses the scene manifest, identifies navigable surfaces, and
    produces a segmented path with height transition flags.
    """

    async def plan_path(
        self,
        manifest: SceneManifest,
        *,
        history: IterationHistory | None = None,
    ) -> PathOutput:
        """Generate a PathOutput from the scene manifest.

        Args:
            manifest: Scene layout from the Set Designer.
            history: Optional iteration history for feedback on retries.

        Returns:
            A frozen PathOutput with segmented path and transition flags.
        """
        scene_text = _format_scene_for_prompt(manifest)
        feedback = (
            history.format_feedback(AgentRole.PATH_SETTER)
            if history
            else "This is your first attempt."
        )

        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {
                "role": "user",
                "content": _build_user_prompt(scene_text, feedback),
            },
        ]

        step_cfg = _get_step_config_safe("concept_developer")
        model = _resolve_model(step_cfg)

        path_output = await call_llm(
            model=model,
            messages=messages,
            response_model=PathOutput,
            cache_step="path_setter",
        )

        # Inject domino_count from manifest if LLM didn't set it
        if path_output.domino_count == 0:
            path_output = PathOutput(
                total_length_estimate=path_output.total_length_estimate,
                segments=path_output.segments,
                domino_count=manifest.domino_count,
            )

        transitions = sum(1 for s in path_output.segments if s.needs_connector)
        logger.info(
            "[path_setter] Path planned: %.1fm total, %d segments, "
            "%d transitions flagged",
            path_output.total_length_estimate,
            len(path_output.segments),
            transitions,
        )
        return path_output


# ─── Helpers ─────────────────────────────────────────────────────────


def _get_step_config_safe(step_name: str):
    try:
        return get_step_config(step_name)
    except Exception:
        return None


def _resolve_model(step_cfg) -> str:
    if step_cfg:
        try:
            return step_cfg.resolve_model()
        except Exception:
            pass
    return "concept-developer"
