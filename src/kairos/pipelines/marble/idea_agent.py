"""Marble Idea Agent.

Implements IdeaAgent for the Blender marble course pipeline.

Uses LLM to select an archetype and generate a MarbleCourseConfig,
then converts it to a ConceptBrief for the shared graph.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any
from uuid import uuid4

from kairos.pipelines.contracts import IdeaAgent
from kairos.exceptions import ConceptGenerationError
from kairos.schemas.contracts import (
    AudioBrief,
    ConceptBrief,
    EnergyLevel,
    IdeaAgentInput,
    MarbleArchetype,
    ScenarioCategory,
    SimulationRequirements,
)
from kairos.pipelines.marble.models import MarbleCourseConfig
from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm

logger = logging.getLogger(__name__)

MAX_CONCEPT_ATTEMPTS = 3

MARBLE_ARCHETYPES = [a.value for a in MarbleArchetype]

# ── System prompt for marble concept generation ────────────────────────

_SYSTEM_PROMPT = """\
You name marble race videos for a short-form video channel.

Every video shows exactly 5 colourful marbles racing each other down a
procedural 3D marble course (ramps, turns, guard rails, finish bins).
The course is rendered in Blender and lasts about 65 seconds.

Given the palette, generate:
- A short, **plain-English** title a viewer would actually search for
  (e.g. "5 Marble Race — Rainbow Edition", "Red vs Blue Marble Showdown").
- A **1-sentence** visual brief describing the course look, NOT the
  physics, NOT metaphors, NOT sci-fi prose.  Keep it concrete.
- A punchy hook_text (≤ 6 words) for the opening caption.

Do NOT change marble_count, camera_style, archetype, or duration_sec —
those are locked.

Output ONLY valid JSON matching the provided schema.
"""

_USER_PROMPT_TEMPLATE = """\
Generate a marble race concept.

Locked values (do not change):
  archetype: race_lane
  marble_count: 5
  camera_style: marble_follow
  duration_sec: 65

Palette for this video: {palette}

Output ONLY valid JSON matching this schema:
{schema}
"""


class MarbleIdeaAgent(IdeaAgent):
    """Idea Agent for the Blender marble course pipeline.

    Selects an archetype (programmatic rotation or forced), then uses
    Claude Sonnet to generate a creative MarbleCourseConfig.
    """

    def __init__(
        self,
        *,
        force_archetype: str | None = None,
    ) -> None:
        self._force_archetype = force_archetype

    async def generate_concept(self, input: IdeaAgentInput) -> ConceptBrief:
        """Generate a production-ready marble concept.

        1. Select archetype (forced or rotated)
        2. Call LLM to generate MarbleCourseConfig
        3. Convert to ConceptBrief for the shared pipeline graph
        """
        last_error: Exception | None = None

        for attempt in range(1, MAX_CONCEPT_ATTEMPTS + 1):
            try:
                # Step 1: Select archetype
                archetype = self._select_archetype()
                logger.info(
                    "Archetype selected: %s (attempt %d/%d)",
                    archetype,
                    attempt,
                    MAX_CONCEPT_ATTEMPTS,
                )

                # Step 2: Generate config via LLM
                config = await self._generate_config(archetype)
                logger.info(
                    "Marble config generated: '%s' (%d marbles, %s palette)",
                    config.title,
                    config.marble_count,
                    config.palette,
                )

                # Step 3: Convert to ConceptBrief
                concept = self._config_to_concept_brief(config, input.pipeline)
                return concept

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Concept generation attempt %d/%d failed: %s",
                    attempt,
                    MAX_CONCEPT_ATTEMPTS,
                    exc,
                )

        msg = f"Failed to generate marble concept after {MAX_CONCEPT_ATTEMPTS} attempts"
        raise ConceptGenerationError(msg) from last_error

    async def get_category_stats(self, pipeline: str) -> dict[str, int]:
        """Get archetype distribution (placeholder — returns empty)."""
        return {a: 0 for a in MARBLE_ARCHETYPES}

    def _select_archetype(self) -> str:
        """Select archetype — locked to race_lane for now."""
        return "race_lane"

    async def _generate_config(self, archetype: str) -> MarbleCourseConfig:
        """Call LLM to generate a MarbleCourseConfig."""
        schema = json.dumps(
            MarbleCourseConfig.model_json_schema(),
            indent=2,
        )

        # Rotate palette each run for variety
        palette = random.choice(["rainbow", "neon", "pastel", "ocean", "earth"])  # noqa: S311

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    palette=palette,
                    schema=schema,
                ),
            },
        ]

        try:
            model = get_step_config("concept_developer").resolve_model()
        except Exception:
            # Fallback: use cloud model directly
            model = "anthropic/claude-sonnet-4-20250514"
            logger.warning("Could not resolve concept_developer model, using fallback: %s", model)

        result = await call_llm(
            model=model,
            messages=messages,
            response_model=MarbleCourseConfig,
            cache_step="marble_concept",
        )

        # Force locked values regardless of what the LLM returned
        overrides: dict[str, Any] = {
            "archetype": MarbleArchetype.RACE_LANE,
            "marble_count": 5,
            "camera_style": "marble_follow",
            "duration_sec": 65,
        }
        if result.seed == 0:
            overrides["seed"] = random.randint(1, 2**31)  # noqa: S311
        result = result.model_copy(update=overrides)

        return result

    @staticmethod
    def _config_to_concept_brief(
        config: MarbleCourseConfig,
        pipeline: str,
    ) -> ConceptBrief:
        """Convert MarbleCourseConfig to the shared ConceptBrief model.

        The graph expects a ConceptBrief with a ScenarioCategory.
        We map marble archetypes to BALL_PIT as a compatible category,
        and store the real archetype in the visual_brief.
        """
        # We embed the full marble config as JSON in simulation_requirements
        # special_effects so the simulation agent can retrieve it.
        marble_config_json = config.model_dump_json()

        return ConceptBrief(
            pipeline=pipeline,
            # Use BALL_PIT as a compatible category — the simulation agent
            # will detect pipeline="marble" and use the embedded config.
            category=ScenarioCategory.BALL_PIT,
            title=config.title,
            visual_brief=config.visual_brief,
            simulation_requirements=SimulationRequirements(
                body_count_initial=config.marble_count,
                body_count_max=config.marble_count,
                interaction_type="marble_course",
                colour_palette=_palette_to_hex(config.palette),
                background_colour="#1a1a2e",
                special_effects=[f"marble_config:{marble_config_json}"],
            ),
            audio_brief=config.audio_brief,
            hook_text=config.hook_text or "Watch till the end!",
            novelty_score=config.novelty_score,
            feasibility_score=config.feasibility_score,
            target_duration_sec=max(62, min(68, config.duration_sec)),
            seed=config.seed,
        )


def _palette_to_hex(palette_name: str) -> list[str]:
    """Convert palette name to hex colours."""
    palettes = {
        "rainbow": ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#8B00FF"],
        "neon": ["#FF00FF", "#00FFFF", "#FFFF00", "#FF0066", "#00FF66"],
        "pastel": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E8BAFF"],
        "monochrome": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333"],
        "earth": ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3", "#556B2F"],
        "ocean": ["#006994", "#40E0D0", "#00CED1", "#20B2AA", "#48D1CC"],
    }
    return palettes.get(palette_name, palettes["rainbow"])


def extract_marble_config(concept: ConceptBrief) -> MarbleCourseConfig | None:
    """Extract the embedded MarbleCourseConfig from a ConceptBrief.

    The idea agent stores the full marble config as JSON in the
    special_effects list of simulation_requirements.
    """
    for effect in concept.simulation_requirements.special_effects:
        if effect.startswith("marble_config:"):
            json_str = effect[len("marble_config:"):]
            try:
                return MarbleCourseConfig.model_validate_json(json_str)
            except Exception:
                logger.warning("Failed to parse embedded marble config")
                return None
    return None
