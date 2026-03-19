"""Domino Idea Agent.

Implements IdeaAgent for the Blender domino run pipeline.

Uses LLM to select an archetype and generate a DominoCourseConfig,
then converts it to a ConceptBrief for the shared graph.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from kairos.pipelines.contracts import IdeaAgent
from kairos.exceptions import ConceptGenerationError
from kairos.schemas.contracts import (
    AudioBrief,
    ConceptBrief,
    DominoArchetype,
    EnergyLevel,
    IdeaAgentInput,
    ScenarioCategory,
    SimulationRequirements,
)
from kairos.pipelines.domino.models import DominoCourseConfig
from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm, call_with_quality_fallback

logger = logging.getLogger(__name__)

MAX_CONCEPT_ATTEMPTS = 3

DOMINO_ARCHETYPES = [a.value for a in DominoArchetype]

_RULEBOOK_PATH = Path(__file__).resolve().parents[4] / "knowledge" / "domino_rulebook.md"


def _load_rulebook() -> str:
    """Load the domino rulebook from knowledge/domino_rulebook.md.

    Returns the rulebook text, or empty string if not found.
    """
    try:
        if _RULEBOOK_PATH.exists():
            text = _RULEBOOK_PATH.read_text(encoding="utf-8").strip()
            if text:
                return f"\n\n## RULEBOOK — IMPORTANT CONSTRAINTS\n{text}\n"
    except Exception as exc:
        logger.warning("Failed to load domino rulebook: %s", exc)
    return ""


# ── System prompt for domino concept generation ────────────────────────

_SYSTEM_PROMPT = """\
You design domino run videos for a short-form video channel.

Every video shows colourful dominoes falling in a satisfying cascade
on a 3D ground plane, rendered in Blender. Videos last about 65 seconds.

Available archetypes:
- spiral: dominoes spiral outward from center
- s_curve: dominoes follow a smooth S-curve path
- branching: trunk path that fans into multiple branches
- cascade: wide zigzag rows filling the frame
- word_spell: dominoes arranged along a shape/arc

Given the archetype, generate:
- A short, **plain-English** title a viewer would actually search for
  (e.g. "500 Dominoes Spiral — So Satisfying", "Domino Chain Reaction").
- A **1-sentence** visual brief describing the look. Be concrete.
- A punchy hook_text (≤ 6 words) for the opening caption.
- Choose a colour palette and domino count that fit the archetype.
- Optionally choose a finale_type: none, tower, ball, or ramp.

Do NOT change physics parameters (mass, friction, spacing) — those are locked.

Output ONLY valid JSON matching the provided schema.
"""

_USER_PROMPT_TEMPLATE = """\
Generate a domino run concept.

Locked values (do not change):
  domino_count: 300
  domino_width: 0.08
  domino_height: 0.4
  domino_depth: 0.06
  spacing_ratio: 0.35
  path_amplitude: 1.0
  path_cycles: 2.0
  domino_mass: 0.3
  domino_friction: 0.6
  domino_bounce: 0.1
  trigger_impulse: 1.5
  trigger_tilt_degrees: 8.0
  duration_sec: 65

Archetype for this video: {archetype}
Palette for this video: {palette}

Output ONLY valid JSON matching this schema:
{schema}
"""


class DominoIdeaAgent(IdeaAgent):
    """Idea Agent for the Blender domino run pipeline.

    Selects an archetype (programmatic rotation), then uses
    Claude Sonnet to generate a creative DominoCourseConfig.
    """

    def __init__(
        self,
        *,
        force_archetype: str | None = None,
    ) -> None:
        self._force_archetype = force_archetype

    async def generate_concept(self, input: IdeaAgentInput) -> ConceptBrief:
        """Generate a production-ready domino concept.

        1. Select archetype (forced or rotated)
        2. Call LLM to generate DominoCourseConfig
        3. Convert to ConceptBrief for the shared pipeline graph
        """
        # ── Cache check ──────────────────────────────────────────────
        from kairos.ai.llm.cache import get_cache
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_idea")
            if cached:
                logger.info("[domino_idea] Cache HIT — skipping LLM call")
                return ConceptBrief.model_validate(cached["concept"])

        last_error: Exception | None = None

        for attempt in range(1, MAX_CONCEPT_ATTEMPTS + 1):
            try:
                archetype = self._select_archetype()
                logger.info(
                    "Archetype selected: %s (attempt %d/%d)",
                    archetype, attempt, MAX_CONCEPT_ATTEMPTS,
                )

                config = await self._generate_config(archetype)
                logger.info(
                    "Domino config generated: '%s' (%d dominos, %s palette)",
                    config.title, config.domino_count, config.palette,
                )

                concept = self._config_to_concept_brief(config, input.pipeline)

                # ── Cache store ──────────────────────────────────────
                if cache:
                    cache.put_step("domino_idea", {
                        "concept": concept.model_dump(mode="json"),
                    })

                return concept

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Concept generation attempt %d/%d failed: %s",
                    attempt, MAX_CONCEPT_ATTEMPTS, exc,
                )

        msg = f"Failed to generate domino concept after {MAX_CONCEPT_ATTEMPTS} attempts"
        raise ConceptGenerationError(msg) from last_error

    async def get_category_stats(self, pipeline: str) -> dict[str, int]:
        """Get archetype distribution (placeholder — returns empty)."""
        return {a: 0 for a in DOMINO_ARCHETYPES}

    def _select_archetype(self) -> str:
        """Select archetype — forced or random rotation."""
        if self._force_archetype:
            return self._force_archetype
        # Weighted selection: s_curve and spiral are most visually reliable
        weights = {
            "spiral": 3,
            "s_curve": 3,
            "branching": 2,
            "cascade": 2,
            "word_spell": 1,
        }
        choices = list(weights.keys())
        w = [weights[c] for c in choices]
        return random.choices(choices, weights=w, k=1)[0]  # noqa: S311

    async def _generate_config(self, archetype: str) -> DominoCourseConfig:
        """Call LLM to generate a DominoCourseConfig."""
        schema = json.dumps(
            DominoCourseConfig.model_json_schema(),
            indent=2,
        )

        palette = random.choice(["rainbow", "neon", "pastel", "ocean", "sunset", "earth"])  # noqa: S311

        rulebook = _load_rulebook()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT + rulebook},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    archetype=archetype,
                    palette=palette,
                    schema=schema,
                ),
            },
        ]

        try:
            step_cfg = get_step_config("concept_developer")
        except Exception:
            step_cfg = None

        if step_cfg and step_cfg.call_pattern == "quality_fallback":
            # Use local-first with cloud fallback — saves cloud cost
            primary, fallback = step_cfg.resolve_primary_and_fallback()

            def _validate_concept(result: DominoCourseConfig) -> bool:
                """Quality gate: title present, visual_brief >50 chars, 5+ colours."""
                if not result.title or len(result.title) < 5:
                    return False
                if not result.visual_brief or len(result.visual_brief) < 50:
                    return False
                if not result.palette:
                    return False
                return True

            result = await call_with_quality_fallback(
                primary_model=primary,
                fallback_model=fallback,
                messages=messages,
                validator=_validate_concept,
                response_model=DominoCourseConfig,
            )
        else:
            # Direct call (single model)
            try:
                model = step_cfg.resolve_model() if step_cfg else "concept-developer"
            except Exception:
                model = "concept-developer"
                logger.warning("Could not resolve concept_developer model, using fallback: %s", model)

            result = await call_llm(
                model=model,
                messages=messages,
                response_model=DominoCourseConfig,
                cache_step="domino_concept",
            )

        # Force locked physics values regardless of what the LLM returned
        overrides: dict[str, Any] = {
            "archetype": DominoArchetype(archetype),
            "domino_width": 0.08,
            "domino_height": 0.4,
            "domino_depth": 0.06,
            "spacing_ratio": 0.35,
            "domino_mass": 0.3,
            "domino_friction": 0.6,
            "domino_bounce": 0.1,
            "ground_friction": 0.8,
            "trigger_tilt_degrees": 8.0,
            "trigger_impulse": 1.5,
            "domino_count": 300,
            "camera_style": "tracking",
            "path_amplitude": 1.0,   # curvature-safe for chain propagation
            "path_cycles": 2.0,     # more visual variety on longer paths
            "duration_sec": 65,
        }
        if result.seed == 0:
            overrides["seed"] = random.randint(1, 2**31)  # noqa: S311
        result = result.model_copy(update=overrides)

        return result

    @staticmethod
    def _config_to_concept_brief(
        config: DominoCourseConfig,
        pipeline: str,
    ) -> ConceptBrief:
        """Convert DominoCourseConfig to the shared ConceptBrief model."""
        domino_config_json = config.model_dump_json()

        return ConceptBrief(
            pipeline=pipeline,
            # Use BALL_PIT as compatible category — the simulation agent
            # detects pipeline="domino" and uses the embedded config.
            category=ScenarioCategory.BALL_PIT,
            title=config.title,
            visual_brief=config.visual_brief,
            simulation_requirements=SimulationRequirements(
                body_count_initial=config.domino_count,
                body_count_max=config.domino_count,
                interaction_type="domino_run",
                colour_palette=_palette_to_hex(config.palette),
                background_colour="#2a2a3e",
                special_effects=[f"domino_config:{domino_config_json}"],
            ),
            audio_brief=config.audio_brief,
            hook_text=config.hook_text or "Watch them all fall!",
            novelty_score=config.novelty_score,
            feasibility_score=config.feasibility_score,
            target_duration_sec=max(62, min(68, config.duration_sec)),
            seed=config.seed,
        )


def _palette_to_hex(palette_name: str) -> list[str]:
    """Convert palette name to hex colours."""
    palettes = {
        "rainbow": ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"],
        "neon": ["#FF1493", "#00FF7F", "#FFD700", "#00BFFF", "#FF4500", "#7FFF00"],
        "pastel": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E8BAFF"],
        "ocean": ["#006994", "#40A4DF", "#8ED1FC", "#B8E6FF", "#004E6A"],
        "sunset": ["#FF6B6B", "#FFA07A", "#FFD700", "#FF4500", "#DC143C"],
        "earth": ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3", "#556B2F"],
        "monochrome": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333"],
    }
    return palettes.get(palette_name, palettes["rainbow"])


def extract_domino_config(concept: ConceptBrief) -> DominoCourseConfig | None:
    """Extract the embedded DominoCourseConfig from a ConceptBrief.

    The idea agent stores the full domino config as JSON in the
    special_effects list of simulation_requirements.
    """
    for effect in concept.simulation_requirements.special_effects:
        if effect.startswith("domino_config:"):
            json_str = effect[len("domino_config:"):]
            try:
                return DominoCourseConfig.model_validate_json(json_str)
            except Exception:
                logger.warning("Failed to parse embedded domino config")
                return None
    return None
