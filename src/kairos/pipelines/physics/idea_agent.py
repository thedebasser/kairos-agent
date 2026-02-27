"""Physics Idea Agent.

Implements BaseIdeaAgent for the "Oddly Satisfying Physics" pipeline.

Subagents:
- Inventory Analyst: pure SQL queries against the database
- Category Selector: uses rotation logic (programmatic + optional LLM)
- Concept Developer: Claude Sonnet via Instructor for creative concept generation

The Idea Agent is the first node in the pipeline graph. It receives
a mostly-empty PipelineState and returns an updated state with a
ConceptBrief attached.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from uuid import UUID

from kairos.agents.base import BaseIdeaAgent
from kairos.exceptions import ConceptGenerationError
from kairos.models.contracts import (
    ConceptBrief,
    PipelineState,
    ScenarioCategory,
)
from kairos.models.idea import (
    CategoryReport,
    CategorySelection,
    ConceptDeveloperResponse,
    InventoryReport,
)
from kairos.services.category_rotation import (
    BOOST_THRESHOLD,
    CategoryInfo,
    compute_streak,
    select_category,
)
from kairos.services.llm_config import get_step_config
from kairos.services.llm_routing import call_llm

logger = logging.getLogger(__name__)

# Maximum concept generation attempts before giving up
MAX_CONCEPT_ATTEMPTS = 3

# All valid categories for the physics pipeline
PHYSICS_CATEGORIES = [c.value for c in ScenarioCategory]


# =============================================================================
# Inventory Analyst (programmatic — no LLM)
# =============================================================================


async def build_inventory_report(
    pipeline: str,
    *,
    category_stats: list[dict[str, object]] | None = None,
    recent_categories: list[str] | None = None,
    last_category: str | None = None,
) -> InventoryReport:
    """Build an inventory report from database stats.

    In production, this queries the database via db.operations.
    For testing, stats can be provided directly.

    Args:
        pipeline: Pipeline identifier (e.g. "physics").
        category_stats: List of category stat dicts (from DB or test fixture).
        recent_categories: Recent category names (newest first).
        last_category: The most recently used category.

    Returns:
        InventoryReport with per-category breakdown.
    """
    stats = category_stats or []
    recent = recent_categories or []

    total_videos = sum(int(s.get("total_count", 0)) for s in stats)
    total_last_30 = sum(int(s.get("videos_last_30_days", 0)) for s in stats)

    categories: list[CategoryReport] = []
    for stat in stats:
        cat_name = str(stat.get("category", ""))
        count_30 = int(stat.get("videos_last_30_days", 0))
        total_count = int(stat.get("total_count", 0))
        streak = compute_streak(recent, cat_name)
        pct = (count_30 / total_last_30 * 100) if total_last_30 > 0 else 0.0

        categories.append(
            CategoryReport(
                category=cat_name,
                total_count=total_count,
                videos_last_30_days=count_30,
                streak_count=streak,
                percentage_last_30_days=round(pct, 1),
                is_last_used=(cat_name == last_category),
                needs_boost=(total_count < BOOST_THRESHOLD),
            )
        )

    return InventoryReport(
        pipeline=pipeline,
        total_videos=total_videos,
        categories=categories,
        last_category=last_category,
        recent_categories=recent,
    )


# =============================================================================
# Category Selector
# =============================================================================


def select_category_from_report(
    report: InventoryReport,
    *,
    seed: int | None = None,
) -> str:
    """Select the next category using rotation rules from the inventory report.

    Converts InventoryReport into CategoryInfo objects and delegates to
    the pure-function category rotation logic.

    Args:
        report: Inventory report with all category stats.
        seed: Optional seed for deterministic selection.

    Returns:
        Selected category name.
    """
    # Convert report categories to CategoryInfo for the rotation engine
    category_infos = [
        CategoryInfo(
            name=cat.category,
            total_count=cat.total_count,
            videos_last_30_days=cat.videos_last_30_days,
            streak_count=cat.streak_count,
            is_last_used=cat.is_last_used,
        )
        for cat in report.categories
    ]

    # If no categories in report (empty DB), use all physics categories
    if not category_infos:
        category_infos = [CategoryInfo(name=c) for c in PHYSICS_CATEGORIES]

    return select_category(category_infos, seed=seed)


async def select_category_with_llm(
    report: InventoryReport,
    *,
    seed: int | None = None,
) -> CategorySelection:
    """Select category using the LLM-backed Category Selector.

    Uses Mistral 7B (local) for a structured, rule-based decision.
    Falls back to pure rotation logic if the LLM call fails.

    Args:
        report: Inventory report with all category stats.
        seed: Optional seed for deterministic fallback selection.

    Returns:
        CategorySelection with selected category and reasoning.
    """
    # Build the prompt for the Category Selector LLM
    categories_text = "\n".join(
        f"- {cat.category}: {cat.total_count} total, "
        f"{cat.videos_last_30_days} last 30 days ({cat.percentage_last_30_days}%), "
        f"streak={cat.streak_count}, last_used={cat.is_last_used}, "
        f"needs_boost={cat.needs_boost}"
        for cat in report.categories
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a content strategist for a physics simulation video channel. "
                "Select the best scenario category for the next video based on rotation "
                "rules and category statistics.\n\n"
                "Rules:\n"
                "1. HARD BLOCK: Never select the same category as the previous video\n"
                "2. STREAK BREAK: Never select a category used 3+ times consecutively\n"
                "3. SOFT BLOCK: Deprioritise categories >30% of last 30 days output\n"
                "4. BOOST: Prefer categories with <5 total videos\n\n"
                f"Available categories: {', '.join(PHYSICS_CATEGORIES)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Category Statistics:\n{categories_text}\n\n"
                f"Last category used: {report.last_category or 'None (first video)'}\n"
                f"Recent sequence: {report.recent_categories[:5] or ['None']}\n\n"
                "Select the best category for the next video."
            ),
        },
    ]

    try:
        model = get_step_config("category_selector").resolve_model()
        return await call_llm(
            model=model,
            messages=messages,
            response_model=CategorySelection,
        )
    except Exception:
        logger.warning(
            "Category Selector LLM failed, falling back to programmatic selection",
            exc_info=True,
        )
        selected = select_category_from_report(report, seed=seed)
        return CategorySelection(
            selected_category=ScenarioCategory(selected),
            reasoning="Fallback to programmatic rotation logic (LLM unavailable)",
        )


# =============================================================================
# Concept Developer
# =============================================================================


def _build_concept_developer_prompt(
    category: ScenarioCategory,
    report: InventoryReport,
) -> list[dict[str, str]]:
    """Build the prompt for the Concept Developer LLM.

    Args:
        category: The selected scenario category.
        report: Inventory report for context on existing content.

    Returns:
        Chat messages for the LLM call.
    """
    category_descriptions = {
        ScenarioCategory.BALL_PIT: (
            "Ball pit / collision cascade: Colourful balls interacting in a confined space. "
            "Collisions, spawning, chain reactions, gravity effects. "
            "Key appeal: satisfying physics, cascading effects, screen filling with colour."
        ),
        ScenarioCategory.MARBLE_FUNNEL: (
            "Marble funnel / sorting: Marbles rolling through funnels, ramps, and sorting "
            "mechanisms. Gravity-driven flow, splitting, merging, colour sorting. "
            "Key appeal: mesmerising flow, predictable yet satisfying paths."
        ),
        ScenarioCategory.DOMINO_CHAIN: (
            "Domino chains: Sequences of dominoes toppling in patterns — spirals, mazes, "
            "splits, reunions. Chain reactions with perfect timing. "
            "Key appeal: anticipation, perfect execution, satisfying completion."
        ),
        ScenarioCategory.DESTRUCTION: (
            "Destruction / stacking: Building tall structures then destroying them, "
            "or stacking objects to impossible heights before collapse. "
            "Key appeal: tension of will-it-hold, dramatic collapse, debris physics."
        ),
    }

    existing_count = 0
    for cat in report.categories:
        if cat.category == category.value:
            existing_count = cat.total_count
            break

    return [
        {
            "role": "system",
            "content": (
                "You are a creative director for a short-form video channel that produces "
                "'Oddly Satisfying' physics simulation videos using Pygame and Pymunk.\n\n"
                "Your job is to generate 3 RANKED concepts for the selected category. "
                "Each concept must be:\n"
                "- Visually distinct and satisfying to watch\n"
                "- Technically feasible with Pygame + Pymunk (2D rigid body physics)\n"
                "- Capable of filling a 65-second video with escalating visual interest\n"
                "- Hook-worthy (the first 2 seconds must intrigue viewers)\n\n"
                "Technical constraints:\n"
                "- Render resolution: 1080x1920 (9:16 portrait)\n"
                "- Target: 60 FPS, 65 seconds\n"
                "- Physics bodies max: ~500 (performance limit)\n"
                "- Dark background (#1a1a2e) for visual contrast\n"
                "- Colourful bodies with distinct colours\n"
                "- Must have a clear payoff/climax in the final 20% of the video\n\n"
                "Hook text rules: Maximum 6 words, posed as a question or intriguing statement.\n\n"
                "Rank concepts by: novelty (how fresh vs existing content) AND "
                "feasibility (how reliable to implement in Pymunk)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Category: {category.value}\n"
                f"Description: {category_descriptions.get(category, 'Physics simulation')}\n"
                f"Existing videos in this category: {existing_count}\n\n"
                "Generate 3 ranked concepts. The top concept should be your strongest "
                "recommendation balancing novelty and feasibility."
            ),
        },
    ]


async def develop_concept(
    category: ScenarioCategory,
    report: InventoryReport,
) -> ConceptDeveloperResponse:
    """Generate ranked concepts for a category using Claude Sonnet.

    Args:
        category: The scenario category to generate concepts for.
        report: Inventory report for context.

    Returns:
        ConceptDeveloperResponse with 3 ranked concepts.

    Raises:
        ConceptGenerationError: If the LLM call fails.
    """
    messages = _build_concept_developer_prompt(category, report)

    try:
        model = get_step_config("concept_developer").resolve_model()
        return await call_llm(
            model=model,
            messages=messages,
            response_model=ConceptDeveloperResponse,
            max_retries=2,
        )
    except Exception as exc:
        msg = f"Concept Developer failed for category '{category.value}'"
        raise ConceptGenerationError(msg) from exc


# =============================================================================
# Physics Idea Agent
# =============================================================================


class PhysicsIdeaAgent(BaseIdeaAgent):
    """Idea Agent for the Oddly Satisfying Physics pipeline.

    Orchestrates three subagents:
    1. Inventory Analyst — builds report from database stats
    2. Category Selector — selects category via rotation rules (+ optional LLM)
    3. Concept Developer — generates ranked concepts via Claude Sonnet

    The agent can operate in two modes:
    - Full mode: queries DB for stats, uses LLM for category selection
    - Lite mode: uses provided stats, programmatic category selection only
    """

    def __init__(self, *, use_llm_selector: bool = True) -> None:
        """Initialise the Physics Idea Agent.

        Args:
            use_llm_selector: If True, use LLM for category selection.
                If False, use pure programmatic rotation logic.
        """
        self._use_llm_selector = use_llm_selector

    async def generate_concept(self, state: PipelineState) -> ConceptBrief:
        """Generate a production-ready concept.

        Full pipeline:
        1. Build inventory report (Inventory Analyst)
        2. Select category (Category Selector)
        3. Generate ranked concepts (Concept Developer)
        4. Convert top concept to ConceptBrief

        Retries up to MAX_CONCEPT_ATTEMPTS times on failure.

        Args:
            state: Current pipeline state.

        Returns:
            A validated ConceptBrief.

        Raises:
            ConceptGenerationError: If concept generation fails after all retries.
        """
        pipeline = state.pipeline

        # Step 1: Build inventory report
        report = await self._build_report(pipeline)
        logger.info(
            "Inventory report: %d total videos, %d categories, last=%s",
            report.total_videos,
            len(report.categories),
            report.last_category,
        )

        last_error: Exception | None = None
        for attempt in range(1, MAX_CONCEPT_ATTEMPTS + 1):
            try:
                # Step 2: Select category
                if self._use_llm_selector:
                    selection = await select_category_with_llm(report)
                else:
                    selected = select_category_from_report(report)
                    selection = CategorySelection(
                        selected_category=ScenarioCategory(selected),
                        reasoning="Programmatic rotation selection",
                    )

                category = selection.selected_category
                logger.info(
                    "Category selected: %s (attempt %d/%d) — %s",
                    category.value,
                    attempt,
                    MAX_CONCEPT_ATTEMPTS,
                    selection.reasoning,
                )

                # Step 3: Generate concepts
                response = await develop_concept(category, report)
                top = response.top_concept
                logger.info(
                    "Top concept: '%s' (novelty=%.1f, feasibility=%.1f)",
                    top.title,
                    top.novelty_score,
                    top.feasibility_score,
                )

                # Step 4: Convert to ConceptBrief
                concept = ConceptBrief(
                    pipeline=pipeline,
                    category=category,
                    title=top.title,
                    visual_brief=top.visual_brief,
                    simulation_requirements=top.simulation_requirements,
                    audio_brief=top.audio_brief,
                    hook_text=top.hook_text,
                    novelty_score=top.novelty_score,
                    feasibility_score=top.feasibility_score,
                    target_duration_sec=65,
                    seed=random.randint(0, 2**31),  # noqa: S311
                )

                return concept

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Concept generation attempt %d/%d failed: %s",
                    attempt,
                    MAX_CONCEPT_ATTEMPTS,
                    exc,
                )

        msg = f"Failed to generate concept after {MAX_CONCEPT_ATTEMPTS} attempts"
        raise ConceptGenerationError(msg) from last_error

    async def get_category_stats(self, pipeline: str) -> dict[str, int]:
        """Get current category distribution.

        Returns a mapping of category name to total video count.

        Args:
            pipeline: Pipeline identifier.

        Returns:
            Dict mapping category names to their total video counts.
        """
        report = await self._build_report(pipeline)
        return {cat.category: cat.total_count for cat in report.categories}

    async def _build_report(self, pipeline: str) -> InventoryReport:
        """Build an inventory report.

        In production, this queries the DB. For now, provides default empty stats
        for all physics categories when no DB is available.

        Args:
            pipeline: Pipeline identifier.

        Returns:
            InventoryReport with category breakdown.
        """
        # Try to get stats from database
        try:
            return await self._build_report_from_db(pipeline)
        except Exception:
            logger.debug(
                "DB unavailable for inventory report, using defaults",
                exc_info=True,
            )

        # Fallback: empty stats for all categories
        return await build_inventory_report(
            pipeline,
            category_stats=[
                {"category": c, "total_count": 0, "videos_last_30_days": 0}
                for c in PHYSICS_CATEGORIES
            ],
        )

    async def _build_report_from_db(self, pipeline: str) -> InventoryReport:
        """Build inventory report from actual database queries.

        Args:
            pipeline: Pipeline identifier.

        Returns:
            InventoryReport from database state.

        Raises:
            Exception: If DB is not available.
        """
        from kairos.db.session import get_session

        async for session in get_session():
            from kairos.db.operations import (
                get_category_stats,
                get_last_idea_category,
                get_recent_ideas,
            )

            # Get category stats
            db_stats = await get_category_stats(session, pipeline)
            category_stats = [
                {
                    "category": stat.category,
                    "total_count": stat.total_count,
                    "videos_last_30_days": stat.videos_last_30_days,
                }
                for stat in db_stats
            ]

            # Get last category for hard block
            last_category = await get_last_idea_category(session, pipeline)

            # Get recent categories for streak detection
            recent_ideas = await get_recent_ideas(session, pipeline, limit=10)
            recent_categories = [
                idea.category for idea in recent_ideas if idea.category
            ]

            return await build_inventory_report(
                pipeline,
                category_stats=category_stats,
                recent_categories=recent_categories,
                last_category=last_category,
            )
