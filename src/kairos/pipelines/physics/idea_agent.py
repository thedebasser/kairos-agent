"""Physics Idea Agent.

Implements IdeaAgent for the "Oddly Satisfying Physics" pipeline.

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

from kairos.pipelines.contracts import IdeaAgent
from kairos.exceptions import ConceptGenerationError
from kairos.schemas.contracts import (
    ConceptBrief,
    IdeaAgentInput,
    ScenarioCategory,
)
from kairos.schemas.idea import (
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
from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import call_llm

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
    # Filter to only currently-active categories (enum members)
    valid_cats = {c.value for c in ScenarioCategory}
    category_infos = [
        CategoryInfo(
            name=cat.category,
            total_count=cat.total_count,
            videos_last_30_days=cat.videos_last_30_days,
            streak_count=cat.streak_count,
            is_last_used=cat.is_last_used,
        )
        for cat in report.categories
        if cat.category in valid_cats
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
    # Filter to only currently-active categories (enum members)
    valid_cats = {c.value for c in ScenarioCategory}
    categories_text = "\n".join(
        f"- {cat.category}: {cat.total_count} total, "
        f"{cat.videos_last_30_days} last 30 days ({cat.percentage_last_30_days}%), "
        f"streak={cat.streak_count}, last_used={cat.is_last_used}, "
        f"needs_boost={cat.needs_boost}"
        for cat in report.categories
        if cat.category in valid_cats
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a content strategist for a short-form video channel. "
                "Select the next scenario category based on rotation rules. "
                f"Available categories: {', '.join(PHYSICS_CATEGORIES)}. "
                "Prioritise variety, boost under-represented categories, "
                "and avoid repeating the last used category."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Category stats:\n{categories_text}\n\n"
                f"Last category: {report.last_category or 'None (first video)'}\n"
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
            cache_step="category_selector",
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
            "Ball pit / collision cascade: Colourful balls filling containers, bouncing, "
            "piling up. Simple obstacles (2-4 platforms/funnels). "
            "Key appeal: satisfying bouncing, cascading colour, accumulation → overflow. "
            "Physics: Blender Bullet rigid bodies with sphere colliders.\n"
            "PSYCHOLOGY: Tier 1 satisfaction (4+ mechanisms). Container overflow triggers "
            "Zeigarnik completion + catharsis + pattern recognition. The container "
            "MUST have a climax event (gate burst, wall collapse, overflow) — balls "
            "bouncing with no end state is the LOWEST satisfaction tier. "
            "Rainbow-sequence spawn colours encode time → visual narrative."
        ),
        # MARBLE_FUNNEL disabled — ramp geometry needs work
        # ScenarioCategory.MARBLE_FUNNEL: ( ... ),
        # DOMINO_CHAIN disabled — handled by the domino pipeline
        # ScenarioCategory.DOMINO_CHAIN: ( ... ),
        ScenarioCategory.DESTRUCTION: (
            "Destruction / tower collapse: A stable block tower hit by a wrecking ball. "
            "Simple tower (10-15 layers × 3-5 blocks). Pre-settled before recording. "
            "Key appeal: tension → dramatic collapse → debris scatter. "
            "Physics: Blender Bullet rigid bodies, mass-based dynamics.\n"
            "PSYCHOLOGY: Catharsis is the primary driver — the viewer must form an "
            "emotional connection to the intact structure BEFORE destruction. Show the "
            "tower standing for 10-15s. The destruction is 'earned' through anticipation. "
            "Contrast between order (structured tower) and chaos (collapse) creates "
            "the cathartic response. Resolution: rubble must FULLY settle."
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
                "'Oddly Satisfying' physics simulation videos using Blender 3D.\n\n"
                "Your job is to generate original, highly satisfying video concepts that are:\n"
                "- Visually appealing with satisfying physics interactions\n"
                "- Technically feasible with Blender 3D (rigid body physics)\n"
                "- Emotionally engaging with clear narrative arc\n"
                "- Optimised for portrait format (9:16)\n\n"
                "Generate 3 ranked concepts. The top concept should be your strongest "
                "recommendation — prioritise FEASIBILITY over novelty."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Category: {category.value}\n\n"
                f"Category description:\n"
                f"{category_descriptions.get(category, 'Physics simulation')}\n\n"
                f"Existing videos in this category: {existing_count}\n\n"
                "Generate 3 ranked concept ideas for this category."
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
            cache_step="concept_developer",
        )
    except Exception as exc:
        msg = f"Concept Developer failed for category '{category.value}'"
        raise ConceptGenerationError(msg) from exc


# =============================================================================
# Physics Idea Agent
# =============================================================================


class PhysicsIdeaAgent(IdeaAgent):
    """Idea Agent for the Oddly Satisfying Physics pipeline.

    Orchestrates three subagents:
    1. Inventory Analyst — builds report from database stats
    2. Category Selector — selects category via rotation rules (+ optional LLM)
    3. Concept Developer — generates ranked concepts via Claude Sonnet

    The agent can operate in two modes:
    - Full mode: queries DB for stats, uses LLM for category selection
    - Lite mode: uses provided stats, programmatic category selection only
    """

    def __init__(self, *, use_llm_selector: bool = True, force_category: str | None = None) -> None:
        """Initialise the Physics Idea Agent.

        Args:
            use_llm_selector: If True, use LLM for category selection.
                If False, use pure programmatic rotation logic.
            force_category: If set, always use this category (e.g. 'ball_pit').
        """
        self._use_llm_selector = use_llm_selector
        self._force_category = force_category

    async def generate_concept(self, input: IdeaAgentInput) -> ConceptBrief:
        """Generate a production-ready concept.

        Full pipeline:
        1. Build inventory report (Inventory Analyst)
        2. Select category (Category Selector)
        3. Generate ranked concepts (Concept Developer)
        4. Convert top concept to ConceptBrief

        Retries up to MAX_CONCEPT_ATTEMPTS times on failure.

        Args:
            input: Narrow DTO containing only the pipeline name.

        Returns:
            A validated ConceptBrief.

        Raises:
            ConceptGenerationError: If concept generation fails after all retries.
        """
        pipeline = input.pipeline

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
                if self._force_category:
                    selection = CategorySelection(
                        selected_category=ScenarioCategory(self._force_category),
                        reasoning=f"Forced category: {self._force_category}",
                    )
                elif self._use_llm_selector:
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
