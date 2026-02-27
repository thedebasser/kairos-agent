"""Kairos Agent — Category Rotation Logic.

Pure functions for determining which scenario category to use next.
No LLM dependency — testable in isolation.

Rotation rules:
- Hard block: cannot repeat same category as previous video
- Soft block: deprioritise categories >30% of last 30 days output
- Boost: categories with <5 total videos
- Streak break: force switch after 3 consecutive same-category videos
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds
SOFT_BLOCK_PERCENTAGE = 0.30
BOOST_THRESHOLD = 5
MAX_STREAK = 3


@dataclass(frozen=True)
class CategoryInfo:
    """Category statistics for rotation decisions."""

    name: str
    total_count: int = 0
    videos_last_30_days: int = 0
    streak_count: int = 0
    is_last_used: bool = False


def select_category(
    categories: list[CategoryInfo],
    *,
    seed: int | None = None,
) -> str:
    """Select the next category respecting all rotation rules.

    Args:
        categories: Current stats for all available categories.
        seed: Optional random seed for deterministic testing.

    Returns:
        Selected category name.

    Raises:
        ValueError: If no categories are provided.
    """
    if not categories:
        msg = "No categories available for selection"
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311

    # If only one category exists, must use it regardless of rules
    if len(categories) == 1:
        return categories[0].name

    # Calculate total videos in last 30 days (for soft block percentage)
    total_last_30 = sum(c.videos_last_30_days for c in categories)

    candidates: list[tuple[str, float]] = []

    for cat in categories:
        weight = 1.0

        # Hard block: cannot repeat same category as previous video
        if cat.is_last_used:
            logger.debug("Hard block: skipping %s (last used)", cat.name)
            continue

        # Streak break: force switch after 3 consecutive same-category
        if cat.streak_count >= MAX_STREAK:
            logger.debug("Streak break: skipping %s (streak=%d)", cat.name, cat.streak_count)
            continue

        # Soft block: deprioritise categories >30% of last 30 days
        if total_last_30 > 0:
            pct = cat.videos_last_30_days / total_last_30
            if pct > SOFT_BLOCK_PERCENTAGE:
                weight *= 0.3
                logger.debug(
                    "Soft block: %s at %.0f%% (weight reduced)",
                    cat.name,
                    pct * 100,
                )

        # Boost: categories with <5 total videos
        if cat.total_count < BOOST_THRESHOLD:
            weight *= 2.0
            logger.debug("Boost: %s has %d videos (weight increased)", cat.name, cat.total_count)

        candidates.append((cat.name, weight))

    if not candidates:
        # Extreme edge case: all categories blocked. Fall back to any non-last category.
        non_last = [c for c in categories if not c.is_last_used]
        if non_last:
            return rng.choice(non_last).name
        # Absolute fallback: pick any category
        return rng.choice(categories).name

    # Weighted random selection
    names, weights = zip(*candidates, strict=True)
    selected = rng.choices(list(names), weights=list(weights), k=1)[0]
    logger.info("Selected category: %s", selected)
    return selected


def compute_streak(recent_categories: list[str], category: str) -> int:
    """Compute how many consecutive times a category was used recently.

    Args:
        recent_categories: Ordered list of recent categories (newest first).
        category: The category to count.

    Returns:
        Number of consecutive uses starting from the most recent.
    """
    streak = 0
    for cat in recent_categories:
        if cat == category:
            streak += 1
        else:
            break
    return streak
