"""Unit tests for category rotation logic.

Tests all rotation rules: hard block, soft block, boost, streak break.
"""

import pytest

from kairos.services.category_rotation import (
    BOOST_THRESHOLD,
    MAX_STREAK,
    SOFT_BLOCK_PERCENTAGE,
    CategoryInfo,
    compute_streak,
    select_category,
)

pytestmark = pytest.mark.unit


class TestHardBlock:
    """Hard block: cannot repeat same category as previous video."""

    def test_last_used_never_selected(self, categories_with_last_used):
        """Category marked as last_used should never be selected."""
        for _ in range(50):
            selected = select_category(categories_with_last_used, seed=_)
            assert selected != "ball_pit", "Hard block violated: selected last-used category"

    def test_all_except_last_available(self, categories_with_last_used):
        """All categories except the last-used should be selectable."""
        selected_set = set()
        for i in range(200):
            selected_set.add(select_category(categories_with_last_used, seed=i))
        assert "ball_pit" not in selected_set
        # At least some of the other categories should be selected
        assert len(selected_set) >= 2


class TestSoftBlock:
    """Soft block: deprioritise categories >30% of last 30 days output."""

    def test_overused_category_deprioritised(self):
        """Category with >30% of last 30 days should be selected less often."""
        categories = [
            CategoryInfo(name="ball_pit", total_count=10, videos_last_30_days=8),
            CategoryInfo(name="marble_funnel", total_count=10, videos_last_30_days=1),
            CategoryInfo(name="domino_chain", total_count=10, videos_last_30_days=1),
        ]
        # ball_pit has 80% of last 30 days (8/10) — should be deprioritised
        counts = {"ball_pit": 0, "marble_funnel": 0, "domino_chain": 0}
        for i in range(500):
            selected = select_category(categories, seed=i)
            counts[selected] += 1

        # ball_pit should be selected less often than others
        assert counts["ball_pit"] < counts["marble_funnel"] + counts["domino_chain"]


class TestBoost:
    """Boost: categories with <5 total videos get higher weight."""

    def test_new_category_boosted(self):
        """Category with <5 total videos should be selected more often."""
        categories = [
            CategoryInfo(name="ball_pit", total_count=50, videos_last_30_days=5),
            CategoryInfo(name="marble_funnel", total_count=50, videos_last_30_days=5),
            CategoryInfo(name="new_category", total_count=2, videos_last_30_days=1),
        ]
        counts = {"ball_pit": 0, "marble_funnel": 0, "new_category": 0}
        for i in range(500):
            selected = select_category(categories, seed=i)
            counts[selected] += 1

        # new_category should be selected more often due to boost
        assert counts["new_category"] > counts["ball_pit"]


class TestStreakBreak:
    """Streak break: force switch after 3 consecutive same-category videos."""

    def test_streak_at_max_blocked(self):
        """Category at max streak count should be blocked."""
        categories = [
            CategoryInfo(
                name="ball_pit",
                total_count=10,
                videos_last_30_days=3,
                streak_count=MAX_STREAK,
            ),
            CategoryInfo(name="marble_funnel", total_count=10, videos_last_30_days=3),
            CategoryInfo(name="domino_chain", total_count=10, videos_last_30_days=3),
        ]
        for i in range(50):
            selected = select_category(categories, seed=i)
            assert selected != "ball_pit", "Streak break violated"

    def test_streak_below_max_allowed(self):
        """Category below max streak should still be selectable."""
        categories = [
            CategoryInfo(
                name="ball_pit",
                total_count=10,
                videos_last_30_days=3,
                streak_count=MAX_STREAK - 1,
            ),
            CategoryInfo(name="marble_funnel", total_count=10, videos_last_30_days=3),
        ]
        selected_set = set()
        for i in range(100):
            selected_set.add(select_category(categories, seed=i))
        assert "ball_pit" in selected_set


class TestEdgeCases:
    """Edge cases for category selection."""

    def test_empty_categories_raises(self):
        """Empty category list should raise ValueError."""
        with pytest.raises(ValueError, match="No categories available"):
            select_category([])

    def test_single_category(self):
        """Single category must be selected regardless of rules."""
        categories = [
            CategoryInfo(
                name="only_one",
                total_count=100,
                videos_last_30_days=30,
                streak_count=5,
                is_last_used=True,
            ),
        ]
        assert select_category(categories) == "only_one"

    def test_empty_database_first_video(self):
        """First video ever — all categories have zero stats."""
        categories = [
            CategoryInfo(name="ball_pit"),
            CategoryInfo(name="marble_funnel"),
            CategoryInfo(name="domino_chain"),
            CategoryInfo(name="destruction"),
        ]
        selected = select_category(categories, seed=42)
        assert selected in {"ball_pit", "marble_funnel", "domino_chain", "destruction"}

    def test_deterministic_with_seed(self, all_categories):
        """Same seed produces same result."""
        result1 = select_category(all_categories, seed=42)
        result2 = select_category(all_categories, seed=42)
        assert result1 == result2


class TestComputeStreak:
    """Tests for compute_streak helper."""

    def test_no_streak(self):
        assert compute_streak(["marble_funnel", "ball_pit"], "domino_chain") == 0

    def test_streak_of_one(self):
        assert compute_streak(["ball_pit", "marble_funnel"], "ball_pit") == 1

    def test_streak_of_three(self):
        assert compute_streak(["ball_pit", "ball_pit", "ball_pit", "marble_funnel"], "ball_pit") == 3

    def test_empty_history(self):
        assert compute_streak([], "ball_pit") == 0
