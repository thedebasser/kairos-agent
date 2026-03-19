"""Tests for scenario categories."""

import pytest

from kairos.schemas.contracts import ScenarioCategory

pytestmark = pytest.mark.pipeline


class TestScenarioCategories:
    """Tests for physics scenario categories."""

    def test_all_poc_categories(self):
        # MARBLE_FUNNEL and DOMINO_CHAIN are disabled in the enum pending rework
        expected = {"ball_pit", "destruction"}
        actual = {c.value for c in ScenarioCategory}
        assert actual == expected

    def test_ball_pit_category(self):
        cat = ScenarioCategory.BALL_PIT
        assert cat.value == "ball_pit"

    def test_destruction_category(self):
        cat = ScenarioCategory.DESTRUCTION
        assert cat.value == "destruction"
