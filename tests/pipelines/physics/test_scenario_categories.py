"""Tests for scenario categories."""

import pytest

from kairos.models.contracts import ScenarioCategory

pytestmark = pytest.mark.pipeline


class TestScenarioCategories:
    """Tests for physics scenario categories."""

    def test_all_poc_categories(self):
        expected = {"ball_pit", "marble_funnel", "domino_chain", "destruction"}
        actual = {c.value for c in ScenarioCategory}
        assert actual == expected

    def test_ball_pit_category(self):
        cat = ScenarioCategory.BALL_PIT
        assert cat.value == "ball_pit"

    def test_marble_funnel_category(self):
        cat = ScenarioCategory.MARBLE_FUNNEL
        assert cat.value == "marble_funnel"

    def test_domino_chain_category(self):
        cat = ScenarioCategory.DOMINO_CHAIN
        assert cat.value == "domino_chain"

    def test_destruction_category(self):
        cat = ScenarioCategory.DESTRUCTION
        assert cat.value == "destruction"
