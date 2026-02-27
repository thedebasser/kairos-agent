"""Unit tests for the Physics Idea Agent.

Tests the full Idea Agent pipeline:
- Inventory Analyst (report building)
- Category Selector (programmatic and LLM-backed)
- Concept Developer (LLM concept generation)
- PhysicsIdeaAgent.generate_concept() end-to-end
- PhysicsIdeaAgent.get_category_stats()
- Adapter integration
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.models.contracts import (
    AudioBrief,
    ConceptBrief,
    EnergyLevel,
    PipelineState,
    ScenarioCategory,
    SimulationRequirements,
)
from kairos.models.idea import (
    CategoryReport,
    CategorySelection,
    ConceptDeveloperResponse,
    InventoryReport,
    RankedConcept,
)
from kairos.pipelines.physics.idea_agent import (
    MAX_CONCEPT_ATTEMPTS,
    PHYSICS_CATEGORIES,
    PhysicsIdeaAgent,
    build_inventory_report,
    develop_concept,
    select_category_from_report,
    select_category_with_llm,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helpers / Fixtures
# =============================================================================


def _make_ranked_concept(
    *,
    title: str = "Test Concept",
    category: ScenarioCategory = ScenarioCategory.BALL_PIT,
    hook: str = "What happens next?",
) -> RankedConcept:
    """Create a valid RankedConcept for testing."""
    return RankedConcept(
        title=title,
        visual_brief="200 balls cascade into a pit creating a satisfying collision chain.",
        simulation_requirements=SimulationRequirements(
            body_count_initial=50,
            body_count_max=500,
            interaction_type="collision_spawn",
        ),
        audio_brief=AudioBrief(
            mood=["upbeat", "energetic"],
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text=hook,
        novelty_score=7.5,
        feasibility_score=8.0,
        rank_reasoning="Strong visual appeal with proven physics mechanics.",
    )


def _make_concept_response(
    category: ScenarioCategory = ScenarioCategory.BALL_PIT,
) -> ConceptDeveloperResponse:
    """Create a valid ConceptDeveloperResponse with 3 ranked concepts."""
    return ConceptDeveloperResponse(
        concepts=[
            _make_ranked_concept(title="Top Concept", category=category),
            _make_ranked_concept(title="Second Concept", category=category),
            _make_ranked_concept(title="Third Concept", category=category),
        ]
    )


def _make_inventory_report(
    *,
    last_category: str | None = "ball_pit",
    recent: list[str] | None = None,
) -> InventoryReport:
    """Create a sample inventory report."""
    return InventoryReport(
        pipeline="physics",
        total_videos=36,
        categories=[
            CategoryReport(
                category="ball_pit",
                total_count=10,
                videos_last_30_days=3,
                streak_count=0,
                percentage_last_30_days=30.0,
                is_last_used=(last_category == "ball_pit"),
            ),
            CategoryReport(
                category="marble_funnel",
                total_count=8,
                videos_last_30_days=2,
                streak_count=0,
                percentage_last_30_days=20.0,
                is_last_used=(last_category == "marble_funnel"),
            ),
            CategoryReport(
                category="domino_chain",
                total_count=12,
                videos_last_30_days=4,
                streak_count=0,
                percentage_last_30_days=40.0,
                is_last_used=(last_category == "domino_chain"),
            ),
            CategoryReport(
                category="destruction",
                total_count=6,
                videos_last_30_days=1,
                streak_count=0,
                percentage_last_30_days=10.0,
                is_last_used=(last_category == "destruction"),
            ),
        ],
        last_category=last_category,
        recent_categories=recent or ["ball_pit", "domino_chain", "marble_funnel"],
    )


# =============================================================================
# Test: Inventory Analyst (build_inventory_report)
# =============================================================================


class TestBuildInventoryReport:
    """Tests for the Inventory Analyst report builder."""

    @pytest.mark.asyncio
    async def test_basic_report(self):
        """Report builds correctly from category stats."""
        report = await build_inventory_report(
            "physics",
            category_stats=[
                {"category": "ball_pit", "total_count": 10, "videos_last_30_days": 3},
                {"category": "marble_funnel", "total_count": 8, "videos_last_30_days": 2},
            ],
            last_category="ball_pit",
            recent_categories=["ball_pit", "marble_funnel"],
        )
        assert report.pipeline == "physics"
        assert report.total_videos == 18
        assert len(report.categories) == 2
        assert report.last_category == "ball_pit"

    @pytest.mark.asyncio
    async def test_empty_stats(self):
        """Empty stats produce empty report."""
        report = await build_inventory_report("physics")
        assert report.total_videos == 0
        assert report.categories == []
        assert report.last_category is None

    @pytest.mark.asyncio
    async def test_percentage_calculation(self):
        """Percentages calculated correctly from videos_last_30_days."""
        report = await build_inventory_report(
            "physics",
            category_stats=[
                {"category": "ball_pit", "total_count": 10, "videos_last_30_days": 3},
                {"category": "marble_funnel", "total_count": 10, "videos_last_30_days": 7},
            ],
        )
        # ball_pit: 3/10 = 30%, marble_funnel: 7/10 = 70%
        ball_pit = next(c for c in report.categories if c.category == "ball_pit")
        marble = next(c for c in report.categories if c.category == "marble_funnel")
        assert ball_pit.percentage_last_30_days == 30.0
        assert marble.percentage_last_30_days == 70.0

    @pytest.mark.asyncio
    async def test_streak_detection(self):
        """Streak count computed from recent categories."""
        report = await build_inventory_report(
            "physics",
            category_stats=[
                {"category": "ball_pit", "total_count": 10, "videos_last_30_days": 5},
                {"category": "marble_funnel", "total_count": 10, "videos_last_30_days": 5},
            ],
            recent_categories=["ball_pit", "ball_pit", "ball_pit", "marble_funnel"],
        )
        ball_pit = next(c for c in report.categories if c.category == "ball_pit")
        assert ball_pit.streak_count == 3

    @pytest.mark.asyncio
    async def test_boost_detection(self):
        """Categories with <5 total videos flagged as needs_boost."""
        report = await build_inventory_report(
            "physics",
            category_stats=[
                {"category": "ball_pit", "total_count": 2, "videos_last_30_days": 1},
                {"category": "marble_funnel", "total_count": 50, "videos_last_30_days": 5},
            ],
        )
        ball_pit = next(c for c in report.categories if c.category == "ball_pit")
        marble = next(c for c in report.categories if c.category == "marble_funnel")
        assert ball_pit.needs_boost is True
        assert marble.needs_boost is False

    @pytest.mark.asyncio
    async def test_is_last_used_flag(self):
        """is_last_used correctly set based on last_category."""
        report = await build_inventory_report(
            "physics",
            category_stats=[
                {"category": "ball_pit", "total_count": 10, "videos_last_30_days": 3},
                {"category": "marble_funnel", "total_count": 8, "videos_last_30_days": 2},
            ],
            last_category="marble_funnel",
        )
        ball_pit = next(c for c in report.categories if c.category == "ball_pit")
        marble = next(c for c in report.categories if c.category == "marble_funnel")
        assert ball_pit.is_last_used is False
        assert marble.is_last_used is True


# =============================================================================
# Test: Category Selector (programmatic)
# =============================================================================


class TestSelectCategoryFromReport:
    """Tests for programmatic category selection from inventory report."""

    def test_respects_hard_block(self):
        """Last-used category is never selected."""
        report = _make_inventory_report(last_category="ball_pit")
        for seed in range(100):
            selected = select_category_from_report(report, seed=seed)
            assert selected != "ball_pit"

    def test_selects_valid_category(self):
        """Selected category is always a valid physics category."""
        report = _make_inventory_report()
        for seed in range(50):
            selected = select_category_from_report(report, seed=seed)
            assert selected in PHYSICS_CATEGORIES

    def test_empty_report_uses_defaults(self):
        """Empty report falls back to all physics categories."""
        report = InventoryReport(
            pipeline="physics",
            total_videos=0,
            categories=[],
        )
        selected = select_category_from_report(report, seed=42)
        assert selected in PHYSICS_CATEGORIES

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        report = _make_inventory_report()
        r1 = select_category_from_report(report, seed=42)
        r2 = select_category_from_report(report, seed=42)
        assert r1 == r2


# =============================================================================
# Test: Category Selector (LLM-backed)
# =============================================================================


class TestSelectCategoryWithLLM:
    """Tests for LLM-backed category selection."""

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_returns_llm_selection(self, mock_call):
        """Returns the LLM's category selection."""
        mock_call.return_value = CategorySelection(
            selected_category=ScenarioCategory.MARBLE_FUNNEL,
            reasoning="Marble funnel is underrepresented.",
        )
        report = _make_inventory_report()
        result = await select_category_with_llm(report)
        assert result.selected_category == ScenarioCategory.MARBLE_FUNNEL
        assert "underrepresented" in result.reasoning
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_fallback_on_llm_failure(self, mock_call):
        """Falls back to programmatic selection when LLM fails."""
        mock_call.side_effect = RuntimeError("LLM unavailable")
        report = _make_inventory_report(last_category="ball_pit")
        result = await select_category_with_llm(report, seed=42)
        # Should still produce a valid selection
        assert result.selected_category.value in PHYSICS_CATEGORIES
        assert result.selected_category.value != "ball_pit"  # respects hard block
        assert "Fallback" in result.reasoning


# =============================================================================
# Test: Concept Developer
# =============================================================================


class TestDevelopConcept:
    """Tests for the Concept Developer LLM call."""

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_returns_concept_response(self, mock_call):
        """Returns a valid ConceptDeveloperResponse from LLM."""
        expected = _make_concept_response(ScenarioCategory.BALL_PIT)
        mock_call.return_value = expected
        report = _make_inventory_report()

        result = await develop_concept(ScenarioCategory.BALL_PIT, report)
        assert len(result.concepts) == 3
        assert result.top_concept.title == "Top Concept"
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_raises_on_failure(self, mock_call):
        """Raises ConceptGenerationError when LLM fails."""
        from kairos.exceptions import ConceptGenerationError

        mock_call.side_effect = RuntimeError("API error")
        report = _make_inventory_report()

        with pytest.raises(ConceptGenerationError, match="Concept Developer failed"):
            await develop_concept(ScenarioCategory.BALL_PIT, report)

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_uses_concept_developer_model(self, mock_call):
        """Calls the 'concept-developer' model (Claude Sonnet)."""
        mock_call.return_value = _make_concept_response()
        report = _make_inventory_report()

        await develop_concept(ScenarioCategory.BALL_PIT, report)
        call_args = mock_call.call_args
        assert call_args.kwargs["model"] == "concept-developer"


# =============================================================================
# Test: PhysicsIdeaAgent.generate_concept()
# =============================================================================


class TestGenerateConcept:
    """Tests for the full generate_concept pipeline."""

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_happy_path(self, mock_call):
        """Full pipeline produces a valid ConceptBrief."""
        # Mock category selector + concept developer LLM calls
        mock_call.side_effect = [
            CategorySelection(
                selected_category=ScenarioCategory.MARBLE_FUNNEL,
                reasoning="Good variety",
            ),
            _make_concept_response(ScenarioCategory.MARBLE_FUNNEL),
        ]

        agent = PhysicsIdeaAgent(use_llm_selector=True)
        state = PipelineState(pipeline="physics")

        # Patch _build_report to avoid DB dependency
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        concept = await agent.generate_concept(state)
        assert isinstance(concept, ConceptBrief)
        assert concept.pipeline == "physics"
        assert concept.title == "Top Concept"
        assert concept.novelty_score == 7.5
        assert concept.feasibility_score == 8.0
        assert concept.target_duration_sec == 65
        assert concept.seed is not None

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_programmatic_selector(self, mock_call):
        """With use_llm_selector=False, only concept developer LLM is called."""
        mock_call.return_value = _make_concept_response()

        agent = PhysicsIdeaAgent(use_llm_selector=False)
        state = PipelineState(pipeline="physics")
        agent._build_report = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_inventory_report(last_category="ball_pit"),
        )

        concept = await agent.generate_concept(state)
        assert isinstance(concept, ConceptBrief)
        # Only one LLM call (concept developer), not two
        assert mock_call.call_count == 1
        assert mock_call.call_args.kwargs["model"] == "concept-developer"

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_concept_has_valid_category(self, mock_call):
        """Generated concept has a valid ScenarioCategory."""
        mock_call.side_effect = [
            CategorySelection(
                selected_category=ScenarioCategory.DESTRUCTION,
                reasoning="Fresh category",
            ),
            _make_concept_response(ScenarioCategory.DESTRUCTION),
        ]

        agent = PhysicsIdeaAgent()
        state = PipelineState(pipeline="physics")
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        concept = await agent.generate_concept(state)
        assert concept.category == ScenarioCategory.DESTRUCTION

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_raises_on_concept_generation_error(self, mock_call):
        """Raises ConceptGenerationError when concept developer fails."""
        from kairos.exceptions import ConceptGenerationError

        mock_call.side_effect = [
            CategorySelection(
                selected_category=ScenarioCategory.BALL_PIT,
                reasoning="test",
            ),
            ConceptGenerationError("LLM failed for category 'ball_pit'"),
        ]

        agent = PhysicsIdeaAgent()
        state = PipelineState(pipeline="physics")
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        with pytest.raises(ConceptGenerationError):
            await agent.generate_concept(state)

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_retries_on_failure(self, mock_call):
        """Retries concept generation on non-ConceptGenerationError failures."""
        # First attempt: category OK, concept fails with generic error
        # Second attempt: both succeed
        mock_call.side_effect = [
            CategorySelection(
                selected_category=ScenarioCategory.BALL_PIT,
                reasoning="test",
            ),
            RuntimeError("Transient error"),
            CategorySelection(
                selected_category=ScenarioCategory.BALL_PIT,
                reasoning="retry",
            ),
            _make_concept_response(),
        ]

        agent = PhysicsIdeaAgent()
        state = PipelineState(pipeline="physics")
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        # Should succeed on second attempt
        concept = await agent.generate_concept(state)
        assert isinstance(concept, ConceptBrief)

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_max_retries_exhausted(self, mock_call):
        """Raises after MAX_CONCEPT_ATTEMPTS failures."""
        from kairos.exceptions import ConceptGenerationError

        # All attempts fail with generic errors
        mock_call.side_effect = RuntimeError("Persistent error")

        agent = PhysicsIdeaAgent(use_llm_selector=False)
        state = PipelineState(pipeline="physics")
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        with pytest.raises(ConceptGenerationError, match="Failed to generate concept"):
            await agent.generate_concept(state)

    @pytest.mark.asyncio
    @patch("kairos.pipelines.physics.idea_agent.call_llm")
    async def test_empty_database_first_video(self, mock_call):
        """Works correctly with empty database (first video ever)."""
        mock_call.return_value = _make_concept_response()

        agent = PhysicsIdeaAgent(use_llm_selector=False)
        state = PipelineState(pipeline="physics")
        # Empty inventory report
        agent._build_report = AsyncMock(  # type: ignore[method-assign]
            return_value=InventoryReport(
                pipeline="physics",
                total_videos=0,
                categories=[],
            ),
        )

        concept = await agent.generate_concept(state)
        assert isinstance(concept, ConceptBrief)


# =============================================================================
# Test: PhysicsIdeaAgent.get_category_stats()
# =============================================================================


class TestGetCategoryStats:
    """Tests for get_category_stats."""

    @pytest.mark.asyncio
    async def test_returns_category_counts(self):
        """Returns correct category -> count mapping."""
        agent = PhysicsIdeaAgent(use_llm_selector=False)
        agent._build_report = AsyncMock(return_value=_make_inventory_report())  # type: ignore[method-assign]

        stats = await agent.get_category_stats("physics")
        assert stats == {
            "ball_pit": 10,
            "marble_funnel": 8,
            "domino_chain": 12,
            "destruction": 6,
        }

    @pytest.mark.asyncio
    async def test_empty_database(self):
        """Returns empty dict for empty database."""
        agent = PhysicsIdeaAgent(use_llm_selector=False)
        agent._build_report = AsyncMock(  # type: ignore[method-assign]
            return_value=InventoryReport(
                pipeline="physics",
                total_videos=0,
                categories=[],
            ),
        )

        stats = await agent.get_category_stats("physics")
        assert stats == {}


# =============================================================================
# Test: Pydantic Model Validation
# =============================================================================


class TestIdeaModels:
    """Tests for Idea Agent response models."""

    def test_category_selection_valid(self):
        """Valid CategorySelection passes validation."""
        sel = CategorySelection(
            selected_category=ScenarioCategory.BALL_PIT,
            reasoning="Fresh category",
        )
        assert sel.selected_category == ScenarioCategory.BALL_PIT

    def test_category_selection_invalid_category(self):
        """Invalid category string is rejected."""
        with pytest.raises(ValueError):
            CategorySelection(
                selected_category="not_a_category",  # type: ignore[arg-type]
                reasoning="test",
            )

    def test_ranked_concept_valid(self):
        """Valid RankedConcept passes validation."""
        concept = _make_ranked_concept()
        assert concept.novelty_score == 7.5
        assert concept.feasibility_score == 8.0

    def test_ranked_concept_score_bounds(self):
        """Scores must be 0-10."""
        with pytest.raises(ValueError):
            RankedConcept(
                title="Test",
                visual_brief="Test",
                simulation_requirements=SimulationRequirements(
                    body_count_initial=50,
                    body_count_max=500,
                    interaction_type="test",
                ),
                hook_text="Test hook",
                novelty_score=11.0,  # Out of bounds
                feasibility_score=8.0,
                rank_reasoning="Test",
            )

    def test_concept_developer_response_top_concept(self):
        """top_concept returns the first concept."""
        response = _make_concept_response()
        assert response.top_concept.title == "Top Concept"

    def test_concept_developer_response_requires_at_least_one(self):
        """Must have at least one concept."""
        with pytest.raises(ValueError):
            ConceptDeveloperResponse(concepts=[])

    def test_inventory_report_valid(self):
        """Valid InventoryReport passes validation."""
        report = _make_inventory_report()
        assert report.total_videos == 36
        assert len(report.categories) == 4

    def test_category_report_defaults(self):
        """CategoryReport has sensible defaults."""
        report = CategoryReport(category="test")
        assert report.total_count == 0
        assert report.videos_last_30_days == 0
        assert report.streak_count == 0
        assert report.is_last_used is False
        assert report.needs_boost is False


# =============================================================================
# Test: Adapter Integration
# =============================================================================


class TestAdapterIntegration:
    """Tests for the PhysicsPipelineAdapter.get_idea_agent()."""

    def test_adapter_returns_idea_agent(self):
        """Adapter's get_idea_agent returns a PhysicsIdeaAgent."""
        from kairos.pipelines.physics.adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_idea_agent()
        assert isinstance(agent, PhysicsIdeaAgent)

    def test_agent_is_base_idea_agent(self):
        """Returned agent is a BaseIdeaAgent subclass."""
        from kairos.agents.base import BaseIdeaAgent
        from kairos.pipelines.physics.adapter import PhysicsPipelineAdapter

        adapter = PhysicsPipelineAdapter()
        agent = adapter.get_idea_agent()
        assert isinstance(agent, BaseIdeaAgent)
