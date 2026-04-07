"""Unit tests for simulation response models."""

from __future__ import annotations

import pytest

from kairos.schemas.simulation import AdjustedSimulationCode, SimulationCode

pytestmark = [pytest.mark.unit]


class TestSimulationCode:
    """Tests for SimulationCode model."""

    def test_creates_with_code(self) -> None:
        sc = SimulationCode(code="import bpy", reasoning="test")
        assert sc.code == "import bpy"
        assert sc.reasoning == "test"

    def test_is_frozen(self) -> None:
        sc = SimulationCode(code="code")
        with pytest.raises(Exception):  # noqa: B017
            sc.code = "other"  # type: ignore[misc]

    def test_default_reasoning(self) -> None:
        sc = SimulationCode(code="code")
        assert sc.reasoning == ""


class TestAdjustedSimulationCode:
    """Tests for AdjustedSimulationCode model."""

    def test_creates_with_changes(self) -> None:
        adj = AdjustedSimulationCode(
            code="fixed code",
            changes_made=["fix1", "fix2"],
            reasoning="swapped dims",
        )
        assert adj.code == "fixed code"
        assert len(adj.changes_made) == 2

    def test_default_changes_empty(self) -> None:
        adj = AdjustedSimulationCode(code="code")
        assert adj.changes_made == []
        assert adj.reasoning == ""

    def test_is_frozen(self) -> None:
        adj = AdjustedSimulationCode(code="code")
        with pytest.raises(Exception):  # noqa: B017
            adj.code = "other"  # type: ignore[misc]
