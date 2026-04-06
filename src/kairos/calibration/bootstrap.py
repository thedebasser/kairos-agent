"""Kairos Agent — Calibration Bootstrap Scenarios.

Defines the Phase 1 bootstrap scenarios (using current Blender capabilities)
and provides a batch runner to seed the knowledge base.

Phase 1 covers: straight, s-curve, spiral, cascade, branching
  with parameter variations (count, amplitude, turns, branches).

Phase 2 (future): inclines, size profiles, transitions — requires new
  Blender script capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kairos.calibration.models import (
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
    SizeProfile,
    SizeProfileDescriptor,
    SurfaceDescriptor,
    SurfaceType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1 Bootstrap Scenarios
# =============================================================================

PHASE_1_SCENARIOS: list[dict[str, Any]] = [
    # 1. Straight line, 30 dominoes (minimal)
    {
        "name": "straight_30",
        "description": "Straight line, 30 dominoes — baseline straight path",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0, cycles=1.0),
            domino_count=30,
        ),
    },
    # 2. Straight line, 300 dominoes (scaling test)
    {
        "name": "straight_300",
        "description": "Straight line, 300 dominoes — scaling behaviour",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0, cycles=1.0),
            domino_count=300,
        ),
    },
    # 3. Gentle S-curve (standard production config)
    {
        "name": "s_curve_gentle",
        "description": "Gentle S-curve, amplitude 1.0, 2 cycles — production default",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.S_CURVE, amplitude=1.0, cycles=2.0),
            domino_count=300,
        ),
    },
    # 4. Tight S-curve (stress test curves)
    {
        "name": "s_curve_tight",
        "description": "Tight S-curve, amplitude 3.0, 3 cycles — stress test",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.S_CURVE, amplitude=3.0, cycles=3.0),
            domino_count=300,
        ),
    },
    # 5. Spiral, 2 turns
    {
        "name": "spiral_2",
        "description": "Spiral, 2 turns, 200 dominoes",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=2.0),
            domino_count=200,
        ),
    },
    # 6. Spiral, 4 turns (dense)
    {
        "name": "spiral_4",
        "description": "Spiral, 4 turns, 300 dominoes — dense spiral",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=4.0),
            domino_count=300,
        ),
    },
    # 7. Cascade, 150 dominoes
    {
        "name": "cascade_150",
        "description": "Cascade, 150 dominoes — moderate zigzag",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.CASCADE),
            domino_count=150,
        ),
    },
    # 8. Cascade, 400 dominoes (large)
    {
        "name": "cascade_400",
        "description": "Cascade, 400 dominoes — large-scale cascade",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.CASCADE),
            domino_count=400,
        ),
    },
    # 9. Branching, 3 branches
    {
        "name": "branching_3",
        "description": "Branching, 3 branches, 200 dominoes",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.BRANCHING, branch_count=3),
            domino_count=200,
        ),
    },
    # 10. Branching, 5 branches
    {
        "name": "branching_5",
        "description": "Branching, 5 branches, 300 dominoes — max branches",
        "scenario": ScenarioDescriptor(
            path=PathDescriptor(type=PathType.BRANCHING, branch_count=5),
            domino_count=300,
        ),
    },
]


async def run_bootstrap(
    *,
    scenarios: list[dict[str, Any]] | None = None,
    max_iterations: int = 10,
) -> list[dict[str, Any]]:
    """Run the bootstrap calibration for all Phase 1 scenarios.

    Args:
        scenarios: Override scenario list (default: PHASE_1_SCENARIOS).
        max_iterations: Max iterations per scenario.

    Returns:
        List of result summaries per scenario.
    """
    from kairos.calibration.knowledge_base import KnowledgeBase
    from kairos.calibration.sandbox import run_calibration

    if scenarios is None:
        scenarios = PHASE_1_SCENARIOS

    kb = KnowledgeBase()
    results: list[dict[str, Any]] = []

    for i, scenario_def in enumerate(scenarios, 1):
        name = scenario_def["name"]
        desc = scenario_def["description"]
        scenario = scenario_def["scenario"]

        logger.info(
            "=" * 60 + "\n"
            "[bootstrap] Scenario %d/%d: %s\n"
            "[bootstrap] %s\n" + "=" * 60,
            i, len(scenarios), name, desc,
        )

        try:
            session = await run_calibration(
                scenario,
                knowledge_base=kb,
                max_iterations=max_iterations,
            )
            results.append({
                "name": name,
                "status": session.status.value,
                "iterations": session.iteration_count,
                "confidence": session.confidence,
                "session_id": str(session.session_id),
            })
            logger.info(
                "[bootstrap] %s: %s in %d iterations (confidence=%.2f)",
                name, session.status.value, session.iteration_count, session.confidence,
            )
        except Exception:
            logger.exception("[bootstrap] %s: FAILED with exception", name)
            results.append({
                "name": name,
                "status": "error",
                "iterations": 0,
                "confidence": 0.0,
                "error": True,
            })

    # Summary
    resolved = sum(1 for r in results if r["status"] == "promoted")
    unresolved = sum(1 for r in results if r["status"] == "unresolved")
    errors = sum(1 for r in results if r.get("error"))
    logger.info(
        "[bootstrap] Complete: %d/%d promoted, %d unresolved, %d errors",
        resolved, len(results), unresolved, errors,
    )
    logger.info("[bootstrap] Knowledge base now has %d entries", kb.count())

    return results
