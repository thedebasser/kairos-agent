"""Kairos Agent — Golden Set Regression Runner.

Runs the pipeline against a fixed set of concept briefs and tracks
success rate, iteration count, and cost trends over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "golden_set" / "concept_briefs.json"


@dataclass
class RegressionResult:
    """Result of running one concept brief through the pipeline."""

    concept_id: str
    category: str
    title: str
    success: bool
    iterations: int = 0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    error: str | None = None


@dataclass
class RegressionReport:
    """Summary report from a full regression run."""

    run_timestamp: datetime = field(default_factory=datetime.now)
    results: list[RegressionResult] = field(default_factory=list)
    total_concepts: int = 0
    total_success: int = 0
    total_failed: int = 0
    success_rate: float = 0.0
    avg_iterations: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_duration_sec: float = 0.0

    def summarize(self) -> dict[str, Any]:
        """Get a summary dict for logging/reporting."""
        return {
            "timestamp": self.run_timestamp.isoformat(),
            "total_concepts": self.total_concepts,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_iterations": f"{self.avg_iterations:.1f}",
            "avg_cost_usd": f"${self.avg_cost_usd:.4f}",
            "total_cost_usd": f"${self.total_cost_usd:.4f}",
            "total_duration_sec": f"{self.total_duration_sec:.1f}",
            "failures": [
                {"concept_id": r.concept_id, "title": r.title, "error": r.error}
                for r in self.results
                if not r.success
            ],
        }


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the golden set of concept briefs from JSON.

    Args:
        path: Override path to the JSON file.

    Returns:
        List of concept brief dicts.
    """
    filepath = path or GOLDEN_SET_PATH
    if not filepath.exists():
        raise FileNotFoundError(f"Golden set not found: {filepath}")

    with open(filepath) as f:
        briefs = json.load(f)

    logger.info("Loaded %d golden set concept briefs from %s", len(briefs), filepath)
    return briefs


async def run_regression(
    briefs: list[dict[str, Any]],
    *,
    pipeline_runner: Any = None,
) -> RegressionReport:
    """Run regression against all golden set briefs.

    Args:
        briefs: List of concept brief dicts.
        pipeline_runner: Callable to run a single concept through the pipeline.
            Signature: async (concept_brief) -> RegressionResult.
            If None, uses a stub that always succeeds (for testing the framework).

    Returns:
        RegressionReport with aggregated results.
    """
    report = RegressionReport(total_concepts=len(briefs))

    for brief in briefs:
        start = time.monotonic()

        if pipeline_runner is not None:
            try:
                result = await pipeline_runner(brief)
            except Exception as e:
                result = RegressionResult(
                    concept_id=brief.get("concept_id", "unknown"),
                    category=brief.get("category", "unknown"),
                    title=brief.get("title", "unknown"),
                    success=False,
                    error=str(e),
                    duration_sec=time.monotonic() - start,
                )
        else:
            # Stub: validate the brief structure without running the full pipeline
            result = _stub_validate(brief)
            result.duration_sec = time.monotonic() - start

        report.results.append(result)

        if result.success:
            report.total_success += 1
        else:
            report.total_failed += 1

    # Calculate aggregates
    if report.total_concepts > 0:
        report.success_rate = report.total_success / report.total_concepts

    iteration_counts = [r.iterations for r in report.results if r.success]
    if iteration_counts:
        report.avg_iterations = sum(iteration_counts) / len(iteration_counts)

    costs = [r.cost_usd for r in report.results]
    report.total_cost_usd = sum(costs)
    if costs:
        report.avg_cost_usd = report.total_cost_usd / len(costs)

    report.total_duration_sec = sum(r.duration_sec for r in report.results)

    logger.info("Regression complete: %s", report.summarize())
    return report


def _stub_validate(brief: dict[str, Any]) -> RegressionResult:
    """Validate brief structure without running the pipeline."""
    concept_id = brief.get("concept_id", "unknown")
    category = brief.get("category", "unknown")
    title = brief.get("title", "unknown")

    required_fields = [
        "concept_id", "pipeline", "category", "title",
        "visual_brief", "simulation_requirements", "audio_brief",
        "hook_text", "novelty_score", "feasibility_score",
    ]

    missing = [f for f in required_fields if f not in brief]
    if missing:
        return RegressionResult(
            concept_id=concept_id,
            category=category,
            title=title,
            success=False,
            error=f"Missing fields: {missing}",
        )

    sim_reqs = brief.get("simulation_requirements", {})
    if not isinstance(sim_reqs, dict):
        return RegressionResult(
            concept_id=concept_id,
            category=category,
            title=title,
            success=False,
            error="simulation_requirements must be a dict",
        )

    return RegressionResult(
        concept_id=concept_id,
        category=category,
        title=title,
        success=True,
        iterations=1,
    )


def save_report(report: RegressionReport, output_dir: Path | None = None) -> Path:
    """Save a regression report to JSON.

    Args:
        report: The regression report.
        output_dir: Directory to write to. Defaults to tests/golden_set/reports/.

    Returns:
        Path to the saved report file.
    """
    if output_dir is None:
        output_dir = GOLDEN_SET_PATH.parent / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"regression_{report.run_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(report.summarize(), f, indent=2)

    logger.info("Regression report saved to %s", filepath)
    return filepath
