"""Kairos Agent — Eval Harness Runner.

Loads curated test cases from ``eval_cases.yaml``, runs each through the
pipeline (idea → simulation → validation), and reports pass/fail results.

Usage::

    # Run all eval cases
    python -m kairos.eval.harness

    # Run a specific case or category
    python -m kairos.eval.harness --case bp-overflow-01
    python -m kairos.eval.harness --category ball_pit

    # Dry-run (just load and validate cases, no execution)
    python -m kairos.eval.harness --dry-run

Finding 6.2: *"No Eval Dataset or Regression Testing Framework"*
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from kairos.models.contracts import (
    AudioBrief,
    ConceptBrief,
    EnergyLevel,
    ScenarioCategory,
    SimulationRequirements,
)

logger = logging.getLogger(__name__)

_EVAL_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.yaml"
_EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "runs" / "_eval"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExpectedOutcome:
    """What we expect from running a test case."""

    validation_passes: bool = True
    min_checks_passed: int = 4
    must_pass_checks: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    """A single curated test case for the eval harness."""

    id: str
    category: str
    title: str
    visual_brief: str
    hook_text: str
    body_count_initial: int
    body_count_max: int
    interaction_type: str
    colour_palette: list[str]
    background_colour: str = "#1a1a2e"
    special_effects: list[str] = field(default_factory=list)
    seed: int = 42
    expected: ExpectedOutcome = field(default_factory=ExpectedOutcome)

    def to_concept_brief(self) -> ConceptBrief:
        """Convert to a ``ConceptBrief`` model."""
        return ConceptBrief(
            concept_id=uuid4(),
            pipeline="physics",
            category=ScenarioCategory(self.category),
            title=self.title,
            visual_brief=self.visual_brief,
            simulation_requirements=SimulationRequirements(
                body_count_initial=self.body_count_initial,
                body_count_max=self.body_count_max,
                interaction_type=self.interaction_type,
                colour_palette=self.colour_palette,
                background_colour=self.background_colour,
                special_effects=self.special_effects,
            ),
            audio_brief=AudioBrief(
                mood=["eval"],
                tempo_bpm_min=100,
                tempo_bpm_max=130,
                energy_curve=EnergyLevel.BUILDING,
            ),
            hook_text=self.hook_text,
            novelty_score=5.0,
            feasibility_score=5.0,
            target_duration_sec=65,
            seed=self.seed,
        )


@dataclass
class EvalResult:
    """Outcome of running a single eval case."""

    case_id: str
    passed: bool
    duration_sec: float = 0.0
    checks_passed: int = 0
    checks_total: int = 0
    missing_checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    iteration_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "duration_sec": round(self.duration_sec, 2),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "missing_checks": self.missing_checks,
            "errors": self.errors,
            "cost_usd": round(self.cost_usd, 4),
            "iteration_count": self.iteration_count,
        }


@dataclass
class EvalSuiteResult:
    """Aggregate results across all eval cases."""

    run_id: str
    timestamp: str
    results: list[EvalResult]
    total_duration_sec: float = 0.0

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.pass_count / len(self.results)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cases": len(self.results),
            "passed": self.pass_count,
            "failed": self.fail_count,
            "pass_rate": round(self.pass_rate * 100, 1),
            "total_duration_sec": round(self.total_duration_sec, 2),
            "total_cost_usd": round(self.total_cost, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Eval Run: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Cases: {len(self.results)} | "
            f"Passed: {self.pass_count} | "
            f"Failed: {self.fail_count} | "
            f"Pass Rate: {self.pass_rate * 100:.1f}%",
            f"Total Duration: {self.total_duration_sec:.1f}s | "
            f"Total Cost: ${self.total_cost:.4f}",
            "",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            detail = ""
            if r.errors:
                detail = f" — {r.errors[0][:80]}"
            elif r.missing_checks:
                detail = f" — missing: {', '.join(r.missing_checks)}"
            lines.append(
                f"  [{status}] {r.case_id}: "
                f"{r.checks_passed}/{r.checks_total} checks, "
                f"{r.duration_sec:.1f}s, ${r.cost_usd:.4f}"
                f"{detail}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eval_cases(
    path: Path = _EVAL_CASES_PATH,
    *,
    case_id: str | None = None,
    category: str | None = None,
) -> list[EvalCase]:
    """Load eval cases from YAML, optionally filtering by id or category."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases: list[EvalCase] = []
    for entry in data.get("cases", []):
        expected_raw = entry.pop("expected", {})
        expected = ExpectedOutcome(
            validation_passes=expected_raw.get("validation_passes", True),
            min_checks_passed=expected_raw.get("min_checks_passed", 4),
            must_pass_checks=expected_raw.get("must_pass_checks", []),
        )
        cases.append(EvalCase(**entry, expected=expected))

    if case_id:
        cases = [c for c in cases if c.id == case_id]
    if category:
        cases = [c for c in cases if c.category == category]

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_eval_case(case: EvalCase) -> EvalResult:
    """Run a single eval case through simulation → validation.

    This runs the simulation agent and validation only (no publishing).
    """
    from kairos.config import get_settings
    from kairos.services import validation

    concept = case.to_concept_brief()
    result = EvalResult(case_id=case.id, passed=False)
    start = time.monotonic()

    try:
        # Import simulation agent
        from kairos.pipelines.physics.simulation_agent import PhysicsSimulationAgent

        settings = get_settings()
        agent = PhysicsSimulationAgent(settings)

        # Run the simulation loop — narrow DTO, no PipelineState (Finding 2.2)
        loop_result = await agent.run_loop(concept)
        result.iteration_count = loop_result.simulation_iteration

        # Check if simulation produced a video
        video_path = loop_result.raw_video_path
        if not video_path:
            result.errors.append("No video produced by simulation")
            result.duration_sec = time.monotonic() - start
            return result

        # Run validation
        val_result = await validation.validate_simulation(video_path)
        result.checks_total = len(val_result.checks)
        result.checks_passed = sum(1 for c in val_result.checks if c.passed)

        # Check expected outcomes
        missing = []
        passed_check_names = {c.name for c in val_result.checks if c.passed}
        for required in case.expected.must_pass_checks:
            if required not in passed_check_names:
                missing.append(required)

        result.missing_checks = missing
        result.passed = (
            len(missing) == 0
            and result.checks_passed >= case.expected.min_checks_passed
        )

        if not result.passed and not missing:
            result.errors.append(
                f"Only {result.checks_passed}/{case.expected.min_checks_passed} "
                f"checks passed (need {case.expected.min_checks_passed})"
            )

    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
        logger.exception("Eval case %s failed", case.id)

    result.duration_sec = time.monotonic() - start
    return result


async def run_eval_suite(
    cases: list[EvalCase],
    *,
    concurrency: int = 1,
) -> EvalSuiteResult:
    """Run all eval cases and aggregate results."""
    run_id = str(uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[EvalResult] = []
    start = time.monotonic()

    logger.info("Starting eval run %s with %d cases", run_id, len(cases))

    if concurrency <= 1:
        for case in cases:
            logger.info("Running eval case: %s (%s)", case.id, case.category)
            result = await run_eval_case(case)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info(
                "  [%s] %s — %d/%d checks, %.1fs",
                status,
                case.id,
                result.checks_passed,
                result.checks_total,
                result.duration_sec,
            )
    else:
        # Run in batches of `concurrency
        for i in range(0, len(cases), concurrency):
            batch = cases[i : i + concurrency]
            batch_results = await asyncio.gather(
                *(run_eval_case(c) for c in batch),
                return_exceptions=True,
            )
            for case, br in zip(batch, batch_results):
                if isinstance(br, Exception):
                    er = EvalResult(
                        case_id=case.id,
                        passed=False,
                        errors=[f"{type(br).__name__}: {br}"],
                    )
                    results.append(er)
                else:
                    results.append(br)

    suite = EvalSuiteResult(
        run_id=run_id,
        timestamp=timestamp,
        results=results,
        total_duration_sec=time.monotonic() - start,
    )

    # Persist results
    _EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _EVAL_RESULTS_DIR / f"eval_{run_id}_{timestamp[:10]}.json"
    result_path.write_text(
        json.dumps(suite.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("Eval results saved to %s", result_path)

    return suite


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kairos Eval Harness")
    parser.add_argument("--case", help="Run a specific eval case by ID")
    parser.add_argument("--category", help="Run all cases in a category")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and display cases without executing",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of eval cases to run in parallel",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    cases = load_eval_cases(case_id=args.case, category=args.category)
    if not cases:
        print("No eval cases matched the filter.")
        sys.exit(1)

    print(f"Loaded {len(cases)} eval case(s):")
    for c in cases:
        print(f"  {c.id}: [{c.category}] {c.title}")

    if args.dry_run:
        print("\nDry run — no execution performed.")
        for c in cases:
            concept = c.to_concept_brief()
            print(f"\n  {c.id}:")
            print(f"    Category: {concept.category.value}")
            print(f"    Bodies: {c.body_count_initial}–{c.body_count_max}")
            print(f"    Expected: ≥{c.expected.min_checks_passed} checks passed")
            print(f"    Must pass: {', '.join(c.expected.must_pass_checks) or '(none)'}")
        return

    suite = asyncio.run(run_eval_suite(cases, concurrency=args.concurrency))

    print("\n" + "=" * 60)
    print(suite.summary())
    print("=" * 60)

    sys.exit(0 if suite.fail_count == 0 else 1)


if __name__ == "__main__":
    main()
