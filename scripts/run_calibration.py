#!/usr/bin/env python
"""CLI runner for the Kairos calibration system.

Runs one or more calibration scenarios against Blender headless.
Use --dry-run to execute the full Blender loop without writing to ChromaDB.

Examples:
    # Validate a single scenario end-to-end without touching the vector DB
    python scripts/run_calibration.py --scenario straight_30 --dry-run

    # Run all Phase 1 bootstrap scenarios in dry-run mode
    python scripts/run_calibration.py --all --dry-run --max-iterations 5

    # Actually populate the knowledge base (only after dry-run validates it works)
    python scripts/run_calibration.py --scenario straight_30 --max-iterations 10
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the src package is importable when run from project root
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_calibration")


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run calibration scenarios against Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenario names and exit",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--scenario", "-s",
        metavar="NAME",
        help="Name of a bootstrap scenario (e.g. straight_30, s_curve_gentle)",
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all Phase 1 bootstrap scenarios",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Run Blender loop but DO NOT write anything to ChromaDB. "
            "Session artifacts are still written to output/calibration/sessions/. "
            "Use this to validate the pipeline before populating the vector DB."
        ),
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=10,
        metavar="N",
        help="Maximum iterations per scenario (default: 10)",
    )
    return parser.parse_args()


def _list_scenarios() -> None:
    from kairos.calibration.bootstrap import PHASE_1_SCENARIOS

    print("\nPhase 1 Bootstrap Scenarios:")
    print("-" * 48)
    for s in PHASE_1_SCENARIOS:
        print(f"  {s['name']:<20}  {s['description']}")
    print()


async def _run_single(
    name: str,
    *,
    dry_run: bool,
    max_iterations: int,
) -> dict[str, Any]:
    from kairos.calibration.bootstrap import PHASE_1_SCENARIOS
    from kairos.calibration.knowledge_base import KnowledgeBase
    from kairos.calibration.sandbox import run_calibration

    lookup = {s["name"]: s for s in PHASE_1_SCENARIOS}
    if name not in lookup:
        available = ", ".join(lookup.keys())
        logger.error("Unknown scenario '%s'. Available: %s", name, available)
        sys.exit(1)

    scenario_def = lookup[name]
    scenario = scenario_def["scenario"]

    logger.info("=" * 60)
    logger.info("Scenario : %s", name)
    logger.info("Description: %s", scenario_def["description"])
    if dry_run:
        logger.info("Mode     : DRY RUN (no ChromaDB writes)")
    else:
        logger.info("Mode     : LIVE (will write to ChromaDB on success)")
    logger.info("Max iters: %d", max_iterations)
    logger.info("=" * 60)

    kb = None if dry_run else KnowledgeBase()

    session = await run_calibration(
        scenario,
        knowledge_base=kb,
        max_iterations=max_iterations,
        dry_run=dry_run,
    )

    result = {
        "name": name,
        "status": session.status.value,
        "iterations": session.iteration_count,
        "confidence": session.confidence,
        "session_id": str(session.session_id),
    }

    _print_result(result, session)
    return result


def _print_result(result: dict[str, Any], session: Any) -> None:
    status = result["status"]
    marker = "✓" if status == "promoted" else ("~" if status == "resolved" else "✗")

    print(f"\n{marker} {result['name']} → {status.upper()}")
    print(f"  Session ID : {result['session_id']}")
    print(f"  Iterations : {result['iterations']}")
    print(f"  Confidence : {result['confidence']:.2f}")

    if session.final_corrections:
        corrections = session.final_corrections.model_dump()
        non_default = {k: v for k, v in corrections.items()
                       if isinstance(v, float) and abs(v - 1.0) > 0.001}
        if non_default:
            print("  Corrections:")
            for k, v in non_default.items():
                print(f"    {k:<30} {v:.4f}×")


async def _run_all(*, dry_run: bool, max_iterations: int) -> None:
    from kairos.calibration.bootstrap import PHASE_1_SCENARIOS, run_bootstrap

    if dry_run:
        # Run one at a time so we can see clear output per scenario
        results = []
        for s in PHASE_1_SCENARIOS:
            r = await _run_single(s["name"], dry_run=True, max_iterations=max_iterations)
            results.append(r)
    else:
        results = await run_bootstrap(max_iterations=max_iterations)

    # Summary table
    print("\n" + "=" * 60)
    print("BOOTSTRAP SUMMARY")
    print("=" * 60)
    for r in results:
        status_col = r["status"].upper().ljust(12)
        iters = str(r.get("iterations", "?")).rjust(3)
        conf = f"{r.get('confidence', 0):.2f}"
        print(f"  {r['name']:<20}  {status_col}  iters={iters}  conf={conf}")

    promoted = sum(1 for r in results if r["status"] == "promoted")
    resolved = sum(1 for r in results if r["status"] == "resolved")
    unresolved = sum(1 for r in results if r["status"] == "unresolved")
    errors = sum(1 for r in results if r.get("status") == "error")
    total = len(results)

    print(f"\nTotal: {total}  |  Promoted: {promoted}  |  Resolved: {resolved}"
          f"  |  Unresolved: {unresolved}  |  Errors: {errors}")

    if dry_run and (promoted > 0 or resolved > 0):
        print("\nDRY RUN complete — nothing was written to ChromaDB.")
        print("Re-run without --dry-run to populate the knowledge base.")


async def main() -> None:
    args = _parse_args()

    if args.list:
        _list_scenarios()
        return

    if not args.scenario and not args.all:
        import argparse
        raise SystemExit("error: one of --scenario, --all, or --list is required")

    if args.all:
        await _run_all(dry_run=args.dry_run, max_iterations=args.max_iterations)
    else:
        await _run_single(
            args.scenario,
            dry_run=args.dry_run,
            max_iterations=args.max_iterations,
        )


if __name__ == "__main__":
    asyncio.run(main())
