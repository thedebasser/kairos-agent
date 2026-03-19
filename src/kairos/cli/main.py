"""CLI entry point for Kairos Agent.

Supports:
    pipeline run       — Start a new pipeline run
    pipeline resume    — Resume a checkpointed pipeline run
    pipeline restart   — Restart a failed pipeline run
    pipeline status    — Show status of recent pipeline runs
    pipeline stats     — Show aggregate pipeline statistics
    pipeline cache     — Show or manage the response cache
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid


logger = logging.getLogger(__name__)


def cli() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Kairos Agent — Simulation Content Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # pipeline run
    run_parser = subparsers.add_parser("run", help="Start a new pipeline run")
    run_parser.add_argument(
        "--pipeline",
        type=str,
        default="physics",
        help="Pipeline to run (default: physics)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level console output",
    )

    # pipeline resume <id>
    resume_parser = subparsers.add_parser("resume", help="Resume a checkpointed run")
    resume_parser.add_argument("pipeline_run_id", type=str, help="Pipeline run ID to resume")
    resume_parser.add_argument("--verbose", "-v", action="store_true")

    # pipeline restart <id>
    restart_parser = subparsers.add_parser("restart", help="Restart a failed run")
    restart_parser.add_argument("pipeline_run_id", type=str, help="Pipeline run ID to restart")
    restart_parser.add_argument("--verbose", "-v", action="store_true")

    # pipeline status
    status_parser = subparsers.add_parser("status", help="Show recent pipeline runs")
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent runs to show",
    )

    # pipeline inspect <run_id>
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a run's step artifacts")
    inspect_parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default=None,
        help="Pipeline run ID to inspect (default: latest run)",
    )
    inspect_parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Show only a specific step number (e.g. --step 1)",
    )
    inspect_parser.add_argument(
        "--speed",
        type=float,
        default=0,
        help="Replay speed (0 = instant, 1 = real-time, 2 = 2x)",
    )

    # pipeline stats
    stats_parser = subparsers.add_parser("stats", help="Show aggregate pipeline statistics")
    stats_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to aggregate (default: 7)",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output stats as JSON",
    )

    # pipeline cache
    cache_parser = subparsers.add_parser("cache", help="Show or manage the response cache")
    cache_parser.add_argument(
        "--evict",
        action="store_true",
        help="Run LRU eviction to bring cache within size limit",
    )
    cache_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output stats as JSON",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Initialise logging for all commands except 'status'
    if args.command != "status":
        from kairos.ai.tracing.logging_config import init_logging
        console_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
        init_logging(console_level=console_level)

    if args.command == "run":
        asyncio.run(_run_pipeline(args.pipeline))
    elif args.command == "resume":
        asyncio.run(_resume_pipeline(uuid.UUID(args.pipeline_run_id)))
    elif args.command == "restart":
        asyncio.run(_restart_pipeline(uuid.UUID(args.pipeline_run_id)))
    elif args.command == "status":
        asyncio.run(_show_status(args.limit))
    elif args.command == "inspect":
        _inspect_run(args.run_id, args.step, speed=args.speed)
    elif args.command == "stats":
        asyncio.run(_show_stats(args.days, args.as_json))
    elif args.command == "cache":
        _show_cache(evict=args.evict, as_json=args.as_json)


async def _run_pipeline(pipeline_name: str) -> None:
    """Start a new pipeline run using LangGraph orchestrator."""
    from kairos.orchestrator.graph import run_pipeline

    logger.info("Starting pipeline: %s", pipeline_name)
    try:
        final_state = await run_pipeline(pipeline_name)
        status = final_state.get("status", "unknown")
        logger.info("Pipeline completed with status: %s", status)
        if final_state.get("errors"):
            for err in final_state["errors"]:
                logger.error("  → %s", err)
        if final_state.get("final_video_path"):
            logger.info("Output: %s", final_state["final_video_path"])
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)


async def _resume_pipeline(pipeline_run_id: uuid.UUID) -> None:
    """Resume a checkpointed pipeline run using LangGraph's PostgreSQL checkpointer."""
    from kairos.orchestrator.graph import resume_pipeline

    logger.info("Resuming pipeline run: %s", pipeline_run_id)
    try:
        final_state = await resume_pipeline(str(pipeline_run_id))
        status = final_state.get("status", "unknown")
        logger.info("Pipeline resumed and completed with status: %s", status)
        if final_state.get("errors"):
            for err in final_state["errors"]:
                logger.error("  → %s", err)
        if final_state.get("final_video_path"):
            logger.info("Output: %s", final_state["final_video_path"])
    except Exception as e:
        logger.exception("Pipeline resume failed")
        sys.exit(1)


async def _restart_pipeline(pipeline_run_id: uuid.UUID) -> None:
    """Restart a failed pipeline run with a fresh state.

    Looks up the original pipeline name from the run, then starts fresh.
    """
    from kairos.orchestrator.graph import run_pipeline

    logger.info("Restarting pipeline run: %s (new run)", pipeline_run_id)

    # Attempt to look up original pipeline name from DB
    pipeline_name = "physics"  # default fallback
    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import get_pipeline_run

        async with async_session_factory() as session:
            original_run = await get_pipeline_run(session, pipeline_run_id)
            if original_run:
                pipeline_name = original_run.pipeline
                logger.info("Restarting pipeline: %s", pipeline_name)
    except Exception:
        logger.warning("Could not look up original pipeline, defaulting to: %s", pipeline_name)

    try:
        final_state = await run_pipeline(pipeline_name)
        status = final_state.get("status", "unknown")
        logger.info("Restart completed with status: %s", status)
    except Exception as e:
        logger.exception("Pipeline restart failed")
        sys.exit(1)


async def _show_status(limit: int) -> None:
    """Show status of recent pipeline runs from the database."""
    try:
        from sqlalchemy import select, desc
        from kairos.db.session import async_session_factory
        from kairos.db.models import PipelineRun

        async with async_session_factory() as session:
            stmt = (
                select(PipelineRun)
                .order_by(desc(PipelineRun.started_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            runs = result.scalars().all()

            if not runs:
                print("No pipeline runs found.")
                return

            print(f"{'ID':>36}  {'Pipeline':<12}  {'Status':<16}  {'Started':<20}  {'Cost':>8}")
            print("-" * 100)
            for run in runs:
                run_id = str(run.pipeline_run_id)
                started = run.started_at.strftime("%Y-%m-%d %H:%M") if run.started_at else "N/A"
                cost = f"${run.total_cost_usd:.4f}" if run.total_cost_usd else "N/A"
                print(f"{run_id}  {run.pipeline:<12}  {run.status:<16}  {started:<20}  {cost:>8}")
    except Exception as e:
        print(f"Could not fetch pipeline status: {e}")
        print("Ensure PostgreSQL is running and database is initialised.")


async def _show_stats(days: int, as_json: bool = False) -> None:
    """Show aggregate pipeline statistics over a time window."""
    try:
        from kairos.tools.pipeline_stats import get_pipeline_stats
        import json

        stats = await get_pipeline_stats(days=days)

        if as_json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print(stats.summary())
    except Exception as e:
        print(f"Could not fetch pipeline stats: {e}")
        print("Ensure PostgreSQL is running and database is initialised.")


def _inspect_run(run_id: str | None, step_number: int | None, *, speed: float = 0) -> None:
    """Inspect a pipeline run using the new tracing layout.

    If no run_id given, shows the latest run.
    If --step N given, shows only that step's decisions/prompts.
    Otherwise, replays the run through the Rich display.
    """
    import json
    from pathlib import Path

    runs_dir = Path(__file__).resolve().parent.parent.parent.parent / "runs"

    if not runs_dir.exists():
        print("No runs directory found. Run a pipeline first.")
        sys.exit(1)

    if run_id is None:
        # Find the latest run by modification time
        run_dirs = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            print("No runs found.")
            sys.exit(1)
        run_dir = run_dirs[0]
        print(f"Latest run: {run_dir.name}\n")
    else:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            # Try partial match
            matches = [d for d in runs_dir.iterdir() if d.is_dir() and run_id in d.name]
            if len(matches) == 1:
                run_dir = matches[0]
            elif len(matches) > 1:
                print(f"Ambiguous run ID '{run_id}'. Matches:")
                for m in matches:
                    print(f"  {m.name}")
                sys.exit(1)
            else:
                print(f"Run '{run_id}' not found in {runs_dir}")
                sys.exit(1)

    # -- Step detail mode --------------------------------------------------
    if step_number is not None:
        steps_dir = run_dir / "steps"
        if steps_dir.exists():
            # New layout: steps/NN_step_name/
            step_dirs = sorted(
                d for d in steps_dir.iterdir()
                if d.is_dir() and d.name.startswith(f"{step_number:02d}_")
            )
            if step_dirs:
                step_dir = step_dirs[0]
                print(f"Step: {step_dir.name}\n")
                for f in sorted(step_dir.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(step_dir)
                        if f.suffix in (".json", ".jsonl"):
                            print(f"--- {rel} ---")
                            print(f.read_text(encoding="utf-8"))
                            print()
                        else:
                            size = f.stat().st_size
                            print(f"  {rel} ({size} bytes)")
                return

        # Fallback: old flat layout (NN_*.json in run_dir)
        step_files = sorted(run_dir.glob(f"{step_number:02d}_*.json"))
        if step_files:
            for sf in step_files:
                data = json.loads(sf.read_text(encoding="utf-8"))
                print(json.dumps(data, indent=2))
            return

        print(f"No artifact found for step {step_number} in {run_dir.name}")
        sys.exit(1)

    # -- Run overview / replay mode ----------------------------------------

    # If events.jsonl exists, use Rich replay
    events_file = run_dir / "events.jsonl"
    if events_file.exists():
        from kairos.cli.ui.run_display import RunDisplay

        display = RunDisplay()
        display.replay(run_dir, speed=speed)  # 0 = instant replay
        return

    # Fallback: show run_summary.json (old or new layout)
    summary_file = run_dir / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        print(f"Run:      {summary.get('pipeline_run_id', 'N/A')}")
        print(f"Pipeline: {summary.get('pipeline', 'N/A')}")
        print(f"Status:   {summary.get('final_status', summary.get('status', 'N/A'))}")
        print(f"Duration: {summary.get('total_duration_ms', 0)}ms")
        print(f"Concept:  {summary.get('concept_title', 'N/A')}")
        print(f"Video:    {summary.get('final_video_path', 'N/A')}")
        print(f"Cost:     ${summary.get('total_cost_usd', 0.0):.4f}")
        print()
        if summary.get("steps"):
            print("Steps:")
            for step in summary["steps"]:
                mark = "+" if step.get("status") == "success" else "x"
                print(f"  {mark} [{step.get('step_number', '?')}] {step.get('step', '?'):<25} {step.get('duration_ms', 0):>8}ms  ({step.get('status', '?')})")
        if summary.get("errors"):
            print("\nErrors:")
            for err in summary["errors"]:
                print(f"  - {err}")
    else:
        print(f"No run summary found.")

    # List all files
    print(f"\nArtifact files in {run_dir.name}/:")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(run_dir)
            size = f.stat().st_size
            print(f"  {str(rel):<50} {size:>8} bytes")
    print(f"\nTo view a step: pipeline inspect {run_dir.name} --step <N>")


def _show_cache(*, evict: bool = False, as_json: bool = False) -> None:
    """Show cache statistics and optionally run eviction."""
    from kairos.ai.llm.cache import ResponseCache

    stats = ResponseCache.cache_stats()

    if evict:
        # Create a temporary instance just for eviction
        cache = ResponseCache(run_id="__cli_evict__")
        deleted = cache.evict_if_needed()
        stats["evicted_files"] = deleted
        # Refresh stats after eviction
        stats = {**ResponseCache.cache_stats(), "evicted_files": deleted}

    if as_json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Global cache: {stats['total_size_mb']:.1f} MB  ({stats['file_count']} files)")
        print(f"  LLM responses:     {stats['llm_count']}")
        print(f"  Sandbox results:   {stats['sandbox_count']}")
        print(f"  Sandbox outputs:   {stats['sandbox_output_count']}")
        if evict:
            print(f"  Evicted:           {stats['evicted_files']} files")


if __name__ == "__main__":
    cli()
