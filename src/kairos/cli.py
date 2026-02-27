"""CLI entry point for Kairos Agent.

Supports:
    pipeline run       — Start a new pipeline run
    pipeline resume    — Resume a checkpointed pipeline run
    pipeline restart   — Restart a failed pipeline run
    pipeline status    — Show status of recent pipeline runs
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Initialise logging for all commands except 'status'
    if args.command != "status":
        from kairos.services.session_logging import init_logging
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


async def _run_pipeline(pipeline_name: str) -> None:
    """Start a new pipeline run using LangGraph orchestrator."""
    from kairos.pipeline.graph import run_pipeline

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
    from kairos.pipeline.graph import resume_pipeline

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
    from kairos.pipeline.graph import run_pipeline

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


if __name__ == "__main__":
    cli()
