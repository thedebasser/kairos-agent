"""Kairos Agent — Database CRUD Operations.

Async database access layer for all tables. Each function takes an
AsyncSession and returns typed results.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from kairos.db.models import (
    AgentRun,
    CategoryStat,
    Output,
    PipelineConfig,
    PipelineRun,
    PublishLog,
    PublishQueue,
    Simulation,
    TrainingExample,
    VideoIdea,
)


# =============================================================================
# Pipeline Runs
# =============================================================================


async def create_pipeline_run(
    session: AsyncSession,
    *,
    pipeline: str,
    pipeline_run_id: UUID | None = None,
) -> PipelineRun:
    """Create a new pipeline run record."""
    run = PipelineRun(pipeline=pipeline)
    if pipeline_run_id:
        run.pipeline_run_id = pipeline_run_id
    session.add(run)
    await session.flush()
    return run


async def get_pipeline_run(session: AsyncSession, pipeline_run_id: UUID) -> PipelineRun | None:
    """Get a pipeline run by ID."""
    return await session.get(PipelineRun, pipeline_run_id)


async def update_pipeline_run_status(
    session: AsyncSession,
    pipeline_run_id: UUID,
    *,
    status: str,
    **kwargs: object,
) -> None:
    """Update pipeline run status and optional fields."""
    values: dict[str, object] = {"status": status, **kwargs}
    stmt = (
        update(PipelineRun)
        .where(PipelineRun.pipeline_run_id == pipeline_run_id)
        .values(**values)
    )
    await session.execute(stmt)


# =============================================================================
# Video Ideas
# =============================================================================


async def create_video_idea(
    session: AsyncSession,
    *,
    pipeline_run_id: UUID,
    pipeline: str,
    concept: dict,  # type: ignore[type-arg]
    category: str | None = None,
    novelty_score: float | None = None,
    feasibility_score: float | None = None,
) -> VideoIdea:
    """Create a new video idea record."""
    idea = VideoIdea(
        pipeline_run_id=pipeline_run_id,
        pipeline=pipeline,
        concept=concept,
        category=category,
        novelty_score=novelty_score,
        feasibility_score=feasibility_score,
    )
    session.add(idea)
    await session.flush()
    return idea


async def get_recent_ideas(
    session: AsyncSession,
    pipeline: str,
    *,
    limit: int = 30,
) -> list[VideoIdea]:
    """Get recent ideas for category rotation analysis."""
    stmt = (
        select(VideoIdea)
        .where(VideoIdea.pipeline == pipeline)
        .order_by(VideoIdea.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_last_idea_category(session: AsyncSession, pipeline: str) -> str | None:
    """Get the category of the most recent idea (for hard block)."""
    stmt = (
        select(VideoIdea.category)
        .where(VideoIdea.pipeline == pipeline)
        .order_by(VideoIdea.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# =============================================================================
# Category Stats
# =============================================================================


async def get_category_stats(
    session: AsyncSession,
    pipeline: str,
) -> list[CategoryStat]:
    """Get all category stats for a pipeline."""
    stmt = select(CategoryStat).where(CategoryStat.pipeline == pipeline)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_category_stat(
    session: AsyncSession,
    *,
    pipeline: str,
    category: str,
    total_count: int = 0,
    streak_count: int = 0,
    videos_last_30_days: int = 0,
    last_used_at: datetime | None = None,
) -> CategoryStat:
    """Create or update a category stat record."""
    existing = await session.get(CategoryStat, (pipeline, category))
    if existing:
        existing.total_count = total_count
        existing.streak_count = streak_count
        existing.videos_last_30_days = videos_last_30_days
        if last_used_at:
            existing.last_used_at = last_used_at
        return existing

    stat = CategoryStat(
        pipeline=pipeline,
        category=category,
        total_count=total_count,
        streak_count=streak_count,
        videos_last_30_days=videos_last_30_days,
        last_used_at=last_used_at,
    )
    session.add(stat)
    await session.flush()
    return stat


# =============================================================================
# Simulations
# =============================================================================


async def create_simulation(
    session: AsyncSession,
    *,
    idea_id: UUID,
    pipeline_run_id: UUID,
    pipeline: str,
) -> Simulation:
    """Create a new simulation record."""
    sim = Simulation(
        idea_id=idea_id,
        pipeline_run_id=pipeline_run_id,
        pipeline=pipeline,
    )
    session.add(sim)
    await session.flush()
    return sim


async def get_simulation(session: AsyncSession, simulation_id: UUID) -> Simulation | None:
    """Get a simulation by ID."""
    return await session.get(Simulation, simulation_id)


async def update_simulation(
    session: AsyncSession,
    simulation_id: UUID,
    **kwargs: object,
) -> None:
    """Update simulation fields."""
    stmt = (
        update(Simulation)
        .where(Simulation.simulation_id == simulation_id)
        .values(**kwargs)
    )
    await session.execute(stmt)


# =============================================================================
# Outputs
# =============================================================================


async def create_output(
    session: AsyncSession,
    *,
    simulation_id: UUID,
    pipeline_run_id: UUID,
    **kwargs: object,
) -> Output:
    """Create a new output record."""
    output = Output(
        simulation_id=simulation_id,
        pipeline_run_id=pipeline_run_id,
        **kwargs,  # type: ignore[arg-type]
    )
    session.add(output)
    await session.flush()
    return output


async def get_pending_review_outputs(session: AsyncSession) -> list[Output]:
    """Get all outputs pending human review."""
    stmt = (
        select(Output)
        .where(Output.status == "pending_review")
        .order_by(Output.created_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_output_review(
    session: AsyncSession,
    output_id: UUID,
    *,
    review_action: str,
    review_feedback: str = "",
) -> None:
    """Update output with review decision."""
    status = "approved" if review_action == "approved" else "rejected"
    stmt = (
        update(Output)
        .where(Output.output_id == output_id)
        .values(
            status=status,
            review_action=review_action,
            review_feedback=review_feedback,
            reviewed_at=datetime.now(),
        )
    )
    await session.execute(stmt)


# =============================================================================
# Publish Queue
# =============================================================================


async def enqueue_for_publishing(
    session: AsyncSession,
    *,
    output_id: UUID,
    platform: str,
    account: str | None = None,
    scheduled_for: datetime | None = None,
    platform_title: str | None = None,
    platform_description: str | None = None,
    platform_tags: dict | None = None,  # type: ignore[type-arg]
) -> PublishQueue:
    """Add a video to the publish queue."""
    entry = PublishQueue(
        output_id=output_id,
        platform=platform,
        account=account,
        scheduled_for=scheduled_for,
        platform_title=platform_title,
        platform_description=platform_description,
        platform_tags=platform_tags,
    )
    session.add(entry)
    await session.flush()
    return entry


async def get_queued_for_publishing(
    session: AsyncSession,
    *,
    platform: str | None = None,
) -> list[PublishQueue]:
    """Get items ready to publish."""
    stmt = select(PublishQueue).where(PublishQueue.status == "queued")
    if platform:
        stmt = stmt.where(PublishQueue.platform == platform)
    stmt = stmt.order_by(PublishQueue.scheduled_for.asc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_publish_queue_status(
    session: AsyncSession,
    queue_id: UUID,
    *,
    status: str,
    attempts: int | None = None,
    last_error: str | None = None,
) -> PublishQueue | None:
    """Update the status of a publish queue item.

    Args:
        session: Database session.
        queue_id: Queue entry ID.
        status: New status ('publishing', 'published', 'failed', 'cancelled').
        attempts: Updated attempt count.
        last_error: Error message from last attempt.
    """
    entry = await session.get(PublishQueue, queue_id)
    if entry is None:
        return None
    entry.status = status
    if attempts is not None:
        entry.attempts = attempts
    if last_error is not None:
        entry.last_error = last_error
    return entry


# =============================================================================
# Agent Runs
# =============================================================================


async def log_agent_run(
    session: AsyncSession,
    *,
    pipeline_run_id: UUID,
    agent_name: str,
    step_name: str,
    model_used: str,
    status: str = "success",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    error_message: str = "",
    input_summary: dict | None = None,  # type: ignore[type-arg]
    output_summary: dict | None = None,  # type: ignore[type-arg]
    idea_id: UUID | None = None,
) -> AgentRun:
    """Log an agent execution."""
    run = AgentRun(
        pipeline_run_id=pipeline_run_id,
        idea_id=idea_id,
        agent_name=agent_name,
        step_name=step_name,
        model_used=model_used,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        error_message=error_message,
        input_summary=input_summary,
        output_summary=output_summary,
    )
    session.add(run)
    await session.flush()
    return run


async def get_rolling_cost_average(
    session: AsyncSession,
    *,
    days: int = 7,
) -> float:
    """Get rolling average cost per pipeline run over N days."""
    cutoff = datetime.now() - timedelta(days=days)
    stmt = (
        select(PipelineRun.total_cost_usd)
        .where(PipelineRun.started_at >= cutoff)
        .where(PipelineRun.total_cost_usd.isnot(None))
    )
    result = await session.execute(stmt)
    costs = [row[0] for row in result.all() if row[0] is not None]
    if not costs:
        return 0.0
    return sum(float(c) for c in costs) / len(costs)


# =============================================================================
# Training Examples
# =============================================================================


async def create_training_example(
    session: AsyncSession,
    *,
    simulation_id: UUID,
    pipeline: str,
    concept_brief: dict,  # type: ignore[type-arg]
    simulation_code: str,
    validation_passed: bool,
    human_approved: bool,
    rejection_reason: str | None = None,
) -> TrainingExample:
    """Create a training example for fine-tuning."""
    example = TrainingExample(
        simulation_id=simulation_id,
        pipeline=pipeline,
        concept_brief=concept_brief,
        simulation_code=simulation_code,
        validation_passed=validation_passed,
        human_approved=human_approved,
        rejection_reason=rejection_reason,
    )
    session.add(example)
    await session.flush()
    return example


async def get_approved_training_examples(
    session: AsyncSession,
    pipeline: str,
) -> list[TrainingExample]:
    """Get all approved training examples for fine-tuning."""
    stmt = (
        select(TrainingExample)
        .where(TrainingExample.pipeline == pipeline)
        .where(TrainingExample.human_approved.is_(True))
        .where(TrainingExample.validation_passed.is_(True))
        .order_by(TrainingExample.created_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# =============================================================================
# Pipeline Config
# =============================================================================


async def get_pipeline_config(
    session: AsyncSession,
    pipeline: str,
) -> PipelineConfig | None:
    """Get pipeline configuration."""
    return await session.get(PipelineConfig, pipeline)


async def upsert_pipeline_config(
    session: AsyncSession,
    *,
    pipeline: str,
    engine: str,
    categories: list[str],
    **kwargs: object,
) -> PipelineConfig:
    """Create or update pipeline configuration."""
    existing = await session.get(PipelineConfig, pipeline)
    if existing:
        existing.engine = engine
        existing.categories = categories  # type: ignore[assignment]
        for key, value in kwargs.items():
            setattr(existing, key, value)
        return existing

    config = PipelineConfig(
        pipeline=pipeline,
        engine=engine,
        categories=categories,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    session.add(config)
    await session.flush()
    return config


# =============================================================================
# Publish Log
# =============================================================================


async def create_publish_log(
    session: AsyncSession,
    *,
    output_id: UUID,
    queue_id: UUID,
    platform: str,
    account: str | None = None,
    platform_video_id: str | None = None,
    published_at: datetime | None = None,
) -> PublishLog:
    """Record a successful publish."""
    log = PublishLog(
        output_id=output_id,
        queue_id=queue_id,
        platform=platform,
        account=account,
        platform_video_id=platform_video_id,
        published_at=published_at or datetime.now(),
    )
    session.add(log)
    await session.flush()
    return log


async def update_publish_log_metrics(
    session: AsyncSession,
    publish_id: UUID,
    *,
    views_7d: int | None = None,
    views_30d: int | None = None,
    likes: int | None = None,
    comments: int | None = None,
    shares: int | None = None,
    avg_view_duration_sec: float | None = None,
    retention_rate: float | None = None,
    revenue_usd: float | None = None,
) -> PublishLog | None:
    """Update engagement metrics on a publish log entry.

    Called by the analytics sync job to write back platform metrics.
    """
    log = await session.get(PublishLog, publish_id)
    if log is None:
        return None
    if views_7d is not None:
        log.views_7d = views_7d
    if views_30d is not None:
        log.views_30d = views_30d
    if likes is not None:
        log.likes = likes
    if comments is not None:
        log.comments = comments
    if shares is not None:
        log.shares = shares
    if avg_view_duration_sec is not None:
        log.avg_view_duration_sec = avg_view_duration_sec
    if retention_rate is not None:
        log.retention_rate = retention_rate
    if revenue_usd is not None:
        log.revenue_usd = revenue_usd
    return log


async def update_category_stat_performance(
    session: AsyncSession,
    *,
    pipeline: str,
    category: str,
    avg_performance: float,
    videos_last_30_days: int | None = None,
) -> CategoryStat | None:
    """Update category performance metrics from analytics data.

    Called by the analytics sync to close the feedback loop to the Idea Agent.
    """
    from sqlalchemy import and_

    stmt = select(CategoryStat).where(
        and_(
            CategoryStat.pipeline == pipeline,
            CategoryStat.category == category,
        )
    )
    result = await session.execute(stmt)
    stat = result.scalar_one_or_none()
    if stat is None:
        return None
    stat.avg_performance = avg_performance
    if videos_last_30_days is not None:
        stat.videos_last_30_days = videos_last_30_days
    return stat
