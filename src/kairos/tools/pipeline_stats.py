"""Kairos Agent — Pipeline Stats Aggregator.

Provides ``get_pipeline_stats()`` which queries the ``PipelineRun`` and
``AgentRun`` tables to produce success-rate, cost, duration, and per-step
failure-rate metrics.

Finding 6.3: *"No Pipeline Success Rate Tracking Over Time"*

Usage::

    # From CLI:
    pipeline stats
    pipeline stats --days 30

    # Programmatically:
    from kairos.tools.pipeline_stats import get_pipeline_stats
    stats = await get_pipeline_stats(days=7)
    print(stats.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, case, and_

from kairos.db.models import AgentRun, PipelineRun
from kairos.db.session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepFailureRate:
    """Failure rate for a specific pipeline step."""

    step_name: str
    total: int
    failed: int

    @property
    def rate(self) -> float:
        return self.failed / self.total if self.total else 0.0


@dataclass
class PipelineStats:
    """Aggregated pipeline statistics for a time period."""

    period_days: int
    total_runs: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0

    avg_duration_sec: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    # Per-step breakdown
    step_failures: list[StepFailureRate] = field(default_factory=list)

    # Per-model token usage
    model_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Trend: this period vs previous period
    prev_total_runs: int = 0
    prev_completed: int = 0

    @property
    def success_rate(self) -> float:
        if not self.total_runs:
            return 0.0
        return self.completed / self.total_runs

    @property
    def prev_success_rate(self) -> float:
        if not self.prev_total_runs:
            return 0.0
        return self.prev_completed / self.prev_total_runs

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Pipeline Stats (last {self.period_days} days)",
            "=" * 50,
            f"Total Runs:    {self.total_runs}",
            f"  Completed:   {self.completed}",
            f"  Failed:      {self.failed}",
            f"  Running:     {self.running}",
            f"Success Rate:  {self.success_rate * 100:.1f}%",
        ]

        if self.prev_total_runs:
            delta = self.success_rate - self.prev_success_rate
            arrow = "↑" if delta >= 0 else "↓"
            lines.append(
                f"  vs previous: {arrow} {abs(delta) * 100:.1f}pp "
                f"({self.prev_success_rate * 100:.1f}% → {self.success_rate * 100:.1f}%)"
            )

        lines.extend([
            "",
            f"Avg Duration:  {self.avg_duration_sec:.0f}s",
            f"Avg Cost:      ${self.avg_cost_usd:.4f}",
            f"Total Cost:    ${self.total_cost_usd:.4f}",
        ])

        if self.step_failures:
            lines.append("")
            lines.append("Per-Step Failure Rates:")
            for sf in sorted(self.step_failures, key=lambda x: x.rate, reverse=True):
                lines.append(
                    f"  {sf.step_name:30s} {sf.failed:3d}/{sf.total:3d} "
                    f"({sf.rate * 100:.1f}%)"
                )

        if self.model_usage:
            lines.append("")
            lines.append("Model Usage:")
            for model, usage in sorted(self.model_usage.items()):
                lines.append(
                    f"  {model:30s} {usage['calls']:4d} calls, "
                    f"${usage['cost']:.4f}, "
                    f"p50={usage.get('p50_ms', 0):.0f}ms"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_days": self.period_days,
            "total_runs": self.total_runs,
            "completed": self.completed,
            "failed": self.failed,
            "running": self.running,
            "success_rate": round(self.success_rate * 100, 1),
            "avg_duration_sec": round(self.avg_duration_sec, 1),
            "avg_cost_usd": round(self.avg_cost_usd, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "step_failures": [
                {"step": sf.step_name, "total": sf.total, "failed": sf.failed, "rate": round(sf.rate * 100, 1)}
                for sf in self.step_failures
            ],
            "model_usage": self.model_usage,
            "prev_success_rate": round(self.prev_success_rate * 100, 1),
        }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

async def get_pipeline_stats(days: int = 7) -> PipelineStats:
    """Query the database for aggregate pipeline statistics.

    Args:
        days: Number of days to look back (default 7).

    Returns:
        ``PipelineStats`` with aggregated metrics.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    prev_cutoff = cutoff - timedelta(days=days)

    stats = PipelineStats(period_days=days)

    async with get_session() as session:
        # ── Current period aggregates ────────────────────────────────
        q = select(
            func.count(PipelineRun.pipeline_run_id).label("total"),
            func.count(
                case(
                    (PipelineRun.status == "completed", PipelineRun.pipeline_run_id),
                    else_=None,
                )
            ).label("completed"),
            func.count(
                case(
                    (PipelineRun.status == "failed", PipelineRun.pipeline_run_id),
                    else_=None,
                )
            ).label("failed"),
            func.count(
                case(
                    (PipelineRun.status == "running", PipelineRun.pipeline_run_id),
                    else_=None,
                )
            ).label("running"),
            func.avg(PipelineRun.total_duration_sec).label("avg_duration"),
            func.avg(PipelineRun.total_cost_usd).label("avg_cost"),
            func.sum(PipelineRun.total_cost_usd).label("total_cost"),
        ).where(PipelineRun.started_at >= cutoff)

        row = (await session.execute(q)).one()
        stats.total_runs = row.total or 0
        stats.completed = row.completed or 0
        stats.failed = row.failed or 0
        stats.running = row.running or 0
        stats.avg_duration_sec = float(row.avg_duration or 0)
        stats.avg_cost_usd = float(row.avg_cost or 0)
        stats.total_cost_usd = float(row.total_cost or 0)

        # ── Previous period (for trend) ──────────────────────────────
        prev_q = select(
            func.count(PipelineRun.pipeline_run_id).label("total"),
            func.count(
                case(
                    (PipelineRun.status == "completed", PipelineRun.pipeline_run_id),
                    else_=None,
                )
            ).label("completed"),
        ).where(
            and_(
                PipelineRun.started_at >= prev_cutoff,
                PipelineRun.started_at < cutoff,
            )
        )

        prev_row = (await session.execute(prev_q)).one()
        stats.prev_total_runs = prev_row.total or 0
        stats.prev_completed = prev_row.completed or 0

        # ── Per-step failure rates ───────────────────────────────────
        step_q = (
            select(
                AgentRun.step_name,
                func.count(AgentRun.run_id).label("total"),
                func.count(
                    case(
                        (AgentRun.status == "failed", AgentRun.run_id),
                        else_=None,
                    )
                ).label("failed"),
            )
            .where(AgentRun.created_at >= cutoff)
            .where(AgentRun.step_name.isnot(None))
            .group_by(AgentRun.step_name)
        )

        step_rows = (await session.execute(step_q)).all()
        stats.step_failures = [
            StepFailureRate(
                step_name=r.step_name,
                total=r.total,
                failed=r.failed,
            )
            for r in step_rows
        ]

        # ── Per-model usage ──────────────────────────────────────────
        model_q = (
            select(
                AgentRun.model_used,
                func.count(AgentRun.run_id).label("calls"),
                func.sum(AgentRun.cost_usd).label("cost"),
                func.percentile_cont(0.5).within_group(
                    AgentRun.latency_ms
                ).label("p50_ms"),
            )
            .where(AgentRun.created_at >= cutoff)
            .where(AgentRun.model_used.isnot(None))
            .group_by(AgentRun.model_used)
        )

        try:
            model_rows = (await session.execute(model_q)).all()
            stats.model_usage = {
                r.model_used: {
                    "calls": r.calls,
                    "cost": float(r.cost or 0),
                    "p50_ms": float(r.p50_ms or 0),
                }
                for r in model_rows
            }
        except Exception:
            # percentile_cont may not be available on all DB backends
            logger.debug("percentile_cont not available, skipping model p50")
            model_q_simple = (
                select(
                    AgentRun.model_used,
                    func.count(AgentRun.run_id).label("calls"),
                    func.sum(AgentRun.cost_usd).label("cost"),
                    func.avg(AgentRun.latency_ms).label("avg_ms"),
                )
                .where(AgentRun.created_at >= cutoff)
                .where(AgentRun.model_used.isnot(None))
                .group_by(AgentRun.model_used)
            )
            model_rows = (await session.execute(model_q_simple)).all()
            stats.model_usage = {
                r.model_used: {
                    "calls": r.calls,
                    "cost": float(r.cost or 0),
                    "p50_ms": float(r.avg_ms or 0),
                }
                for r in model_rows
            }

    return stats
