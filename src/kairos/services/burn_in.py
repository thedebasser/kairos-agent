"""Kairos Agent — Production Burn-In Configuration & Runner.

Runs the full pipeline at target throughput (2 videos/day) for a
burn-in period (default 2 weeks). Monitors all metrics: cost,
success rate, review approval, queue depth, model latency.

This module provides:
- Burn-in schedule configuration
- Metrics tracking across the burn-in period
- Daily summary reports
- Prompt tuning feedback collection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kairos.config import get_settings
from kairos.services.monitoring import AlertManager, MetricsStore, get_metrics_store

logger = logging.getLogger(__name__)


# =============================================================================
# Burn-In Configuration
# =============================================================================


@dataclass
class BurnInConfig:
    """Configuration for a production burn-in run."""

    duration_days: int = 14
    videos_per_day: int = 2
    pipeline: str = "physics"
    auto_publish: bool = False  # Require human review during burn-in
    alert_on_failure: bool = True
    collect_training_examples: bool = True
    min_training_examples: int = 20


@dataclass
class DailySummary:
    """Summary of one day's burn-in results."""

    date: str
    videos_produced: int = 0
    videos_approved: int = 0
    videos_rejected: int = 0
    total_cost_usd: float = 0.0
    avg_iterations: float = 0.0
    avg_latency_ms: float = 0.0
    failure_modes: list[str] = field(default_factory=list)


@dataclass
class BurnInReport:
    """Summary report for the entire burn-in period."""

    config: BurnInConfig = field(default_factory=BurnInConfig)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime | None = None
    daily_summaries: list[DailySummary] = field(default_factory=list)
    total_videos_produced: int = 0
    total_videos_approved: int = 0
    total_videos_rejected: int = 0
    total_cost_usd: float = 0.0
    overall_success_rate: float = 0.0
    overall_approval_rate: float = 0.0
    training_examples_collected: int = 0
    common_failure_modes: list[str] = field(default_factory=list)
    prompt_tuning_notes: list[str] = field(default_factory=list)

    def summarize(self) -> dict[str, Any]:
        """Get a summary dict for reporting."""
        return {
            "pipeline": self.config.pipeline,
            "duration_days": self.config.duration_days,
            "videos_per_day": self.config.videos_per_day,
            "total_videos_produced": self.total_videos_produced,
            "total_videos_approved": self.total_videos_approved,
            "overall_success_rate": f"{self.overall_success_rate:.1%}",
            "overall_approval_rate": f"{self.overall_approval_rate:.1%}",
            "total_cost_usd": f"${self.total_cost_usd:.2f}",
            "training_examples_collected": self.training_examples_collected,
            "common_failure_modes": self.common_failure_modes[:5],
            "prompt_tuning_notes": self.prompt_tuning_notes,
        }


# =============================================================================
# Burn-In Metrics Tracker
# =============================================================================


class BurnInTracker:
    """Tracks metrics across a burn-in period."""

    def __init__(
        self,
        config: BurnInConfig | None = None,
        metrics_store: MetricsStore | None = None,
    ):
        self.config = config or BurnInConfig()
        self._store = metrics_store or get_metrics_store()
        self._daily: dict[str, DailySummary] = {}
        self._failure_modes: list[str] = []
        self._training_count = 0

    def record_video_produced(
        self,
        *,
        date: str | None = None,
        cost_usd: float = 0.0,
        iterations: int = 1,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a video production during burn-in."""
        day = date or datetime.now().strftime("%Y-%m-%d")
        summary = self._get_or_create_daily(day)
        summary.videos_produced += 1
        summary.total_cost_usd += cost_usd

        # Update running average for iterations
        n = summary.videos_produced
        summary.avg_iterations = (
            (summary.avg_iterations * (n - 1) + iterations) / n
        )
        summary.avg_latency_ms = (
            (summary.avg_latency_ms * (n - 1) + latency_ms) / n
        )

    def record_review_decision(
        self,
        *,
        date: str | None = None,
        approved: bool,
        failure_mode: str | None = None,
    ) -> None:
        """Record a human review decision."""
        day = date or datetime.now().strftime("%Y-%m-%d")
        summary = self._get_or_create_daily(day)

        if approved:
            summary.videos_approved += 1
            self._training_count += 1
        else:
            summary.videos_rejected += 1
            if failure_mode:
                summary.failure_modes.append(failure_mode)
                self._failure_modes.append(failure_mode)

    def get_daily_summary(self, date: str) -> DailySummary | None:
        """Get the summary for a specific date."""
        return self._daily.get(date)

    def generate_report(self) -> BurnInReport:
        """Generate the full burn-in report."""
        report = BurnInReport(config=self.config)
        report.daily_summaries = list(self._daily.values())

        report.total_videos_produced = sum(d.videos_produced for d in report.daily_summaries)
        report.total_videos_approved = sum(d.videos_approved for d in report.daily_summaries)
        report.total_videos_rejected = sum(d.videos_rejected for d in report.daily_summaries)
        report.total_cost_usd = sum(d.total_cost_usd for d in report.daily_summaries)
        report.training_examples_collected = self._training_count

        if report.total_videos_produced > 0:
            report.overall_success_rate = (
                report.total_videos_produced
                / (self.config.duration_days * self.config.videos_per_day)
            )

        reviewed = report.total_videos_approved + report.total_videos_rejected
        if reviewed > 0:
            report.overall_approval_rate = report.total_videos_approved / reviewed

        # Aggregate failure modes
        from collections import Counter
        mode_counts = Counter(self._failure_modes)
        report.common_failure_modes = [
            f"{mode} ({count}x)" for mode, count in mode_counts.most_common(5)
        ]

        report.end_date = datetime.now()
        return report

    def check_targets(self) -> dict[str, bool]:
        """Check if burn-in targets are being met."""
        report = self.generate_report()
        settings = get_settings()

        return {
            "min_training_examples": (
                report.training_examples_collected >= self.config.min_training_examples
            ),
            "success_rate_above_80": report.overall_success_rate >= 0.8,
            "approval_rate_above_70": report.overall_approval_rate >= 0.7,
            "cost_below_threshold": (
                report.total_cost_usd / max(report.total_videos_produced, 1)
                <= settings.cost_alert_threshold_usd
            ),
        }

    def _get_or_create_daily(self, date: str) -> DailySummary:
        if date not in self._daily:
            self._daily[date] = DailySummary(date=date)
        return self._daily[date]
