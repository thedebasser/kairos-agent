"""Kairos Agent — Monitoring & Observability.

Provides:
- Langfuse integration for LLM call tracing (wraps LiteLLM calls)
- Cost tracking and alerting
- Success rate monitoring
- Pipeline metrics dashboard data
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from kairos.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Langfuse Integration
# =============================================================================


_langfuse_client = None


def get_langfuse_client():
    """Get or create the singleton Langfuse client.

    Returns None if Langfuse is not configured (no keys set).
    """
    global _langfuse_client  # noqa: PLW0603

    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    if _langfuse_client is None:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse client initialised (host=%s)", settings.langfuse_host)

    return _langfuse_client


def trace_llm_call(
    *,
    trace_name: str,
    model: str,
    input_messages: list[dict[str, str]],
    output: Any,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    status: str = "success",
    error: str | None = None,
    pipeline_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log an LLM call to Langfuse for tracing.

    If Langfuse is not configured, logs locally only.
    """
    client = get_langfuse_client()

    # Always record in local metrics
    record_metric(
        "llm_call",
        cost_usd=cost_usd,
        model=model,
        latency_ms=latency_ms,
        status=status,
    )

    if client is None:
        logger.debug(
            "Langfuse not configured — LLM call logged locally: %s model=%s cost=$%.4f",
            trace_name,
            model,
            cost_usd,
        )
        return

    try:
        generation = client.start_observation(
            name=f"{trace_name}_generation",
            as_type="generation",
            model=model,
            input=input_messages,
            output=str(output)[:2000] if output else None,
            usage_details={
                "input": tokens_in,
                "output": tokens_out,
                "total": tokens_in + tokens_out,
            },
            metadata={
                "pipeline_run_id": pipeline_run_id,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "status": status,
                "error": error,
                **(metadata or {}),
            },
        )
        generation.end()

        logger.debug("Langfuse generation logged: %s", trace_name)

    except Exception as e:
        logger.warning("Failed to log to Langfuse: %s", e)


def trace_pipeline_step(
    *,
    pipeline_run_id: str,
    step_name: str,
    status: str = "success",
    duration_ms: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a pipeline step to Langfuse as a span."""
    client = get_langfuse_client()

    record_metric(
        "pipeline_step",
        step=step_name,
        status=status,
        duration_ms=duration_ms,
    )

    if client is None:
        return

    try:
        span = client.start_observation(
            name=step_name,
            as_type="span",
            metadata={
                "pipeline_run_id": pipeline_run_id,
                "status": status,
                "duration_ms": duration_ms,
                **(metadata or {}),
            },
        )
        span.end()
    except Exception as e:
        logger.warning("Failed to log pipeline step to Langfuse: %s", e)


# =============================================================================
# In-Memory Metrics Store
# =============================================================================


@dataclass
class MetricEntry:
    """Single metric data point."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    cost_usd: float = 0.0
    model: str = ""
    latency_ms: int = 0
    status: str = "success"
    step: str = ""
    duration_ms: int = 0


class MetricsStore:
    """In-memory rolling metrics store for monitoring.

    Keeps a fixed-size deque of recent metrics for fast querying.
    Production would use a time-series database.
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: deque[MetricEntry] = deque(maxlen=max_entries)
        self._daily_counts: defaultdict[str, int] = defaultdict(int)
        self._daily_costs: defaultdict[str, float] = defaultdict(float)

    def record(self, entry: MetricEntry) -> None:
        """Record a new metric entry."""
        self._entries.append(entry)

        day_key = entry.timestamp.strftime("%Y-%m-%d")
        if entry.name == "llm_call":
            self._daily_costs[day_key] += entry.cost_usd

    def get_recent(self, *, minutes: int = 60) -> list[MetricEntry]:
        """Get entries from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [e for e in self._entries if e.timestamp >= cutoff]

    def get_videos_today(self) -> int:
        """Count pipeline_step completions for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return sum(
            1
            for e in self._entries
            if e.name == "pipeline_step"
            and e.step == "publish"
            and e.status == "success"
            and e.timestamp.strftime("%Y-%m-%d") == today
        )

    def get_success_rate(self, *, days: int = 7) -> float:
        """Calculate pipeline success rate over the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        pipeline_steps = [
            e
            for e in self._entries
            if e.name == "pipeline_step" and e.timestamp >= cutoff
        ]
        if not pipeline_steps:
            return 1.0  # No data = assume OK

        successes = sum(1 for e in pipeline_steps if e.status == "success")
        return successes / len(pipeline_steps)

    def get_rolling_cost_average(self, *, days: int = 7) -> float:
        """Calculate rolling average cost per video over the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        costs = [
            e.cost_usd
            for e in self._entries
            if e.name == "llm_call" and e.timestamp >= cutoff and e.cost_usd > 0
        ]
        if not costs:
            return 0.0
        return sum(costs) / len(costs)

    def get_total_cost(self, *, days: int = 7) -> float:
        """Total cost over the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return sum(
            e.cost_usd
            for e in self._entries
            if e.name == "llm_call" and e.timestamp >= cutoff
        )

    def get_model_latency_stats(self, *, minutes: int = 60) -> dict[str, dict[str, float]]:
        """Get average latency per model over the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        by_model: defaultdict[str, list[int]] = defaultdict(list)

        for e in self._entries:
            if e.name == "llm_call" and e.timestamp >= cutoff and e.model:
                by_model[e.model].append(e.latency_ms)

        return {
            model: {
                "avg_ms": sum(vals) / len(vals),
                "max_ms": max(vals),
                "min_ms": min(vals),
                "count": len(vals),
            }
            for model, vals in by_model.items()
        }

    def get_queue_depth(self) -> int:
        """Placeholder for queue depth metric."""
        return 0

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get all dashboard metrics in one call."""
        return {
            "videos_today": self.get_videos_today(),
            "success_rate_7d": self.get_success_rate(days=7),
            "rolling_cost_avg_7d": self.get_rolling_cost_average(days=7),
            "total_cost_7d": self.get_total_cost(days=7),
            "model_latency": self.get_model_latency_stats(),
            "queue_depth": self.get_queue_depth(),
        }


# Global metrics store
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Get the global metrics store."""
    return _metrics_store


def record_metric(name: str, **kwargs: Any) -> None:
    """Convenience function to record a metric."""
    entry = MetricEntry(name=name, **kwargs)
    _metrics_store.record(entry)


# =============================================================================
# Alerting
# =============================================================================


@dataclass
class Alert:
    """An alert triggered by metric thresholds."""

    alert_type: str
    message: str
    severity: str  # "warning" | "critical"
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)


class AlertManager:
    """Checks metrics against thresholds and fires alerts.

    Alerts are accumulated in-memory and sent to Slack when configured.
    """

    def __init__(self, metrics_store: MetricsStore | None = None):
        self._store = metrics_store or get_metrics_store()
        self._alerts: list[Alert] = []
        self._settings = get_settings()

    def check_all(self) -> list[Alert]:
        """Run all alert checks and return any triggered alerts."""
        new_alerts: list[Alert] = []
        new_alerts.extend(self._check_cost_threshold())
        new_alerts.extend(self._check_success_rate())
        self._alerts.extend(new_alerts)

        # Send alerts to Slack
        for alert in new_alerts:
            self._send_slack_alert(alert)

        return new_alerts

    def _send_slack_alert(self, alert: Alert) -> None:
        """Send an alert notification to Slack webhook."""
        webhook_url = self._settings.slack_webhook_url
        if not webhook_url:
            return

        try:
            import httpx

            emoji = "🚨" if alert.severity == "critical" else "⚠️"
            message = {
                "text": f"{emoji} [{alert.severity.upper()}] {alert.alert_type}: {alert.message}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"{emoji} *Pipeline Alert — {alert.severity.upper()}*\n"
                                f"• Type: `{alert.alert_type}`\n"
                                f"• {alert.message}\n"
                                f"• Value: `{alert.value:.4f}` | Threshold: `{alert.threshold:.4f}`"
                            ),
                        },
                    }
                ],
            }

            with httpx.Client() as client:
                client.post(webhook_url, json=message, timeout=10)
        except Exception as e:
            logger.warning("Failed to send Slack alert: %s", e)

    def _check_cost_threshold(self) -> list[Alert]:
        """Alert if 7-day rolling cost average > threshold."""
        avg_cost = self._store.get_rolling_cost_average(days=7)
        threshold = self._settings.cost_alert_threshold_usd

        if avg_cost > threshold:
            alert = Alert(
                alert_type="cost_threshold",
                message=(
                    f"7-day rolling cost average (${avg_cost:.4f}) "
                    f"exceeds threshold (${threshold:.2f})"
                ),
                severity="warning",
                value=avg_cost,
                threshold=threshold,
            )
            logger.warning("ALERT: %s", alert.message)
            return [alert]
        return []

    def _check_success_rate(self) -> list[Alert]:
        """Alert if 7-day success rate drops below 80%."""
        success_rate = self._store.get_success_rate(days=7)
        threshold = 0.80

        if success_rate < threshold:
            alert = Alert(
                alert_type="success_rate",
                message=(
                    f"7-day success rate ({success_rate:.1%}) "
                    f"is below threshold ({threshold:.0%})"
                ),
                severity="critical",
                value=success_rate,
                threshold=threshold,
            )
            logger.warning("ALERT: %s", alert.message)
            return [alert]
        return []

    def get_alerts(self, *, days: int = 7) -> list[Alert]:
        """Get alerts from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [a for a in self._alerts if a.timestamp >= cutoff]

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()
