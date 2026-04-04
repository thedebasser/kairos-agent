"""Kairos Agent -- Langfuse Tracing Sink & In-Memory Metrics.

Provides:
- ``LangfuseSink`` -- ``TracingSink`` implementation that forwards **all**
  pipeline events to Langfuse with proper hierarchy:

    Trace (1 per pipeline run)
    +-- Span  (1 per pipeline step / graph node)
    |   +-- Generation (1 per LLM call)
    |   +-- Span       (decisions, console messages)
    +-- ...

- Legacy ``trace_llm_call`` / ``trace_pipeline_step`` functions retained
  for backward compatibility (called directly from ``routing.py``).
- In-memory ``MetricsStore`` + ``AlertManager`` for cost tracking.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from kairos.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Langfuse Client Singleton
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


# =============================================================================
# Hierarchical TracingSink Implementation
# =============================================================================

class LangfuseSink:
    """TracingSink that forwards events to Langfuse with proper hierarchy.

    Creates one **trace** per pipeline run, with **spans** for each step
    and **generations** for each LLM call nested inside the step span.

    Handles all 9 event types:
    - RunStarted      -> creates the root Langfuse trace
    - RunCompleted    -> updates the trace with final status/cost
    - StepStarted     -> creates a span under the run trace
    - StepCompleted   -> ends the step span with status/duration
    - LLMCallStarted  -> (logged, generation created on completion)
    - LLMCallCompleted -> creates a generation under the step span
    - PromptRendered   -> attaches prompt metadata to the step span
    - Decision         -> creates a sub-span for agent reasoning
    - ConsoleMessage   -> creates an event on the step or run span

    Also feeds the in-memory ``MetricsStore`` for dashboard queries.
    """

    def __init__(self) -> None:
        # Langfuse object refs keyed by run_id / step_name
        self._run_traces: dict[str, Any] = {}       # run_id -> langfuse trace
        self._step_spans: dict[str, Any] = {}        # f"{run_id}:{step_name}" -> langfuse span
        self._run_metadata: dict[str, dict[str, Any]] = {}  # accumulated data per run
        self._current_step_name: str | None = None   # active step for generation nesting

    # -- TracingSink protocol ------------------------------------------------

    def on_event(self, event: Any) -> None:
        """Dispatch an event to Langfuse and the metrics store."""
        from kairos.ai.tracing.events import (
            ConsoleMessage,
            Decision,
            LLMCallCompleted,
            LLMCallStarted,
            PromptRendered,
            RunCompleted,
            RunStarted,
            StepCompleted,
            StepStarted,
        )

        try:
            if isinstance(event, RunStarted):
                self._on_run_started(event)
            elif isinstance(event, RunCompleted):
                self._on_run_completed(event)
            elif isinstance(event, StepStarted):
                self._on_step_started(event)
            elif isinstance(event, StepCompleted):
                self._on_step_completed(event)
            elif isinstance(event, LLMCallStarted):
                pass  # Generation is created on completion with full data
            elif isinstance(event, LLMCallCompleted):
                self._on_llm_call_completed(event)
            elif isinstance(event, PromptRendered):
                self._on_prompt_rendered(event)
            elif isinstance(event, Decision):
                self._on_decision(event)
            elif isinstance(event, ConsoleMessage):
                self._on_console(event)

            # Always feed local metrics store
            self._record_local_metrics(event)

        except Exception as exc:
            logger.warning("LangfuseSink.on_event(%s) failed: %s", getattr(event, "event_type", "?"), exc)

    def flush(self) -> None:
        client = get_langfuse_client()
        if client is not None:
            try:
                client.flush()
            except Exception as exc:
                logger.warning("LangfuseSink.flush failed: %s", exc)

    def close(self) -> None:
        self.flush()
        # Clean up references
        self._run_traces.clear()
        self._step_spans.clear()
        self._run_metadata.clear()

    # -- Event handlers (private) -------------------------------------------

    def _on_run_started(self, event: Any) -> None:
        """Create the root Langfuse trace for this pipeline run."""
        client = get_langfuse_client()
        if client is None:
            return

        trace = client.trace(
            id=event.run_id,
            name=f"pipeline:{event.pipeline}",
            session_id=event.pipeline_run_id,
            metadata={
                "pipeline": event.pipeline,
                "pipeline_run_id": event.pipeline_run_id,
            },
            tags=["kairos", f"pipeline:{event.pipeline}"],
        )
        self._run_traces[event.run_id] = trace
        self._run_metadata[event.run_id] = {
            "pipeline": event.pipeline,
            "total_cost": 0.0,
            "total_llm_calls": 0,
        }
        logger.debug("Langfuse trace created: run=%s pipeline=%s", event.run_id, event.pipeline)

    def _on_run_completed(self, event: Any) -> None:
        """Update the root trace with final run status and metrics."""
        client = get_langfuse_client()
        if client is None:
            return

        trace = self._run_traces.get(event.run_id)
        if trace is None:
            # Trace wasn't created (possibly Langfuse was disabled at start)
            return

        # Update the trace with final status
        trace.update(
            metadata={
                "pipeline": event.pipeline,
                "pipeline_run_id": event.pipeline_run_id,
                "status": event.status,
                "total_duration_ms": event.total_duration_ms,
                "total_cost_usd": event.total_cost_usd,
                "total_llm_calls": event.total_llm_calls,
                "errors": event.errors,
                "final_video_path": event.final_video_path,
                "concept_title": event.concept_title,
            },
            tags=["kairos", f"pipeline:{event.pipeline}", f"status:{event.status}"],
        )
        logger.debug(
            "Langfuse trace completed: run=%s status=%s cost=$%.4f calls=%d",
            event.run_id, event.status, event.total_cost_usd, event.total_llm_calls,
        )

    def _on_step_started(self, event: Any) -> None:
        """Create a span under the run trace for this step."""
        client = get_langfuse_client()
        if client is None:
            return

        trace = self._run_traces.get(event.run_id)
        if trace is None:
            return

        span = trace.span(
            name=event.step_name,
            metadata={
                "step_number": event.step_number,
                "attempt": event.attempt,
            },
        )
        key = f"{event.run_id}:{event.step_name}"
        self._step_spans[key] = span
        self._current_step_name = event.step_name
        logger.debug("Langfuse span created: step=%s (span key=%s)", event.step_name, key)

    def _on_step_completed(self, event: Any) -> None:
        """End the step span with status and duration."""
        key = f"{event.run_id}:{event.step_name}"
        span = self._step_spans.get(key)
        if span is None:
            # Also feed metrics even without Langfuse
            return

        span.end(
            metadata={
                "status": event.status,
                "duration_ms": event.duration_ms,
                "errors": event.errors,
            },
            level="ERROR" if event.status == "error" else "DEFAULT",
            status_message=f"{event.step_name}: {event.status} ({event.duration_ms}ms)",
        )
        # Clean up
        self._step_spans.pop(key, None)
        self._current_step_name = None
        logger.debug("Langfuse span ended: step=%s status=%s", event.step_name, event.status)

    def _on_llm_call_completed(self, event: Any) -> None:
        """Create a Langfuse generation under the step span."""
        # Always record metrics locally
        record_metric(
            "llm_call",
            cost_usd=event.cost_usd,
            model=event.model_resolved,
            latency_ms=event.latency_ms,
            status=event.status,
        )

        # Accumulate run totals
        meta = self._run_metadata.get(event.run_id)
        if meta:
            meta["total_cost"] += event.cost_usd
            meta["total_llm_calls"] += 1

        client = get_langfuse_client()
        if client is None:
            return

        # Try to nest under step span; fall back to run trace
        key = f"{event.run_id}:{event.step_name}"
        parent = self._step_spans.get(key) or self._run_traces.get(event.run_id)
        if parent is None:
            return

        gen_metadata: dict[str, Any] = {
            "call_id": event.call_id,
            "model_alias": event.model_alias,
            "model_type": event.model_type,
            "provider": event.provider,
            "call_pattern": event.call_pattern,
            "routing_outcome": event.routing_outcome,
            "has_thinking": event.has_thinking,
            "status": event.status,
        }
        if event.error:
            gen_metadata["error"] = event.error

        parent.generation(
            name=f"{event.step_name}:{event.model_alias}",
            model=event.model_resolved,
            usage={
                "input": event.tokens_in,
                "output": event.tokens_out,
                "total": event.tokens_in + event.tokens_out,
            },
            metadata=gen_metadata,
            level="ERROR" if event.status == "error" else "DEFAULT",
            status_message=event.error if event.error else None,
        )

        logger.debug(
            "Langfuse generation: %s:%s model=%s tokens=%d/%d cost=$%.4f",
            event.step_name, event.model_alias, event.model_resolved,
            event.tokens_in, event.tokens_out, event.cost_usd,
        )

    def _on_prompt_rendered(self, event: Any) -> None:
        """Attach prompt template metadata to the step span."""
        key = f"{event.run_id}:{event.step_name}"
        span = self._step_spans.get(key)
        if span is None:
            return

        span.event(
            name=f"prompt:{event.template_name}",
            metadata={
                "template_name": event.template_name,
                "template_version": event.template_version,
                "template_hash": event.template_hash,
                "variables": event.variables,
            },
        )

    def _on_decision(self, event: Any) -> None:
        """Record an agent decision as an event on the step span."""
        key = f"{event.run_id}:{event.step_name}"
        span = self._step_spans.get(key)
        if span is None:
            return

        span.event(
            name=f"decision:{event.step_name}",
            metadata={
                "saw": event.saw,
                "decided": event.decided,
                "action": event.action,
                "reasoning": event.reasoning[:2000] if event.reasoning else "",
            },
        )

    def _on_console(self, event: Any) -> None:
        """Record a console message as an event on the step or run."""
        key = f"{event.run_id}:{event.step_name}" if event.step_name else None
        parent = (self._step_spans.get(key) if key else None) or self._run_traces.get(event.run_id)
        if parent is None:
            return

        level_map = {
            "error": "ERROR",
            "warning": "WARNING",
            "debug": "DEBUG",
        }
        parent.event(
            name=f"console:{event.level}",
            metadata={"message": event.message, "step_name": event.step_name},
            level=level_map.get(event.level, "DEFAULT"),
        )

    # -- Local metrics helper -----------------------------------------------

    def _record_local_metrics(self, event: Any) -> None:
        """Feed the in-memory MetricsStore from relevant events."""
        from kairos.ai.tracing.events import LLMCallCompleted, StepCompleted

        if isinstance(event, StepCompleted):
            record_metric(
                "pipeline_step",
                step=event.step_name,
                status=event.status,
                duration_ms=event.duration_ms,
            )


# =============================================================================
# Legacy Functions (called directly from routing.py)
# =============================================================================
# These are called from call_llm / call_llm_code / call_with_quality_fallback
# in routing.py. They create independent generations that are also attached
# to the run trace if one exists. This gives us Langfuse coverage even for
# LLM calls that happen outside the RunTracer step context managers.
# =============================================================================


def trace_llm_call(
    *,
    trace_name: str,
    model: str,
    model_resolved: str | None = None,
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
    thinking: str | None = None,
) -> None:
    """Log an LLM call to Langfuse for tracing.

    If Langfuse is not configured, records to local metrics only.

    This function is called directly from ``routing.py`` for every LLM
    call (``call_llm``, ``call_llm_code``, ``call_with_quality_fallback``).
    It creates a generation on the **active run trace** if one exists,
    or creates an independent trace if called outside a pipeline run.

    ``thinking`` is the extended-thinking / chain-of-thought content
    returned by Anthropic/Qwen models.
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
        # Build output representation
        output_str: str | None = None
        if output is not None:
            if hasattr(output, "model_dump_json"):
                output_str = output.model_dump_json(indent=2)[:5000]
            else:
                output_str = str(output)[:5000]

        # Use the resolved model for Langfuse so price lookup works;
        # keep the alias in metadata for human reference.
        langfuse_model = model_resolved or model

        gen_metadata: dict[str, Any] = {
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "model_alias": model,
            **(metadata or {}),
        }
        if thinking:
            # Store thinking content in Langfuse metadata (truncated to 10k
            # to stay within Langfuse field limits)
            gen_metadata["thinking"] = thinking[:10_000]
            gen_metadata["thinking_chars"] = len(thinking)

        # Try to find the active run trace from the LangfuseSink singleton
        # so this generation nests under the pipeline run.
        from kairos.ai.tracing.tracer import get_tracer

        parent: Any = None
        tracer = get_tracer()
        if tracer and tracer._run_id:
            # Prefer the active step span so generations nest properly;
            # fall back to the run trace if no step is active.
            for sink in tracer._sinks:
                if isinstance(sink, LangfuseSink):
                    if sink._current_step_name:
                        key = f"{tracer._run_id}:{sink._current_step_name}"
                        parent = sink._step_spans.get(key) or sink._run_traces.get(tracer._run_id)
                    else:
                        parent = sink._run_traces.get(tracer._run_id)
                    break

        if parent is None:
            # No active run — create an independent trace
            trace_meta: dict[str, Any] = {"pipeline_run_id": pipeline_run_id}
            parent = client.trace(
                name=trace_name,
                metadata=trace_meta,
                tags=["kairos", "standalone_llm_call"],
            )

        parent.generation(
            name=f"{trace_name}_generation",
            model=langfuse_model,
            input=input_messages,
            output=output_str,
            usage={
                "input": tokens_in,
                "output": tokens_out,
                "total": tokens_in + tokens_out,
            },
            metadata=gen_metadata,
            level="ERROR" if status == "error" else "DEFAULT",
            status_message=error if error else None,
        )

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
    """Log a pipeline step to Langfuse as a span (legacy).

    Prefer using ``RunTracer.step()`` context manager which emits
    ``StepStarted``/``StepCompleted`` events through the ``LangfuseSink``.
    Retained for backward compatibility.
    """
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
        trace = client.trace(
            name=f"pipeline:{pipeline_run_id}",
            metadata={"pipeline_run_id": pipeline_run_id},
        )
        trace.span(
            name=step_name,
            metadata={
                "status": status,
                "duration_ms": duration_ms,
                **(metadata or {}),
            },
        )
    except Exception as e:
        logger.warning("Failed to log pipeline step to Langfuse: %s", e)


# =============================================================================
# In-Memory Metrics Store
# =============================================================================


@dataclass
class MetricEntry:
    """Single metric data point."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [e for e in self._entries if e.timestamp >= cutoff]

    def get_videos_today(self) -> int:
        """Count pipeline_step completions for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return sum(
            e.cost_usd
            for e in self._entries
            if e.name == "llm_call" and e.timestamp >= cutoff
        )

    def get_model_latency_stats(self, *, minutes: int = 60) -> dict[str, dict[str, float]]:
        """Get average latency per model over the last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
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
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AlertManager:
    """Checks metrics against thresholds and fires alerts.

    Alerts are accumulated in-memory and sent to Slack / Discord when configured.
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

        # Send alerts to configured webhooks
        for alert in new_alerts:
            self._send_slack_alert(alert)
            self._send_discord_alert(alert)

        return new_alerts

    def check_run_cost(
        self,
        run_id: str,
        total_cost_usd: float,
    ) -> Alert | None:
        """Check if a single pipeline run's cost exceeds the threshold.

        Called after each pipeline run completes. Compares the run's
        total cost against ``cost_alert_threshold_usd`` from settings.

        Returns:
            An Alert if threshold exceeded, else None.
        """
        threshold = self._settings.cost_alert_threshold_usd
        if total_cost_usd <= threshold:
            return None

        alert = Alert(
            alert_type="run_cost",
            message=(
                f"Pipeline run {run_id} cost (${total_cost_usd:.4f}) "
                f"exceeds threshold (${threshold:.2f})"
            ),
            severity="warning",
            value=total_cost_usd,
            threshold=threshold,
        )
        logger.warning("COST ALERT: %s", alert.message)
        self._alerts.append(alert)
        self._send_slack_alert(alert)
        self._send_discord_alert(alert)
        return alert

    def _send_slack_alert(self, alert: Alert) -> None:
        """Send an alert notification to Slack webhook."""
        webhook_url = getattr(self._settings, "slack_webhook_url", "")
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

    def _send_discord_alert(self, alert: Alert) -> None:
        """Send an alert notification to Discord webhook.

        Uses the Discord webhook format with embeds for rich formatting.
        Only sends if ``discord_webhook_url`` is configured in settings.
        """
        webhook_url = self._settings.discord_webhook_url
        if not webhook_url:
            return

        try:
            import httpx

            color = 0xFF0000 if alert.severity == "critical" else 0xFFA500
            payload = {
                "embeds": [
                    {
                        "title": f"Pipeline Alert — {alert.severity.upper()}",
                        "description": alert.message,
                        "color": color,
                        "fields": [
                            {"name": "Type", "value": alert.alert_type, "inline": True},
                            {"name": "Value", "value": f"{alert.value:.4f}", "inline": True},
                            {"name": "Threshold", "value": f"{alert.threshold:.4f}", "inline": True},
                        ],
                        "timestamp": alert.timestamp.isoformat(),
                    }
                ],
            }

            with httpx.Client() as client:
                client.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.warning("Failed to send Discord alert: %s", e)

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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [a for a in self._alerts if a.timestamp >= cutoff]

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()
