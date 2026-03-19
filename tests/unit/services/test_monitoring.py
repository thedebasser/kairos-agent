"""Tests for kairos.ai.tracing.sinks.langfuse_sink — Metrics, Alerting & Langfuse integration."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from kairos.ai.tracing.sinks.langfuse_sink import (
    Alert,
    AlertManager,
    MetricEntry,
    MetricsStore,
    get_langfuse_client,
    get_metrics_store,
    record_metric,
    trace_llm_call,
    trace_pipeline_step,
)


# ── MetricsStore ──────────────────────────────────────────────────────

class TestMetricsStore:
    def test_record_and_get_recent(self):
        store = MetricsStore()
        store.record(MetricEntry(name="llm_call", cost_usd=0.01, model="gpt-4"))
        store.record(MetricEntry(name="llm_call", cost_usd=0.02, model="gpt-4"))

        recent = store.get_recent(minutes=5)
        assert len(recent) == 2

    def test_get_recent_filters_old_entries(self):
        store = MetricsStore()
        old = MetricEntry(
            name="llm_call",
            timestamp=datetime.now() - timedelta(hours=2),
        )
        new = MetricEntry(name="llm_call")
        store.record(old)
        store.record(new)

        recent = store.get_recent(minutes=60)
        assert len(recent) == 1

    def test_max_entries_limit(self):
        store = MetricsStore(max_entries=5)
        for i in range(10):
            store.record(MetricEntry(name="test", cost_usd=float(i)))

        recent = store.get_recent(minutes=9999)
        assert len(recent) == 5

    def test_get_videos_today(self):
        store = MetricsStore()
        # Published video today
        store.record(MetricEntry(name="pipeline_step", step="publish", status="success"))
        store.record(MetricEntry(name="pipeline_step", step="publish", status="success"))
        # Failed publish
        store.record(MetricEntry(name="pipeline_step", step="publish", status="error"))
        # Different step
        store.record(MetricEntry(name="pipeline_step", step="simulate", status="success"))

        assert store.get_videos_today() == 2

    def test_get_success_rate_all_success(self):
        store = MetricsStore()
        for _ in range(10):
            store.record(MetricEntry(name="pipeline_step", status="success"))

        assert store.get_success_rate(days=7) == 1.0

    def test_get_success_rate_mixed(self):
        store = MetricsStore()
        for _ in range(8):
            store.record(MetricEntry(name="pipeline_step", status="success"))
        for _ in range(2):
            store.record(MetricEntry(name="pipeline_step", status="error"))

        assert store.get_success_rate(days=7) == 0.8

    def test_get_success_rate_no_data(self):
        store = MetricsStore()
        assert store.get_success_rate(days=7) == 1.0  # No data = assume OK

    def test_get_rolling_cost_average(self):
        store = MetricsStore()
        store.record(MetricEntry(name="llm_call", cost_usd=0.10))
        store.record(MetricEntry(name="llm_call", cost_usd=0.20))
        store.record(MetricEntry(name="llm_call", cost_usd=0.30))

        avg = store.get_rolling_cost_average(days=7)
        assert abs(avg - 0.20) < 0.001

    def test_get_rolling_cost_average_no_data(self):
        store = MetricsStore()
        assert store.get_rolling_cost_average(days=7) == 0.0

    def test_get_total_cost(self):
        store = MetricsStore()
        store.record(MetricEntry(name="llm_call", cost_usd=0.10))
        store.record(MetricEntry(name="llm_call", cost_usd=0.25))

        assert abs(store.get_total_cost(days=7) - 0.35) < 0.001

    def test_get_model_latency_stats(self):
        store = MetricsStore()
        store.record(MetricEntry(name="llm_call", model="gpt-4", latency_ms=100))
        store.record(MetricEntry(name="llm_call", model="gpt-4", latency_ms=200))
        store.record(MetricEntry(name="llm_call", model="claude", latency_ms=150))

        stats = store.get_model_latency_stats(minutes=60)
        assert "gpt-4" in stats
        assert "claude" in stats
        assert stats["gpt-4"]["avg_ms"] == 150.0
        assert stats["gpt-4"]["max_ms"] == 200
        assert stats["gpt-4"]["min_ms"] == 100
        assert stats["gpt-4"]["count"] == 2
        assert stats["claude"]["count"] == 1

    def test_get_dashboard_data(self):
        store = MetricsStore()
        store.record(MetricEntry(name="pipeline_step", step="publish", status="success"))
        store.record(MetricEntry(name="llm_call", cost_usd=0.15, model="gpt-4", latency_ms=300))

        data = store.get_dashboard_data()
        assert "videos_today" in data
        assert "success_rate_7d" in data
        assert "rolling_cost_avg_7d" in data
        assert "total_cost_7d" in data
        assert "model_latency" in data
        assert "queue_depth" in data
        assert data["videos_today"] == 1

    def test_get_queue_depth(self):
        store = MetricsStore()
        assert store.get_queue_depth() == 0


# ── AlertManager ──────────────────────────────────────────────────────

class TestAlertManager:
    def test_no_alerts_when_healthy(self):
        store = MetricsStore()
        for _ in range(10):
            store.record(MetricEntry(name="pipeline_step", status="success"))
        store.record(MetricEntry(name="llm_call", cost_usd=0.05))

        am = AlertManager(metrics_store=store)
        alerts = am.check_all()
        assert len(alerts) == 0

    def test_cost_alert_triggers(self):
        store = MetricsStore()
        # $0.50 avg exceeds $0.30 threshold
        for _ in range(5):
            store.record(MetricEntry(name="llm_call", cost_usd=0.50))

        am = AlertManager(metrics_store=store)
        alerts = am.check_all()
        cost_alerts = [a for a in alerts if a.alert_type == "cost_threshold"]
        assert len(cost_alerts) == 1
        assert cost_alerts[0].severity == "warning"
        assert cost_alerts[0].value == 0.50

    def test_success_rate_alert_triggers(self):
        store = MetricsStore()
        # 70% success rate < 80% threshold
        for _ in range(7):
            store.record(MetricEntry(name="pipeline_step", status="success"))
        for _ in range(3):
            store.record(MetricEntry(name="pipeline_step", status="error"))

        am = AlertManager(metrics_store=store)
        alerts = am.check_all()
        rate_alerts = [a for a in alerts if a.alert_type == "success_rate"]
        assert len(rate_alerts) == 1
        assert rate_alerts[0].severity == "critical"

    def test_both_alerts_trigger(self):
        store = MetricsStore()
        for _ in range(5):
            store.record(MetricEntry(name="llm_call", cost_usd=0.50))
        for _ in range(5):
            store.record(MetricEntry(name="pipeline_step", status="success"))
        for _ in range(5):
            store.record(MetricEntry(name="pipeline_step", status="error"))

        am = AlertManager(metrics_store=store)
        alerts = am.check_all()
        assert len(alerts) == 2

    def test_get_alerts_history(self):
        store = MetricsStore()
        for _ in range(5):
            store.record(MetricEntry(name="llm_call", cost_usd=0.50))

        am = AlertManager(metrics_store=store)
        am.check_all()
        history = am.get_alerts(days=7)
        assert len(history) >= 1

    def test_clear_alerts(self):
        store = MetricsStore()
        for _ in range(5):
            store.record(MetricEntry(name="llm_call", cost_usd=0.50))

        am = AlertManager(metrics_store=store)
        am.check_all()
        am.clear_alerts()
        assert len(am.get_alerts(days=7)) == 0


# ── Langfuse Client ──────────────────────────────────────────────────

class TestLangfuseClient:
    def test_returns_none_without_keys(self):
        """Langfuse client returns None when keys are not configured."""
        import kairos.ai.tracing.sinks.langfuse_sink as m

        m._langfuse_client = None  # Reset singleton
        client = get_langfuse_client()
        # Default settings have empty keys
        assert client is None

    @patch("kairos.services.monitoring.get_settings")
    @patch("kairos.services.monitoring.get_langfuse_client")
    def test_trace_llm_call_without_langfuse(self, mock_client, mock_settings):
        """trace_llm_call works fine without Langfuse configured."""
        mock_client.return_value = None

        # Should not raise
        trace_llm_call(
            trace_name="test",
            model="gpt-4",
            input_messages=[{"role": "user", "content": "hi"}],
            output="hello",
            latency_ms=100,
        )

    def test_trace_llm_call_records_local_metric(self):
        """trace_llm_call always records to local metrics store."""
        import kairos.ai.tracing.sinks.langfuse_sink as m

        store = MetricsStore()
        original_store = m._metrics_store
        m._metrics_store = store

        try:
            trace_llm_call(
                trace_name="test",
                model="test-model",
                input_messages=[],
                output="result",
                cost_usd=0.05,
                latency_ms=200,
            )
            recent = store.get_recent(minutes=1)
            assert len(recent) == 1
            assert recent[0].model == "test-model"
            assert recent[0].cost_usd == 0.05
        finally:
            m._metrics_store = original_store

    def test_trace_pipeline_step_records_metric(self):
        """trace_pipeline_step records to local metrics."""
        import kairos.ai.tracing.sinks.langfuse_sink as m

        store = MetricsStore()
        original_store = m._metrics_store
        m._metrics_store = store

        try:
            trace_pipeline_step(
                pipeline_run_id="abc-123",
                step_name="simulate",
                status="success",
                duration_ms=5000,
            )
            recent = store.get_recent(minutes=1)
            assert len(recent) == 1
            assert recent[0].step == "simulate"
        finally:
            m._metrics_store = original_store


# ── Global helpers ────────────────────────────────────────────────────

class TestGlobalHelpers:
    def test_get_metrics_store_returns_singleton(self):
        s1 = get_metrics_store()
        s2 = get_metrics_store()
        assert s1 is s2

    def test_record_metric_convenience(self):
        import kairos.ai.tracing.sinks.langfuse_sink as m

        store = MetricsStore()
        original_store = m._metrics_store
        m._metrics_store = store

        try:
            record_metric("test", cost_usd=0.1, model="x")
            assert len(store.get_recent(minutes=1)) == 1
        finally:
            m._metrics_store = original_store


# ── Alert dataclass ──────────────────────────────────────────────────

class TestAlertDataclass:
    def test_alert_fields(self):
        a = Alert(
            alert_type="cost_threshold",
            message="Cost too high",
            severity="warning",
            value=0.50,
            threshold=0.30,
        )
        assert a.alert_type == "cost_threshold"
        assert a.severity == "warning"
        assert a.value == 0.50
        assert a.threshold == 0.30
        assert isinstance(a.timestamp, datetime)
