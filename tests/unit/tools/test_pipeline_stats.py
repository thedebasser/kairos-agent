"""Unit tests for the pipeline_stats module."""

from __future__ import annotations

import pytest

from kairos.tools.pipeline_stats import PipelineStats, StepFailureRate


# ---------------------------------------------------------------------------
# PipelineStats data model tests
# ---------------------------------------------------------------------------


class TestPipelineStats:
    """Tests for the PipelineStats dataclass."""

    def test_success_rate_no_runs(self) -> None:
        stats = PipelineStats(period_days=7, total_runs=0)
        assert stats.success_rate == 0.0

    def test_success_rate_some_runs(self) -> None:
        stats = PipelineStats(period_days=7, total_runs=10, completed=7)
        assert stats.success_rate == pytest.approx(0.7)

    def test_success_rate_all_complete(self) -> None:
        stats = PipelineStats(period_days=7, total_runs=5, completed=5)
        assert stats.success_rate == pytest.approx(1.0)

    def test_prev_success_rate_zero(self) -> None:
        stats = PipelineStats(period_days=7, prev_total_runs=0)
        assert stats.prev_success_rate == 0.0

    def test_prev_success_rate(self) -> None:
        stats = PipelineStats(period_days=7, prev_total_runs=20, prev_completed=15)
        assert stats.prev_success_rate == pytest.approx(0.75)

    def test_summary_contains_key_fields(self) -> None:
        stats = PipelineStats(
            period_days=7,
            total_runs=10,
            completed=8,
            failed=1,
            running=1,
            avg_duration_sec=120.5,
            avg_cost_usd=0.025,
            total_cost_usd=0.25,
        )
        text = stats.summary()
        assert "last 7 days" in text
        assert "Total Runs:    10" in text
        assert "Completed:   8" in text
        assert "Failed:      1" in text
        assert "80.0%" in text
        assert "$0.0250" in text

    def test_summary_with_trend(self) -> None:
        stats = PipelineStats(
            period_days=7,
            total_runs=10,
            completed=9,
            prev_total_runs=10,
            prev_completed=7,
        )
        text = stats.summary()
        assert "↑" in text
        assert "70.0%" in text
        assert "90.0%" in text

    def test_summary_with_step_failures(self) -> None:
        stats = PipelineStats(
            period_days=7,
            total_runs=10,
            completed=8,
            step_failures=[
                StepFailureRate(step_name="simulation", total=10, failed=3),
                StepFailureRate(step_name="video_editor", total=10, failed=1),
            ],
        )
        text = stats.summary()
        assert "Per-Step Failure Rates:" in text
        assert "simulation" in text
        assert "30.0%" in text

    def test_summary_with_model_usage(self) -> None:
        stats = PipelineStats(
            period_days=7,
            total_runs=5,
            model_usage={
                "claude-sonnet-4-20250514": {"calls": 20, "cost": 0.05, "p50_ms": 1200},
            },
        )
        text = stats.summary()
        assert "Model Usage:" in text
        assert "claude-sonnet-4-20250514" in text

    def test_to_dict_structure(self) -> None:
        stats = PipelineStats(
            period_days=14,
            total_runs=20,
            completed=15,
            failed=3,
            running=2,
            avg_duration_sec=90.0,
            avg_cost_usd=0.015,
            total_cost_usd=0.3,
            step_failures=[
                StepFailureRate(step_name="idea", total=20, failed=2),
            ],
        )
        d = stats.to_dict()
        assert d["period_days"] == 14
        assert d["total_runs"] == 20
        assert d["success_rate"] == 75.0
        assert d["total_cost_usd"] == 0.3
        assert len(d["step_failures"]) == 1
        assert d["step_failures"][0]["step"] == "idea"
        assert d["step_failures"][0]["rate"] == 10.0


class TestStepFailureRate:
    """Tests for the StepFailureRate dataclass."""

    def test_rate_normal(self) -> None:
        sf = StepFailureRate(step_name="sim", total=100, failed=25)
        assert sf.rate == pytest.approx(0.25)

    def test_rate_zero_total(self) -> None:
        sf = StepFailureRate(step_name="sim", total=0, failed=0)
        assert sf.rate == 0.0

    def test_rate_no_failures(self) -> None:
        sf = StepFailureRate(step_name="sim", total=50, failed=0)
        assert sf.rate == 0.0
