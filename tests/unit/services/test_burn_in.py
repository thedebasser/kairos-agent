"""Tests for kairos.services.burn_in — Production Burn-In."""

from __future__ import annotations

import pytest

from kairos.services.burn_in import (
    BurnInConfig,
    BurnInReport,
    BurnInTracker,
    DailySummary,
)


# ── BurnInConfig ──────────────────────────────────────────────────────

class TestBurnInConfig:
    def test_defaults(self):
        c = BurnInConfig()
        assert c.duration_days == 14
        assert c.videos_per_day == 2
        assert c.pipeline == "physics"
        assert c.auto_publish is False
        assert c.min_training_examples == 20

    def test_custom(self):
        c = BurnInConfig(duration_days=7, videos_per_day=3)
        assert c.duration_days == 7
        assert c.videos_per_day == 3


# ── DailySummary ──────────────────────────────────────────────────────

class TestDailySummary:
    def test_defaults(self):
        s = DailySummary(date="2025-01-01")
        assert s.videos_produced == 0
        assert s.total_cost_usd == 0.0
        assert s.failure_modes == []


# ── BurnInTracker ─────────────────────────────────────────────────────

class TestBurnInTracker:
    def test_record_video(self):
        tracker = BurnInTracker()
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.25, iterations=3)

        summary = tracker.get_daily_summary("2025-01-01")
        assert summary is not None
        assert summary.videos_produced == 1
        assert summary.total_cost_usd == 0.25
        assert summary.avg_iterations == 3.0

    def test_record_multiple_videos_same_day(self):
        tracker = BurnInTracker()
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.20, iterations=2, latency_ms=100)
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.30, iterations=4, latency_ms=200)

        summary = tracker.get_daily_summary("2025-01-01")
        assert summary.videos_produced == 2
        assert abs(summary.total_cost_usd - 0.50) < 0.001
        assert summary.avg_iterations == 3.0
        assert summary.avg_latency_ms == 150.0

    def test_record_review_approved(self):
        tracker = BurnInTracker()
        tracker.record_review_decision(date="2025-01-01", approved=True)

        summary = tracker.get_daily_summary("2025-01-01")
        assert summary.videos_approved == 1
        assert summary.videos_rejected == 0

    def test_record_review_rejected_with_mode(self):
        tracker = BurnInTracker()
        tracker.record_review_decision(
            date="2025-01-01",
            approved=False,
            failure_mode="bad_physics",
        )

        summary = tracker.get_daily_summary("2025-01-01")
        assert summary.videos_rejected == 1
        assert "bad_physics" in summary.failure_modes

    def test_get_nonexistent_day(self):
        tracker = BurnInTracker()
        assert tracker.get_daily_summary("2099-01-01") is None

    def test_generate_empty_report(self):
        tracker = BurnInTracker()
        report = tracker.generate_report()
        assert report.total_videos_produced == 0
        assert report.overall_success_rate == 0.0

    def test_generate_report_with_data(self):
        config = BurnInConfig(duration_days=2, videos_per_day=2)
        tracker = BurnInTracker(config=config)

        # Day 1: 2 videos, 1 approved
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.20)
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.30)
        tracker.record_review_decision(date="2025-01-01", approved=True)
        tracker.record_review_decision(date="2025-01-01", approved=False, failure_mode="bad_edit")

        # Day 2: 2 videos, 2 approved
        tracker.record_video_produced(date="2025-01-02", cost_usd=0.15)
        tracker.record_video_produced(date="2025-01-02", cost_usd=0.25)
        tracker.record_review_decision(date="2025-01-02", approved=True)
        tracker.record_review_decision(date="2025-01-02", approved=True)

        report = tracker.generate_report()
        assert report.total_videos_produced == 4
        assert report.total_videos_approved == 3
        assert report.total_videos_rejected == 1
        assert abs(report.total_cost_usd - 0.90) < 0.001
        assert report.overall_approval_rate == 0.75
        assert report.training_examples_collected == 3

    def test_report_summarize(self):
        tracker = BurnInTracker()
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.10)
        tracker.record_review_decision(date="2025-01-01", approved=True)

        report = tracker.generate_report()
        summary = report.summarize()
        assert "total_videos_produced" in summary
        assert "overall_success_rate" in summary

    def test_check_targets_all_met(self):
        config = BurnInConfig(duration_days=1, videos_per_day=2, min_training_examples=2)
        tracker = BurnInTracker(config=config)

        tracker.record_video_produced(date="2025-01-01", cost_usd=0.10)
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.10)
        tracker.record_review_decision(date="2025-01-01", approved=True)
        tracker.record_review_decision(date="2025-01-01", approved=True)

        targets = tracker.check_targets()
        assert targets["min_training_examples"] is True
        assert targets["approval_rate_above_70"] is True

    def test_check_targets_not_met(self):
        config = BurnInConfig(duration_days=14, videos_per_day=2, min_training_examples=20)
        tracker = BurnInTracker(config=config)

        # Only 1 video, far from target
        tracker.record_video_produced(date="2025-01-01", cost_usd=0.50)
        tracker.record_review_decision(date="2025-01-01", approved=False, failure_mode="bad")

        targets = tracker.check_targets()
        assert targets["min_training_examples"] is False
        assert targets["approval_rate_above_70"] is False

    def test_common_failure_modes(self):
        tracker = BurnInTracker()
        for _ in range(5):
            tracker.record_review_decision(approved=False, failure_mode="bad_physics")
        for _ in range(3):
            tracker.record_review_decision(approved=False, failure_mode="bad_audio")
        for _ in range(1):
            tracker.record_review_decision(approved=False, failure_mode="bad_caption")

        report = tracker.generate_report()
        assert "bad_physics (5x)" in report.common_failure_modes[0]
        assert "bad_audio (3x)" in report.common_failure_modes[1]


# ── BurnInReport ──────────────────────────────────────────────────────

class TestBurnInReport:
    def test_defaults(self):
        r = BurnInReport()
        assert r.total_videos_produced == 0
        assert r.prompt_tuning_notes == []
        assert r.common_failure_modes == []

    def test_summarize(self):
        r = BurnInReport(
            total_videos_produced=28,
            total_videos_approved=24,
            overall_success_rate=0.95,
            overall_approval_rate=0.86,
            total_cost_usd=7.50,
        )
        s = r.summarize()
        assert s["total_videos_produced"] == 28
        assert "95.0%" in s["overall_success_rate"]
