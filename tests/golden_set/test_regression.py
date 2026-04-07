"""Tests for the golden set regression suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kairos.services.regression import (
    RegressionReport,
    RegressionResult,
    _stub_validate,
    load_golden_set,
    run_regression,
    save_report,
)


# ── Golden Set Loading ────────────────────────────────────────────────

class TestLoadGoldenSet:
    def test_load_default_golden_set(self):
        briefs = load_golden_set()
        assert len(briefs) == 15
        assert all("concept_id" in b for b in briefs)
        assert all("category" in b for b in briefs)
        assert all("title" in b for b in briefs)

    def test_all_briefs_have_required_fields(self):
        briefs = load_golden_set()
        required = [
            "concept_id", "pipeline", "category", "title",
            "visual_brief", "simulation_requirements", "audio_brief",
            "hook_text", "novelty_score", "feasibility_score",
        ]
        for brief in briefs:
            for field_name in required:
                assert field_name in brief, f"Missing {field_name} in {brief['title']}"

    def test_categories_diverse(self):
        briefs = load_golden_set()
        categories = {b["category"] for b in briefs}
        assert len(categories) >= 5  # At least 5 different categories

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_golden_set(Path("/nonexistent/briefs.json"))


# ── Stub Validation ──────────────────────────────────────────────────

class TestStubValidation:
    def test_valid_brief_passes(self):
        brief = {
            "concept_id": "test-1",
            "pipeline": "physics",
            "category": "gravity_chaos",
            "title": "Test",
            "visual_brief": "A test concept",
            "simulation_requirements": {"engine": "blender"},
            "audio_brief": {"mood": "calm"},
            "hook_text": "Test Hook",
            "novelty_score": 7.0,
            "feasibility_score": 8.0,
        }
        result = _stub_validate(brief)
        assert result.success is True

    def test_missing_field_fails(self):
        brief = {"concept_id": "test-1", "title": "Incomplete"}
        result = _stub_validate(brief)
        assert result.success is False
        assert "Missing fields" in result.error

    def test_bad_simulation_requirements(self):
        brief = {
            "concept_id": "test-1",
            "pipeline": "physics",
            "category": "gravity_chaos",
            "title": "Test",
            "visual_brief": "A test",
            "simulation_requirements": "not a dict",
            "audio_brief": {"mood": "calm"},
            "hook_text": "Hook",
            "novelty_score": 7.0,
            "feasibility_score": 8.0,
        }
        result = _stub_validate(brief)
        assert result.success is False


# ── Regression Runner ─────────────────────────────────────────────────

class TestRegressionRunner:
    async def test_run_with_stub_all_pass(self):
        briefs = load_golden_set()
        report = await run_regression(briefs)
        assert report.total_concepts == 15
        assert report.total_success == 15
        assert report.success_rate == 1.0

    async def test_run_empty_set(self):
        report = await run_regression([])
        assert report.total_concepts == 0
        assert report.success_rate == 0.0

    async def test_run_with_custom_runner(self):
        async def always_fail(brief):
            return RegressionResult(
                concept_id=brief["concept_id"],
                category=brief.get("category", "unknown"),
                title=brief.get("title", "unknown"),
                success=False,
                error="Intentional failure",
            )

        briefs = load_golden_set()[:3]  # Use 3 briefs
        report = await run_regression(briefs, pipeline_runner=always_fail)
        assert report.total_success == 0
        assert report.total_failed == 3
        assert report.success_rate == 0.0

    async def test_run_with_exception_runner(self):
        async def raise_error(brief):
            raise RuntimeError("Pipeline crashed")

        briefs = [{"concept_id": "x", "category": "y", "title": "z"}]
        report = await run_regression(briefs, pipeline_runner=raise_error)
        assert report.total_failed == 1
        assert "crashed" in report.results[0].error

    async def test_report_summary(self):
        briefs = load_golden_set()[:5]
        report = await run_regression(briefs)
        summary = report.summarize()
        assert "success_rate" in summary
        assert "total_concepts" in summary
        assert summary["total_concepts"] == 5


# ── Report Saving ─────────────────────────────────────────────────────

class TestSaveReport:
    async def test_save_report_to_disk(self, tmp_path):
        briefs = load_golden_set()[:3]
        report = await run_regression(briefs)
        filepath = save_report(report, output_dir=tmp_path)

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["total_concepts"] == 3

    async def test_report_file_naming(self, tmp_path):
        report = RegressionReport()
        filepath = save_report(report, output_dir=tmp_path)
        assert filepath.name.startswith("regression_")
        assert filepath.suffix == ".json"


# ── RegressionResult dataclass ────────────────────────────────────────

class TestRegressionResult:
    def test_defaults(self):
        r = RegressionResult(
            concept_id="test",
            category="physics",
            title="Test",
            success=True,
        )
        assert r.iterations == 0
        assert r.cost_usd == 0.0
        assert r.error is None
