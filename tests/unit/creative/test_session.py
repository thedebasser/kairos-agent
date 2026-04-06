"""Tests for session storage."""

from __future__ import annotations

import pytest
import yaml

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    EnvironmentSpec,
    EnvironmentType,
    GroundConfig,
    LightingConfig,
    ObjectRole,
    PlacedObject,
    SceneManifest,
    StepValidationResult,
)
from kairos.pipelines.domino.creative.session import SessionStorage

pytestmark = pytest.mark.unit


def _env() -> EnvironmentSpec:
    return EnvironmentSpec(
        type=EnvironmentType.INDOOR,
        ground=GroundConfig(texture="tile"),
        lighting=LightingConfig(preset="warm"),
    )


def _manifest() -> SceneManifest:
    return SceneManifest(
        theme="test",
        narrative="Test scene.",
        environment=_env(),
        objects=[
            PlacedObject(
                asset_id="t1",
                position=(0.0, 0.0, 0.5),
                role=ObjectRole.FUNCTIONAL,
                surface_name="top",
            ),
        ],
        domino_count=100,
    )


def _passing_validation() -> StepValidationResult:
    return StepValidationResult(
        agent=AgentRole.SET_DESIGNER,
        passed=True,
        checks=[{"name": "test", "passed": True, "message": "OK"}],
    )


def _failing_validation() -> StepValidationResult:
    return StepValidationResult(
        agent=AgentRole.SET_DESIGNER,
        passed=False,
        checks=[{"name": "bad", "passed": False, "message": "oops"}],
        error_summary="oops",
    )


class TestSessionStorage:
    def test_creates_directory(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        assert storage.root.exists()
        assert storage.root == tmp_path / "sessions" / "sess1"

    def test_save_manifest(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        storage.save_manifest(_manifest())
        manifest_path = storage.root / "manifest.yaml"
        assert manifest_path.exists()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert data["theme"] == "test"

    def test_save_attempt_creates_files(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        m = _manifest()
        v = _passing_validation()
        attempt_dir = storage.save_attempt(AgentRole.SET_DESIGNER, 1, m, v)
        assert (attempt_dir / "output.yaml").exists()
        assert (attempt_dir / "validation.yaml").exists()
        assert (attempt_dir / "summary.md").exists()

    def test_summary_content_passing(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        attempt_dir = storage.save_attempt(
            AgentRole.SET_DESIGNER, 1, _manifest(), _passing_validation(),
        )
        summary = (attempt_dir / "summary.md").read_text(encoding="utf-8")
        assert "PASSED" in summary
        assert "Attempt 1" in summary

    def test_summary_content_failing(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        attempt_dir = storage.save_attempt(
            AgentRole.SET_DESIGNER, 2, _manifest(), _failing_validation(),
        )
        summary = (attempt_dir / "summary.md").read_text(encoding="utf-8")
        assert "FAILED" in summary
        assert "oops" in summary

    def test_save_result(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        storage.save_result({"success": True, "total_attempts": 3})
        result_path = storage.root / "result.yaml"
        assert result_path.exists()
        data = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        assert data["success"] is True

    def test_save_final_review(self, tmp_path):
        storage = SessionStorage(tmp_path, "sess1")
        review = {"status": "EXHAUSTED", "issues": []}
        storage.save_final_review(review)
        path = storage.root / "final_review.yaml"
        assert path.exists()
