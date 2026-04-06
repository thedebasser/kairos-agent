"""Unit tests for the calibration perceptual validator.

All Blender and VLM calls are mocked — these tests run without Ollama or
Blender installed.  They verify:
  - Response parsing (_parse_perceptual_response)
  - Graceful degradation when render or VLM fails
  - Full happy-path integration
  - Sandbox correctly routes on perceptual pass / fail
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.calibration.models import (
    CalibrationStatus,
    FailureMode,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.perceptual_validator import (
    PERCEPTUAL_PASS_THRESHOLD,
    PerceptualResult,
    _parse_perceptual_response,
    validate_perceptually,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def scenario() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0),
        domino_count=30,
    )


def _make_render_result(
    ok: bool = True,
    frame_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Simulate a run_blender_script render result."""
    frames_dir = Path("/tmp/iter/perceptual_frames")
    paths = frame_paths or [
        str(frames_dir / f"frame_{n:05d}.png") for n in [1, 50, 100, 150, 200, 250, 300, 350]
    ]
    frames_rendered = [
        {"frame": i + 1, "path": p, "size_bytes": 25_000 if ok else 0, "ok": ok}
        for i, p in enumerate(paths)
    ]
    return {
        "returncode": 0 if ok else 1,
        "stdout": json.dumps({
            "total_scene_frames": 350,
            "frames_requested": [1, 50, 100, 150, 200, 250, 300, 350],
            "frames_rendered": frames_rendered,
            "ok_count": len(paths) if ok else 0,
        }),
        "stderr": "",
        "json_output": None,
    }


def _make_vlm_response(
    passed: bool = True,
    confidence: float = 0.9,
    issues: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Simulate an OllamaDirectResponse."""
    payload = {
        "passed": passed,
        "overall_confidence": confidence,
        "issues": issues or [],
    }
    resp = MagicMock()
    resp.content = json.dumps(payload)
    resp.thinking = None
    resp.tokens_in = 100
    resp.tokens_out = 50
    resp.model = "video-reviewer-default"
    return resp


# =============================================================================
# _parse_perceptual_response
# =============================================================================

class TestParsePerceptualResponse:
    def test_clean_pass(self):
        raw = json.dumps({"passed": True, "overall_confidence": 0.92, "issues": []})
        result = _parse_perceptual_response(raw, "test-model")
        assert result.passed is True
        assert result.confidence == pytest.approx(0.92)
        assert result.issues == []
        assert result.skipped is False
        assert result.model_used == "test-model"

    def test_clean_fail_with_explosion_issue(self):
        raw = json.dumps({
            "passed": False,
            "overall_confidence": 0.3,
            "issues": [
                {
                    "category": "explosion",
                    "severity": "critical",
                    "description": "All dominoes fell in frame 1",
                }
            ],
        })
        result = _parse_perceptual_response(raw, "test-model")
        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0]["category"] == "explosion"
        assert result.skipped is False

    def test_low_confidence_overrides_passed_true(self):
        """Confidence below threshold should override passed=True → False."""
        raw = json.dumps({
            "passed": True,
            "overall_confidence": PERCEPTUAL_PASS_THRESHOLD - 0.01,
            "issues": [],
        })
        result = _parse_perceptual_response(raw, "test-model")
        assert result.passed is False

    def test_confidence_at_threshold_passes(self):
        """Confidence at exactly the threshold should pass."""
        raw = json.dumps({
            "passed": True,
            "overall_confidence": PERCEPTUAL_PASS_THRESHOLD,
            "issues": [],
        })
        result = _parse_perceptual_response(raw, "test-model")
        assert result.passed is True

    def test_non_json_response_is_graceful_skip(self):
        result = _parse_perceptual_response("Sorry, I can't help with that.", "test-model")
        assert result.skipped is True
        assert result.passed is True  # graceful — don't block calibration

    def test_malformed_json_is_graceful_skip(self):
        result = _parse_perceptual_response("{not valid json", "test-model")
        assert result.skipped is True
        assert result.passed is True

    def test_json_embedded_in_prose(self):
        """VLM sometimes wraps JSON in prose — regex extraction should handle it."""
        raw = 'Here is my assessment: {"passed": false, "overall_confidence": 0.4, "issues": [{"category": "chain_break", "severity": "major", "description": "Gap visible"}]}'
        result = _parse_perceptual_response(raw, "test-model")
        assert result.passed is False
        assert result.issues[0]["category"] == "chain_break"


# =============================================================================
# validate_perceptually — graceful degradation paths
# =============================================================================

class TestValidatePerceptuallyGracefulDegradation:
    @pytest.mark.asyncio
    async def test_skips_when_render_fails_and_no_json(self, scenario, tmp_path):
        """render_calibration_frames.py failure with no JSON → graceful skip."""
        with patch("kairos.calibration.perceptual_validator.run_blender_script",
                   new_callable=AsyncMock) as mock_render:
            mock_render.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "Blender crashed",
                "json_output": None,
            }
            result = await validate_perceptually(
                blend_path=tmp_path / "domino.blend",
                scenario=scenario,
                output_dir=tmp_path,
            )

        assert result.skipped is True
        assert result.passed is True  # graceful — calibration can continue

    @pytest.mark.asyncio
    async def test_skips_when_no_ok_frames(self, scenario, tmp_path):
        """render returns JSON but all frames failed → graceful skip."""
        render_json = {
            "total_scene_frames": 350,
            "frames_requested": [1, 100, 200],
            "frames_rendered": [
                {"frame": 1, "path": str(tmp_path / "frame_00001.png"), "size_bytes": 0, "ok": False},
                {"frame": 100, "path": str(tmp_path / "frame_00100.png"), "size_bytes": 0, "ok": False},
            ],
            "ok_count": 0,
        }
        render_json_path = tmp_path / "perceptual_render.json"
        render_json_path.write_text(json.dumps(render_json), encoding="utf-8")

        with patch("kairos.calibration.perceptual_validator.run_blender_script",
                   new_callable=AsyncMock) as mock_render:
            mock_render.return_value = {
                "returncode": 0,
                "stdout": json.dumps(render_json),
                "stderr": "",
                "json_output": None,
            }
            result = await validate_perceptually(
                blend_path=tmp_path / "domino.blend",
                scenario=scenario,
                output_dir=tmp_path,
            )

        assert result.skipped is True
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_skips_when_ollama_unavailable(self, scenario, tmp_path):
        """VLM call raises ConnectionError → graceful skip."""
        # Create fake rendered frames
        frames_dir = tmp_path / "perceptual_frames"
        frames_dir.mkdir()
        frame_path = frames_dir / "frame_00001.png"
        frame_path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 2000)

        render_json = {
            "total_scene_frames": 350,
            "frames_requested": [1],
            "frames_rendered": [{"frame": 1, "path": str(frame_path), "size_bytes": 2000, "ok": True}],
            "ok_count": 1,
        }
        render_json_path = tmp_path / "perceptual_render.json"
        render_json_path.write_text(json.dumps(render_json), encoding="utf-8")

        with (
            patch("kairos.calibration.perceptual_validator.run_blender_script",
                  new_callable=AsyncMock) as mock_render,
            patch("kairos.calibration.perceptual_validator.call_ollama_direct",
                  side_effect=ConnectionError("Ollama not running")) as mock_vlm,
        ):
            mock_render.return_value = _make_render_result()
            result = await validate_perceptually(
                blend_path=tmp_path / "domino.blend",
                scenario=scenario,
                output_dir=tmp_path,
            )

        assert result.skipped is True
        assert result.passed is True


# =============================================================================
# validate_perceptually — happy paths
# =============================================================================

class TestValidatePerceptuallyHappyPath:
    @pytest.mark.asyncio
    async def test_returns_passed_result(self, scenario, tmp_path):
        """Full happy path: render succeeds, VLM returns pass."""
        frames_dir = tmp_path / "perceptual_frames"
        frames_dir.mkdir()
        frame_path = frames_dir / "frame_00001.png"
        frame_path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 2000)

        render_json = {
            "total_scene_frames": 350,
            "frames_requested": [1],
            "frames_rendered": [{"frame": 1, "path": str(frame_path), "size_bytes": 2000, "ok": True}],
            "ok_count": 1,
        }
        render_json_path = tmp_path / "perceptual_render.json"
        render_json_path.write_text(json.dumps(render_json), encoding="utf-8")

        with (
            patch("kairos.calibration.perceptual_validator.run_blender_script",
                  new_callable=AsyncMock) as mock_render,
            patch("kairos.calibration.perceptual_validator.call_ollama_direct",
                  return_value=_make_vlm_response(passed=True, confidence=0.88)) as mock_vlm,
        ):
            mock_render.return_value = _make_render_result()
            result = await validate_perceptually(
                blend_path=tmp_path / "domino.blend",
                scenario=scenario,
                output_dir=tmp_path,
            )

        assert result.passed is True
        assert result.skipped is False
        assert result.confidence == pytest.approx(0.88)
        assert (tmp_path / "perceptual_result.json").exists()

    @pytest.mark.asyncio
    async def test_returns_failed_result_on_explosion(self, scenario, tmp_path):
        """VLM reports explosion → passed=False, issues populated."""
        frames_dir = tmp_path / "perceptual_frames"
        frames_dir.mkdir()
        frame_path = frames_dir / "frame_00001.png"
        frame_path.write_bytes(b"\x89PNG\r\n" + b"\x00" * 2000)

        render_json = {
            "total_scene_frames": 350,
            "frames_requested": [1],
            "frames_rendered": [{"frame": 1, "path": str(frame_path), "size_bytes": 2000, "ok": True}],
            "ok_count": 1,
        }
        (tmp_path / "perceptual_render.json").write_text(json.dumps(render_json), encoding="utf-8")

        explosion_issues = [
            {"category": "explosion", "severity": "critical", "description": "All fell in frame 1"}
        ]

        with (
            patch("kairos.calibration.perceptual_validator.run_blender_script",
                  new_callable=AsyncMock) as mock_render,
            patch("kairos.calibration.perceptual_validator.call_ollama_direct",
                  return_value=_make_vlm_response(passed=False, confidence=0.2, issues=explosion_issues)),
        ):
            mock_render.return_value = _make_render_result()
            result = await validate_perceptually(
                blend_path=tmp_path / "domino.blend",
                scenario=scenario,
                output_dir=tmp_path,
            )

        assert result.passed is False
        assert result.skipped is False
        assert len(result.issues) == 1
        assert result.issues[0]["category"] == "explosion"


# =============================================================================
# Sandbox integration: perceptual failure routing
# =============================================================================

class TestSandboxPerceptualIntegration:
    """Verify that sandbox.run_calibration correctly routes on perceptual pass/fail."""

    def _make_smoke_success(self) -> dict[str, Any]:
        return {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "json_output": {
                "passed": True,
                "checks": [{"name": "chain_propagation", "passed": True}],
                "completion_ratio": 1.0,
                "physics_anomalies": 0,
            },
        }

    def _make_gen_success(self) -> dict[str, Any]:
        return {
            "returncode": 0,
            "stdout": json.dumps({"status": "ok"}),
            "stderr": "",
            "json_output": {"status": "ok"},
        }

    @pytest.mark.asyncio
    async def test_resolves_when_perceptual_passes(self, scenario, tmp_path):
        """Smoke PASS + perceptual PASS → session RESOLVED."""
        perceptual_pass = PerceptualResult(
            passed=True, confidence=0.85, skipped=False
        )

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  new_callable=AsyncMock) as mock_blender,
            patch("kairos.calibration.sandbox._detect_blender_version", return_value="5.0"),
            patch("kairos.calibration.sandbox.validate_perceptually",
                  new_callable=AsyncMock, return_value=perceptual_pass) as mock_vp,
            patch("kairos.calibration.sandbox._calibration_base_dir", return_value=tmp_path),
        ):
            gen_call = self._make_gen_success()
            smoke_call = self._make_smoke_success()
            mock_blender.side_effect = [gen_call, smoke_call]

            from kairos.calibration.sandbox import run_calibration
            session = await run_calibration(scenario, max_iterations=1, dry_run=True)

        assert session.status == CalibrationStatus.RESOLVED
        mock_vp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_continues_iterating_when_perceptual_fails(self, scenario, tmp_path):
        """Smoke PASS + perceptual FAIL → iterates; after 2 consecutive perfect-smoke/perceptual-fail
        iterations the sandbox trusts the smoke test and resolves (not UNRESOLVED)."""
        perceptual_fail = PerceptualResult(
            passed=False,
            confidence=0.2,
            skipped=False,
            issues=[{"category": "explosion", "severity": "critical",
                      "description": "All fell in frame 1"}],
        )

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  new_callable=AsyncMock) as mock_blender,
            patch("kairos.calibration.sandbox._detect_blender_version", return_value="5.0"),
            patch("kairos.calibration.sandbox.validate_perceptually",
                  new_callable=AsyncMock, return_value=perceptual_fail) as mock_vp,
            patch("kairos.calibration.sandbox._calibration_base_dir", return_value=tmp_path),
        ):
            # Each iteration: generation + smoke test (perceptual is called via mock_vp)
            mock_blender.side_effect = [
                self._make_gen_success(), self._make_smoke_success(),  # iter 1
                self._make_gen_success(), self._make_smoke_success(),  # iter 2
            ]

            from kairos.calibration.sandbox import run_calibration
            session = await run_calibration(scenario, max_iterations=2, dry_run=True)

        # After 2 consecutive perceptual-only failures with perfect smoke test,
        # the sandbox promotes (trusts smoke test over VLM false-negatives).
        assert session.status in (CalibrationStatus.RESOLVED, CalibrationStatus.PROMOTED)
        assert mock_vp.await_count == 2  # called on each smoke-pass

    @pytest.mark.asyncio
    async def test_unresolved_when_smoke_fails(self, scenario, tmp_path):
        """Smoke FAIL (incomplete cascade) → UNRESOLVED after max_iterations."""
        smoke_fail = {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "json_output": {
                "passed": False,
                "checks": [{"name": "chain_propagation", "passed": False,
                             "message": "Only 40% of dominoes fell"}],
                "completion_ratio": 0.4,
                "physics_anomalies": 0,
            },
        }

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  new_callable=AsyncMock) as mock_blender,
            patch("kairos.calibration.sandbox._detect_blender_version", return_value="5.0"),
            patch("kairos.calibration.sandbox._calibration_base_dir", return_value=tmp_path),
        ):
            mock_blender.side_effect = [
                self._make_gen_success(), smoke_fail,  # iter 1
                self._make_gen_success(), smoke_fail,  # iter 2
            ]

            from kairos.calibration.sandbox import run_calibration
            session = await run_calibration(scenario, max_iterations=2, dry_run=True)

        assert session.status == CalibrationStatus.UNRESOLVED

    @pytest.mark.asyncio
    async def test_resolves_when_perceptual_skipped(self, scenario, tmp_path):
        """Smoke PASS + perceptual SKIPPED → session RESOLVED (graceful degradation)."""
        perceptual_skipped = PerceptualResult(
            passed=True, confidence=0.0, skipped=True, skip_reason="Ollama not running"
        )

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  new_callable=AsyncMock) as mock_blender,
            patch("kairos.calibration.sandbox._detect_blender_version", return_value="5.0"),
            patch("kairos.calibration.sandbox.validate_perceptually",
                  new_callable=AsyncMock, return_value=perceptual_skipped),
            patch("kairos.calibration.sandbox._calibration_base_dir", return_value=tmp_path),
        ):
            mock_blender.side_effect = [self._make_gen_success(), self._make_smoke_success()]

            from kairos.calibration.sandbox import run_calibration
            session = await run_calibration(scenario, max_iterations=1, dry_run=True)

        assert session.status == CalibrationStatus.RESOLVED
