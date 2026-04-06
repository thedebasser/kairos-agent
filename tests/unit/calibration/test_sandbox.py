"""Integration tests for the calibration sandbox loop.

Blender is fully mocked — these tests run without Blender installed.
They validate the iteration state machine, correction logic, dry-run
behaviour, and session artifact writing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.calibration.models import (
    CalibrationStatus,
    CorrectionFactors,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.sandbox import (
    _compute_correction,
    _parse_failure_modes,
    run_calibration,
)
from kairos.calibration.models import FailureMode


def _make_smoke_crash() -> dict[str, Any]:
    """Simulate a smoke test that crashes with no JSON output."""
    return {
        "returncode": 1,
        "stdout": "",
        "stderr": "Traceback: RuntimeError: rigid body world missing",
        "json_output": None,
    }

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def straight_30() -> ScenarioDescriptor:
    return ScenarioDescriptor(
        path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0),
        domino_count=30,
    )


def _make_blender_success(completion: float = 1.0) -> dict[str, Any]:
    """Simulate a passing smoke test result from Blender."""
    return {
        "returncode": 0,
        "stdout": json.dumps({
            "passed": True,
            "reason": "All smoke checks passed",
            "checks": [
                {"name": "trigger_works", "passed": True, "message": "30/30 fell"},
                {"name": "chain_propagation", "passed": True, "message": f"{completion:.0%} fell"},
                {"name": "physics_stability", "passed": True, "message": "Physics stable"},
            ],
            "fallen_count": int(30 * completion),
            "total_count": 30,
            "completion_ratio": completion,
        }),
        "stderr": "",
        "json_output": {
            "passed": True,
            "checks": [
                {"name": "trigger_works", "passed": True},
                {"name": "chain_propagation", "passed": True},
                {"name": "physics_stability", "passed": True},
            ],
            "fallen_count": int(30 * completion),
            "total_count": 30,
            "completion_ratio": completion,
        },
    }


def _make_blender_failure(
    failure_check: str = "chain_propagation",
    completion: float = 0.5,
) -> dict[str, Any]:
    """Simulate a failing smoke test result from Blender."""
    return {
        "returncode": 0,  # Blender itself ran fine
        "stdout": json.dumps({
            "passed": False,
            "checks": [
                {"name": "trigger_works", "passed": True, "message": "triggered"},
                {"name": failure_check, "passed": False, "message": f"Only {completion:.0%} fell"},
                {"name": "physics_stability", "passed": True, "message": "stable"},
            ],
            "completion_ratio": completion,
        }),
        "stderr": "",
        "json_output": {
            "passed": False,
            "checks": [
                {"name": "trigger_works", "passed": True},
                {"name": failure_check, "passed": False, "message": f"Only {completion:.0%} fell"},
                {"name": "physics_stability", "passed": True},
            ],
            "completion_ratio": completion,
        },
    }


def _make_generation_success() -> dict[str, Any]:
    """Simulate a successful generate_domino_course.py result."""
    return {
        "returncode": 0,
        "stdout": json.dumps({"status": "ok", "domino_count": 30}),
        "stderr": "",
        "json_output": {"status": "ok", "domino_count": 30},
    }


# =============================================================================
# _parse_failure_modes
# =============================================================================

class TestParseFailureModes:
    def test_chain_propagation_failure(self) -> None:
        validation = {
            "checks": [
                {"name": "chain_propagation", "passed": False, "message": "Only 50% fell"},
            ]
        }
        modes = _parse_failure_modes(validation)
        assert FailureMode.INCOMPLETE_PROPAGATION in modes

    def test_trigger_failure(self) -> None:
        validation = {
            "checks": [
                {"name": "trigger_works", "passed": False, "message": "trigger missed"},
            ]
        }
        modes = _parse_failure_modes(validation)
        assert FailureMode.TRIGGER_MISS in modes

    def test_physics_stability_failure(self) -> None:
        validation = {
            "checks": [
                {"name": "physics_stability", "passed": False, "message": "flew away"},
            ]
        }
        modes = _parse_failure_modes(validation)
        assert FailureMode.EXPLOSION in modes

    def test_all_passing_returns_unknown(self) -> None:
        """When no checks failed, we get the fallback UNKNOWN mode."""
        modes = _parse_failure_modes({"checks": []})
        assert modes == [FailureMode.UNKNOWN]


# =============================================================================
# _compute_correction
# =============================================================================

class TestComputeCorrection:
    def test_trigger_miss_increases_impulse(self) -> None:
        from kairos.calibration.models import BASELINE_PHYSICS
        params = dict(BASELINE_PHYSICS)
        result = _compute_correction(params, [FailureMode.TRIGGER_MISS], iteration=1)
        assert result["trigger_impulse"] > params["trigger_impulse"]
        assert result["trigger_tilt_degrees"] > params["trigger_tilt_degrees"]

    def test_incomplete_propagation_tightens_spacing(self) -> None:
        from kairos.calibration.models import BASELINE_PHYSICS
        params = dict(BASELINE_PHYSICS)
        result = _compute_correction(params, [FailureMode.INCOMPLETE_PROPAGATION], iteration=1)
        assert result["spacing_ratio"] < params["spacing_ratio"]

    def test_explosion_reduces_bounce(self) -> None:
        from kairos.calibration.models import BASELINE_PHYSICS
        params = dict(BASELINE_PHYSICS)
        result = _compute_correction(params, [FailureMode.EXPLOSION], iteration=1)
        assert result["domino_bounce"] < params["domino_bounce"]

    def test_no_param_goes_below_zero(self) -> None:
        from kairos.calibration.models import BASELINE_PHYSICS
        params = dict(BASELINE_PHYSICS)
        params["domino_bounce"] = 0.01  # near zero already
        result = _compute_correction(params, [FailureMode.EXPLOSION], iteration=1)
        assert result["domino_bounce"] >= 0.0


# =============================================================================
# run_calibration — mocked Blender
# =============================================================================

class TestRunCalibration:
    """Full sandbox loop with mocked run_blender_script."""

    @pytest.mark.asyncio
    async def test_passes_on_first_iteration(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """If Blender passes immediately, session resolves in 1 iteration."""
        # Always success
        blender_responses = [
            _make_generation_success(),
            _make_blender_success(1.0),
        ]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=5,
                dry_run=True,
            )

        assert session.status in (CalibrationStatus.RESOLVED, CalibrationStatus.PROMOTED)
        assert session.iteration_count == 1

    @pytest.mark.asyncio
    async def test_iterates_on_failure_then_resolves(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """Fail twice, succeed on third try — resolves in 3 iterations."""
        blender_responses = [
            # iter 1
            _make_generation_success(), _make_blender_failure(completion=0.5),
            # iter 2
            _make_generation_success(), _make_blender_failure(completion=0.7),
            # iter 3
            _make_generation_success(), _make_blender_success(1.0),
        ]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=5,
                dry_run=True,
            )

        assert session.status in (CalibrationStatus.RESOLVED, CalibrationStatus.PROMOTED)
        assert session.iteration_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_iterations_is_unresolved(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """Never pass → status is UNRESOLVED."""
        # Alternate generation + smoke failure pairs
        gen = _make_generation_success()
        fail = _make_blender_failure(completion=0.3)
        blender_responses = [gen, fail] * 3

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=3,
                dry_run=True,
            )

        assert session.status == CalibrationStatus.UNRESOLVED

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_store(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """With dry_run=True and a KnowledgeBase mock, store() must never be called."""
        blender_responses = [_make_generation_success(), _make_blender_success(1.0)]

        mock_kb = MagicMock()
        mock_kb.lookup_starting_params.return_value = None  # No prior knowledge

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=mock_kb,
                max_iterations=5,
                dry_run=True,
            )

        mock_kb.store.assert_not_called()
        # Status should be RESOLVED (not PROMOTED) since dry_run skips the promote step
        assert session.status == CalibrationStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_live_run_calls_store(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """With dry_run=False and passing calibration, store() is called exactly once."""
        blender_responses = [_make_generation_success(), _make_blender_success(1.0)]

        mock_kb = MagicMock()
        mock_kb.lookup_starting_params.return_value = None

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=mock_kb,
                max_iterations=5,
                dry_run=False,
            )

        mock_kb.store.assert_called_once()
        assert session.status == CalibrationStatus.PROMOTED

    @pytest.mark.asyncio
    async def test_session_artifacts_written_to_disk(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """Session directories and artifact files are created."""
        blender_responses = [_make_generation_success(), _make_blender_success(1.0)]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=5,
                dry_run=True,
            )

        session_dir = tmp_path / "sessions" / str(session.session_id)
        assert (session_dir / "scenario.yaml").exists()
        assert (session_dir / "result.yaml").exists()

    @pytest.mark.asyncio
    async def test_generation_failure_skips_smoke_test(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """If generation fails, skip smoke test and keep iterating."""
        gen_failure = {"returncode": 1, "stdout": "", "stderr": "Blender error", "json_output": None}
        # Second generation succeeds
        blender_responses = [
            gen_failure,
            _make_generation_success(), _make_blender_success(1.0),
        ]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=5,
                dry_run=True,
            )

        # Should still resolve on the second attempt
        assert session.status in (CalibrationStatus.RESOLVED, CalibrationStatus.PROMOTED)

    @pytest.mark.asyncio
    async def test_script_crash_breaks_immediately(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """When smoke test produces no JSON, session breaks immediately as SCRIPT_CRASH."""
        blender_responses = [
            _make_generation_success(),
            _make_smoke_crash(),
        ]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=10,
                dry_run=True,
            )

        assert session.status == CalibrationStatus.UNRESOLVED
        # Must stop after 1 iteration — no pointless retries
        assert session.iteration_count == 1
        last = session.iterations[0]
        assert last.failure_modes == [FailureMode.SCRIPT_CRASH]
        assert "crashed" in last.failure_details.lower()

    @pytest.mark.asyncio
    async def test_stuck_loop_escalates_at_5(
        self,
        straight_30: ScenarioDescriptor,
        tmp_path: Path,
    ) -> None:
        """After 5 consecutive empty corrections, the stuck-loop breaker
        escalates to broader parameter exploration."""
        # All failures return the same result — solver_iterations already capped
        def _make_capped_failure() -> dict[str, Any]:
            return {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "json_output": {
                    "passed": False,
                    "checks": [{"name": "generic_fail", "passed": False}],
                    "completion_ratio": 0.0,
                },
            }

        # Starting from baseline (solver=20, substeps=20), UNKNOWN bumps both
        # by +5 per iter.  substeps caps at 30 (iter 3), solver caps at 60
        # (iter 9).  First empty correction at iter 9; 5th consecutive empty
        # at iter 13.  Need 14 iters to see the escalation applied.
        blender_responses = [
            val for _ in range(14)
            for val in (_make_generation_success(), _make_capped_failure())
        ]

        with (
            patch("kairos.calibration.sandbox.run_blender_script",
                  side_effect=blender_responses),
            patch("kairos.calibration.sandbox._calibration_base_dir",
                  return_value=tmp_path),
            patch("kairos.calibration.sandbox._detect_blender_version",
                  return_value="Blender 5.0.0"),
        ):
            session = await run_calibration(
                straight_30,
                knowledge_base=None,
                max_iterations=14,
                dry_run=True,
            )

        assert session.status == CalibrationStatus.UNRESOLVED
        # The stuck-loop breaker escalates at the 5th consecutive empty
        # correction with broader params (spacing, impulse, friction, mass).
        found_escalation = False
        for it in session.iterations:
            delta = it.correction_applied or {}
            if "spacing_ratio" in delta or "trigger_impulse" in delta:
                found_escalation = True
                break
        assert found_escalation, (
            "Stuck-loop breaker should have escalated with broader parameters "
            "but no spacing_ratio/trigger_impulse corrections were found"
        )
