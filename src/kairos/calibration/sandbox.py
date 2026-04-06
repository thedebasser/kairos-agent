"""Kairos Agent — Calibration Sandbox (Iteration Loop).

The core calibration engine.  Receives a scenario, looks up prior
knowledge, iterates on parameters via Blender headless execution +
validation, and promotes successful calibrations through the quality gate
into the ChromaDB knowledge base.

Session artifacts are written to output/calibration/sessions/{session_id}/
and cleaned up after promotion or moved to unresolved/.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CalibrationEntry,
    CalibrationSession,
    CalibrationStatus,
    CorrectionFactors,
    FailureMode,
    IterationRecord,
    QualityGateResult,
    ScenarioDescriptor,
)
from kairos.calibration.knowledge_base import KnowledgeBase
from kairos.calibration.perceptual_validator import PerceptualResult, validate_perceptually
from kairos.calibration.quality_gate import evaluate as evaluate_quality_gate
from kairos.calibration.scenario import scenario_to_blender_config
from kairos.config import get_settings
from kairos.engines.blender.executor import run_blender_script

logger = logging.getLogger(__name__)


def _calibration_base_dir() -> Path:
    """Return the base directory for calibration output."""
    settings = get_settings()
    base = settings.calibration_output_dir
    if base is None:
        base = settings.project_root / "output" / "calibration"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _session_dir(session_id: str) -> Path:
    """Return the directory for a calibration session."""
    d = _calibration_base_dir() / "sessions" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iteration_dir(session_id: str, iteration: int) -> Path:
    """Return the directory for a specific iteration."""
    d = _session_dir(session_id) / "iterations" / f"iter_{iteration:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unresolved_dir(session_id: str) -> Path:
    """Return the unresolved directory for a failed session."""
    d = _calibration_base_dir() / "unresolved" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_blender_version() -> str:
    """Detect the installed Blender version."""
    from kairos.engines.blender.executor import find_blender
    blender = find_blender()
    if blender is None:
        return "unknown"

    import asyncio
    import subprocess
    try:
        result = subprocess.run(  # noqa: S603
            [str(blender), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Blender"):
                return line.strip()
    except Exception:
        pass
    return "unknown"


def _parse_failure_modes(validation: dict[str, Any]) -> list[FailureMode]:
    """Parse failure modes from smoke test validation output."""
    modes: list[FailureMode] = []
    checks = validation.get("checks", [])
    for check in checks:
        if check.get("passed", True):
            continue
        name = check.get("name", "")
        if "trigger" in name:
            modes.append(FailureMode.TRIGGER_MISS)
        elif "chain" in name or "propagation" in name:
            modes.append(FailureMode.INCOMPLETE_PROPAGATION)
        elif "stability" in name or "flew" in name:
            modes.append(FailureMode.EXPLOSION)
        elif "clipping" in name:
            modes.append(FailureMode.CLIPPING)
        else:
            modes.append(FailureMode.UNKNOWN)
    return modes or [FailureMode.UNKNOWN]


def _perceptual_issues_to_failure_modes(issues: list[dict[str, str]]) -> list[FailureMode]:
    """Map VLM issue categories to FailureMode enum values."""
    category_map = {
        "explosion": FailureMode.EXPLOSION,
        "launch": FailureMode.EXPLOSION,
        "chain_break": FailureMode.CHAIN_BREAK,
        "chain_incomplete": FailureMode.INCOMPLETE_PROPAGATION,
        "camera_coverage": FailureMode.UNKNOWN,
        "physics_unrealistic": FailureMode.UNKNOWN,
    }
    modes = [category_map.get(i.get("category", ""), FailureMode.UNKNOWN) for i in issues]
    return modes or [FailureMode.UNKNOWN]


def _compute_correction(
    current_params: dict[str, float],
    failure_modes: list[FailureMode],
    iteration: int,
) -> dict[str, float]:
    """Compute parameter adjustments based on failure modes.

    Returns a dict of parameter names → new absolute values.
    Each failure mode maps to specific parameter adjustments.
    """
    adjustments: dict[str, float] = dict(current_params)

    for mode in failure_modes:
        if mode == FailureMode.TRIGGER_MISS:
            # Increase trigger force
            adjustments["trigger_impulse"] = min(
                8.0, adjustments.get("trigger_impulse", 1.5) + 0.5
            )
            adjustments["trigger_tilt_degrees"] = min(
                15.0, adjustments.get("trigger_tilt_degrees", 8.0) + 2.0
            )

        elif mode == FailureMode.INCOMPLETE_PROPAGATION:
            # Tighten spacing for better energy transfer
            adjustments["spacing_ratio"] = max(
                0.25, adjustments.get("spacing_ratio", 0.35) - 0.02
            )
            # Increase physics fidelity
            adjustments["substeps_per_frame"] = min(
                30, adjustments.get("substeps_per_frame", 20) + 3
            )

        elif mode == FailureMode.CHAIN_BREAK:
            # Chain broke mid-sequence — tighten spacing, increase substeps
            adjustments["spacing_ratio"] = max(
                0.25, adjustments.get("spacing_ratio", 0.35) - 0.03
            )
            adjustments["solver_iterations"] = min(
                60, adjustments.get("solver_iterations", 20) + 10
            )

        elif mode == FailureMode.EXPLOSION:
            # Physics explosion — reduce bounce, increase friction
            adjustments["domino_bounce"] = max(
                0.0, adjustments.get("domino_bounce", 0.1) - 0.03
            )
            adjustments["domino_friction"] = min(
                0.9, adjustments.get("domino_friction", 0.6) + 0.05
            )
            adjustments["ground_friction"] = min(
                1.0, adjustments.get("ground_friction", 0.8) + 0.05
            )

        elif mode == FailureMode.CLIPPING:
            # Clipping — widen spacing slightly, reduce bounce
            adjustments["spacing_ratio"] = min(
                0.5, adjustments.get("spacing_ratio", 0.35) + 0.02
            )
            adjustments["domino_bounce"] = max(
                0.0, adjustments.get("domino_bounce", 0.1) - 0.02
            )

        elif mode in (FailureMode.FLOATING, FailureMode.DIRECTION_ERROR, FailureMode.UNKNOWN):
            # Generic: increase physics fidelity
            adjustments["substeps_per_frame"] = min(
                30, adjustments.get("substeps_per_frame", 20) + 5
            )
            adjustments["solver_iterations"] = min(
                60, adjustments.get("solver_iterations", 20) + 5
            )

    return adjustments


async def run_calibration(
    scenario: ScenarioDescriptor,
    *,
    knowledge_base: KnowledgeBase | None = None,
    max_iterations: int = 10,
    session_id: str | None = None,
    dry_run: bool = False,
) -> CalibrationSession:
    """Run a full calibration session for a scenario.

    1. Look up similar scenarios in ChromaDB
    2. Iterate: generate → validate → correct
    3. If resolved: submit to quality gate → promote to knowledge base
    4. If unresolved: archive for analysis

    Args:
        scenario: The scenario to calibrate.
        knowledge_base: ChromaDB knowledge base.  If None or dry_run=True,
            lookups and writes are skipped.
        max_iterations: Maximum calibration iterations before giving up.
        session_id: Optional stable session ID (useful for testing/retries).
        dry_run: When True, run the full Blender loop but skip all ChromaDB
            writes.  Session artifacts are still written to disk.  Use this
            to validate the loop end-to-end before committing to the DB.

    Returns the completed CalibrationSession.
    """
    settings = get_settings()
    sid = session_id or str(uuid4())
    session_path = _session_dir(sid)
    blender_version = _detect_blender_version()

    session = CalibrationSession(
        session_id=sid,
        scenario=scenario,
        max_iterations=max_iterations,
        blender_version=blender_version,
    )

    # Write scenario descriptor
    scenario_file = session_path / "scenario.yaml"
    scenario_file.write_text(
        yaml.dump(scenario.model_dump(mode="json"), default_flow_style=False),
        encoding="utf-8",
    )

    # ── Step 1: Look up prior calibrations ───────────────────────────
    starting_corrections: CorrectionFactors | None = None
    if knowledge_base is not None:
        try:
            starting_corrections = knowledge_base.lookup_starting_params(scenario)
            if starting_corrections:
                logger.info(
                    "Found prior calibrations — starting from composited corrections"
                )
        except Exception:
            logger.warning("ChromaDB lookup failed, starting from baseline", exc_info=True)

    if starting_corrections is None:
        starting_corrections = CorrectionFactors()

    current_params = starting_corrections.apply_to_baseline()
    session.starting_params = dict(current_params)

    # ── Step 2: Iteration loop ───────────────────────────────────────
    resolved = False
    last_anomalies = 0  # physics_anomalies from the passing smoke test
    _consecutive_perceptual_only_failures = 0  # smoke passed but VLM disagreed
    _consecutive_stuck_failures = 0  # track consecutive identical-param failures

    for iteration in range(1, max_iterations + 1):
        iter_dir = _iteration_dir(sid, iteration)
        logger.info("[calibration] === Session %s, iteration %d/%d ===", sid[:8], iteration, max_iterations)

        # Write params for this iteration
        params_file = iter_dir / "params.yaml"
        params_file.write_text(
            yaml.dump(current_params, default_flow_style=False),
            encoding="utf-8",
        )

        # Generate Blender config
        config = scenario_to_blender_config(scenario)
        # Override with current calibration params
        config.update({k: v for k, v in current_params.items() if k in config})
        # Ensure integer params
        for int_key in ("substeps_per_frame", "solver_iterations"):
            if int_key in config:
                config[int_key] = int(config[int_key])

        config_path = iter_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        # Run generate_domino_course.py
        blend_path = iter_dir / "domino_course.blend"
        gen_result_path = iter_dir / "generation_result.json"

        gen_result = await run_blender_script(
            "generate_domino_course.py",
            script_args=[
                "--config", str(config_path),
                "--output-blend", str(blend_path),
                "--output-json", str(gen_result_path),
            ],
            timeout_sec=300,
        )

        # Blender 5.x exits with code 1 even on success in background mode.
        # Treat generation as truly failed only when the blend file was NOT created.
        if gen_result["returncode"] != 0 and not blend_path.exists():
            logger.error(
                "[calibration] Course generation failed on iter %d: %s",
                iteration, gen_result.get("stderr", "")[-500:],
            )
            record = IterationRecord(
                iteration=iteration,
                params_used=current_params,
                validation_passed=False,
                failure_modes=[FailureMode.UNKNOWN],
                failure_details=f"Generation failed: {gen_result.get('stderr', '')[-200:]}",
            )
            session.iterations.append(record)
            continue
        elif gen_result["returncode"] != 0:
            logger.warning(
                "[calibration] Generation exited with code %d on iter %d but blend file exists "
                "(Blender 5.x background mode quirk — treating as success)",
                gen_result["returncode"], iteration,
            )

        # Run smoke test in calibration mode
        smoke_result = await run_blender_script(
            "smoke_test_domino.py",
            blend_file=str(blend_path),
            script_args=["--calibration-mode"],
            timeout_sec=300,
        )

        validation = smoke_result.get("json_output") or {}

        # ── Fail loudly on empty / null validation ──────────────────
        # If the smoke test produced no JSON at all, the Blender script
        # crashed before it could report results.  Treat this as a hard
        # error — log at ERROR level, record as SCRIPT_CRASH, and skip
        # the correction loop (the parameters are irrelevant when the
        # script itself is broken).
        if not validation:
            stderr_tail = smoke_result.get("stderr", "")[-500:]
            logger.error(
                "[calibration] SCRIPT CRASH on iter %d — smoke test produced "
                "no output (empty validation). This is likely a Blender "
                "script error, NOT a physics tuning problem.\n"
                "  stderr (last 500 chars): %s",
                iteration, stderr_tail or "<empty>",
            )
            record = IterationRecord(
                iteration=iteration,
                params_used=current_params,
                validation_passed=False,
                completion_ratio=0.0,
                failure_modes=[FailureMode.SCRIPT_CRASH],
                failure_details=(
                    f"Smoke test script crashed with no output. "
                    f"stderr: {stderr_tail or '<empty>'}"
                ),
            )
            session.iterations.append(record)
            # No point correcting physics params — the script didn't run.
            # Jump straight to UNRESOLVED.
            break

        if smoke_result["returncode"] != 0 and smoke_result.get("json_output") is not None:
            logger.warning(
                "[calibration] Smoke test exited with code %d on iter %d but JSON was parsed "
                "(Blender 5.x background mode quirk — using parsed output)",
                smoke_result["returncode"], iteration,
            )

        validation_file = iter_dir / "validation.json"
        validation_file.write_text(json.dumps(validation, indent=2), encoding="utf-8")

        smoke_passed = validation.get("passed", False)
        completion_ratio = validation.get("completion_ratio", 0.0)
        physics_anomalies = validation.get("physics_anomalies", 0)

        if smoke_passed:
            last_anomalies = physics_anomalies

            # ── Perceptual validation (VLM) ────────────────────────────
            perceptual = await validate_perceptually(
                blend_path, scenario, iter_dir
            )

            if not perceptual.skipped and not perceptual.passed:
                # VLM found visual physics problems — derive failure modes and continue
                perceptual_modes = _perceptual_issues_to_failure_modes(perceptual.issues)
                perceptual_details = "; ".join(
                    f"[{i['category']}] {i['description']}" for i in perceptual.issues
                )
                logger.warning(
                    "[calibration] Perceptual check FAILED on iter %d: %s",
                    iteration, perceptual_details,
                )

                # If smoke test was perfect (100% completion, no anomalies) but the
                # VLM keeps disagreeing, it's likely a VLM rendering false-negative
                # (e.g. curved/complex layouts that qwen3-vl misreads).  After 2
                # consecutive perceptual-only failures, trust the smoke test.
                if completion_ratio >= 1.0 and physics_anomalies == 0:
                    _consecutive_perceptual_only_failures += 1
                else:
                    _consecutive_perceptual_only_failures = 0

                if _consecutive_perceptual_only_failures >= 2:
                    logger.warning(
                        "[calibration] %d consecutive perceptual-only failures with perfect "
                        "smoke test — trusting smoke test and accepting iter %d "
                        "(VLM may be misreading this layout type)",
                        _consecutive_perceptual_only_failures, iteration,
                    )
                    record = IterationRecord(
                        iteration=iteration,
                        params_used=current_params,
                        validation_passed=True,
                        completion_ratio=completion_ratio,
                    )
                    session.iterations.append(record)
                    resolved = True
                    break

                correction = _compute_correction(current_params, perceptual_modes, iteration)
                correction_delta = {
                    k: round(correction[k] - current_params.get(k, 0), 6)
                    for k in correction
                    if correction[k] != current_params.get(k)
                }
                record = IterationRecord(
                    iteration=iteration,
                    params_used=current_params,
                    validation_passed=False,
                    completion_ratio=completion_ratio,
                    failure_modes=perceptual_modes,
                    failure_details=f"[perceptual] {perceptual_details}",
                    correction_applied=correction_delta,
                )
                session.iterations.append(record)
                current_params = correction
                continue

            logger.info(
                "[calibration] Validation PASSED on iter %d (completion=%.0f%%)%s",
                iteration, completion_ratio * 100,
                " [perceptual skipped]" if perceptual.skipped else "",
            )
            record = IterationRecord(
                iteration=iteration,
                params_used=current_params,
                validation_passed=True,
                completion_ratio=completion_ratio,
            )
            session.iterations.append(record)
            resolved = True
            break

        # Parse failure and compute correction
        failure_modes = _parse_failure_modes(validation)
        failure_details = "; ".join(
            c.get("message", "") for c in validation.get("checks", []) if not c.get("passed", True)
        )

        logger.warning(
            "[calibration] Iter %d FAILED: %s (completion=%.0f%%)",
            iteration,
            ", ".join(m.value for m in failure_modes),
            completion_ratio * 100,
        )

        correction = _compute_correction(current_params, failure_modes, iteration)
        correction_delta = {
            k: round(correction[k] - current_params.get(k, 0), 6)
            for k in correction
            if correction[k] != current_params.get(k)
        }

        # ── Stuck-loop breaker ──────────────────────────────────────
        # If the correction delta is empty (params unchanged), the correction
        # engine has exhausted its strategies for this failure mode.  Track
        # consecutive stuck iterations and escalate every 5 failures by
        # broadening the parameter exploration.
        if not correction_delta:
            _consecutive_stuck_failures += 1
            if _consecutive_stuck_failures % 5 == 0:
                logger.warning(
                    "[calibration] Stuck-loop detected on iter %d "
                    "(%d consecutive identical-param failures). "
                    "Escalating to broader parameter exploration.",
                    iteration, _consecutive_stuck_failures,
                )
                # Broaden: spacing, impulse, friction, mass
                correction["spacing_ratio"] = max(
                    0.20, correction.get("spacing_ratio", 0.35) - 0.04
                )
                correction["trigger_impulse"] = min(
                    8.0, correction.get("trigger_impulse", 1.5) + 1.0
                )
                correction["domino_friction"] = min(
                    0.9, correction.get("domino_friction", 0.6) + 0.1
                )
                correction["domino_mass"] = min(
                    1.0, correction.get("domino_mass", 0.3) + 0.1
                )
                correction_delta = {
                    k: round(correction[k] - current_params.get(k, 0), 6)
                    for k in correction
                    if correction[k] != current_params.get(k)
                }
        else:
            _consecutive_stuck_failures = 0

        correction_file = iter_dir / "correction.json"
        correction_file.write_text(
            json.dumps({
                "failure_modes": [m.value for m in failure_modes],
                "delta": correction_delta,
                "new_params": correction,
            }, indent=2),
            encoding="utf-8",
        )

        record = IterationRecord(
            iteration=iteration,
            params_used=current_params,
            validation_passed=False,
            completion_ratio=completion_ratio,
            failure_modes=failure_modes,
            failure_details=failure_details,
            correction_applied=correction_delta,
        )
        session.iterations.append(record)
        current_params = correction

    # ── Step 3: Finalise session ─────────────────────────────────────
    if resolved:
        session.status = CalibrationStatus.RESOLVED
        session.final_corrections = CorrectionFactors.from_absolute(current_params)
    else:
        session.status = CalibrationStatus.UNRESOLVED
        logger.warning(
            "[calibration] Session %s UNRESOLVED after %d iterations",
            sid[:8], session.iteration_count,
        )

    # Write result
    result_file = session_path / "result.yaml"
    result_file.write_text(
        yaml.dump(session.model_dump(mode="json"), default_flow_style=False),
        encoding="utf-8",
    )

    # ── Step 4: Quality gate + promotion ─────────────────────────────
    if resolved:
        last = session.last_iteration
        gate_result = evaluate_quality_gate(
            session,
            chain_completion=last.completion_ratio if last else 0.0,
            physics_anomalies=last_anomalies,
        )

        session.confidence = gate_result.confidence

        if gate_result.passed:
            entry = CalibrationEntry(
                scenario=scenario,
                corrections=session.final_corrections or CorrectionFactors(),
                confidence=gate_result.confidence,
                iteration_count=session.iteration_count,
                blender_version=blender_version,
                calibration_type="resolved",
            )

            # Write calibration extract
            cal_file = session_path / "calibration.yaml"
            cal_file.write_text(
                yaml.dump(entry.model_dump(mode="json"), default_flow_style=False),
                encoding="utf-8",
            )

            if dry_run or knowledge_base is None:
                logger.info(
                    "[calibration] DRY RUN — would have promoted session %s "
                    "(confidence=%.2f, corrections=%s)",
                    sid[:8], gate_result.confidence,
                    entry.corrections.model_dump(),
                )
                session.status = CalibrationStatus.RESOLVED  # not promoted
            else:
                knowledge_base.store(entry)
                session.status = CalibrationStatus.PROMOTED
                logger.info(
                    "[calibration] Session %s PROMOTED to knowledge base (confidence=%.2f)",
                    sid[:8], gate_result.confidence,
                )

            # In dry-run mode keep the .blend files so they can be opened in
            # Blender GUI for visual inspection. Clean up only on real promotions.
            if dry_run:
                iter_base = session_path / "iterations"
                blend_files = sorted(iter_base.glob("*/*.blend")) if iter_base.exists() else []
                if blend_files:
                    logger.info(
                        "[calibration] DRY RUN — open in Blender to visually inspect: %s",
                        blend_files[-1],
                    )
            else:
                iter_base = session_path / "iterations"
                if iter_base.exists():
                    shutil.rmtree(iter_base, ignore_errors=True)

        elif gate_result.requires_human_review:
            logger.info(
                "[calibration] Session %s passed but requires human review: %s",
                sid[:8], gate_result.human_review_reason,
            )
            # Leave artifacts in place for review

        else:
            logger.warning(
                "[calibration] Session %s FAILED quality gate: %s",
                sid[:8],
                "; ".join(c["message"] for c in gate_result.checks if not c.get("passed")),
            )

    # Move unresolved sessions to unresolved/
    if session.status == CalibrationStatus.UNRESOLVED:
        unresolved_dest = _unresolved_dir(sid)
        if session_path.exists() and session_path != unresolved_dest:
            # Move contents
            if unresolved_dest.exists():
                shutil.rmtree(unresolved_dest, ignore_errors=True)
            shutil.copytree(session_path, unresolved_dest, dirs_exist_ok=True)
            shutil.rmtree(session_path, ignore_errors=True)
            logger.info("[calibration] Moved unresolved session to %s", unresolved_dest)

        # Store as negative example if knowledge base available
        if knowledge_base is not None and session.final_corrections is None and not dry_run:
            # Use the best attempt's params as a "don't go here" marker
            best = max(
                session.iterations,
                key=lambda i: i.completion_ratio,
                default=None,
            )
            if best:
                neg_corrections = CorrectionFactors.from_absolute(best.params_used)
                neg_entry = CalibrationEntry(
                    scenario=scenario,
                    corrections=neg_corrections,
                    confidence=0.0,
                    iteration_count=session.iteration_count,
                    blender_version=blender_version,
                    calibration_type="unresolved",
                )
                knowledge_base.store(neg_entry)

    return session
