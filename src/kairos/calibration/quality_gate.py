"""Kairos Agent — Calibration Quality Gate.

Automated checks + confidence scoring + human review flagging.
Only calibrations that pass the quality gate get promoted to
the ChromaDB knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CalibrationEntry,
    CalibrationSession,
    CalibrationStatus,
    CorrectionFactors,
    QualityGateResult,
)

logger = logging.getLogger(__name__)

# ── Gate thresholds ──────────────────────────────────────────────────────

# Hard requirements — all must pass for promotion
CHAIN_COMPLETION_THRESHOLD = 1.0      # 100% of dominoes must topple
MAX_PHYSICS_ANOMALIES = 0             # Zero floating/clipping/explosion events
PERCEPTUAL_SCORE_THRESHOLD = 0.8      # VLM confidence in physics correctness

# Soft requirements — affect confidence, don't block promotion
IDEAL_ITERATION_COUNT = 7             # Fewer iterations = higher confidence
MAX_DEVIATION_FROM_BASELINE = 3.0     # No correction factor >3× baseline

# Human review triggers
AUTO_APPROVE_AFTER = 20  # After N successful reviews of an archetype, auto-approve


def compute_confidence(session: CalibrationSession) -> float:
    """Compute confidence score (0.0–1.0) for a calibration session.

    Higher scores indicate more trustworthy calibrations.
    """
    score = 1.0

    # Penalise high iteration count (many tries = less reliable)
    if session.iteration_count > 5:
        score -= 0.1 * (session.iteration_count - 5)

    # Penalise large deviation from baseline
    if session.final_corrections:
        max_dev = _max_deviation(session.final_corrections)
        if max_dev > 1.5:
            score -= 0.1 * (max_dev - 1.5)

    # Boost if converged quickly
    if session.iteration_count <= 2:
        score += 0.1

    return max(0.0, min(1.0, round(score, 3)))


def _max_deviation(corrections: CorrectionFactors) -> float:
    """Compute the maximum deviation of any correction factor from 1.0."""
    max_dev = 0.0
    for field_name in CorrectionFactors.model_fields:
        if field_name == "notes":
            continue
        val = getattr(corrections, field_name)
        dev = abs(val - 1.0)
        max_dev = max(max_dev, dev)
    return round(max_dev, 4)


def evaluate(
    session: CalibrationSession,
    *,
    chain_completion: float,
    physics_anomalies: int,
    perceptual_score: float = 0.0,
    archetype_review_count: int = 0,
) -> QualityGateResult:
    """Evaluate a completed calibration session against the quality gate.

    Args:
        session: The completed calibration session.
        chain_completion: Fraction of dominoes that toppled (0.0–1.0).
        physics_anomalies: Count of physics anomaly events.
        perceptual_score: VLM confidence score (0.0–1.0). Pass 0.0 to skip.
        archetype_review_count: How many calibrations of this archetype
            have been reviewed previously.

    Returns:
        QualityGateResult with pass/fail, confidence, and review flags.
    """
    checks: list[dict[str, Any]] = []
    passed = True

    # Hard check 1: Chain completion
    cc_ok = chain_completion >= CHAIN_COMPLETION_THRESHOLD
    checks.append({
        "name": "chain_completion",
        "passed": cc_ok,
        "value": chain_completion,
        "threshold": CHAIN_COMPLETION_THRESHOLD,
        "message": f"Chain completion: {chain_completion:.0%}"
                   + ("" if cc_ok else f" (need ≥{CHAIN_COMPLETION_THRESHOLD:.0%})"),
    })
    if not cc_ok:
        passed = False

    # Hard check 2: Physics anomalies
    pa_ok = physics_anomalies <= MAX_PHYSICS_ANOMALIES
    checks.append({
        "name": "physics_anomalies",
        "passed": pa_ok,
        "value": physics_anomalies,
        "threshold": MAX_PHYSICS_ANOMALIES,
        "message": f"{physics_anomalies} anomalies"
                   + ("" if pa_ok else f" (need ≤{MAX_PHYSICS_ANOMALIES})"),
    })
    if not pa_ok:
        passed = False

    # Hard check 3: Perceptual score (if provided)
    if perceptual_score > 0:
        ps_ok = perceptual_score >= PERCEPTUAL_SCORE_THRESHOLD
        checks.append({
            "name": "perceptual_score",
            "passed": ps_ok,
            "value": perceptual_score,
            "threshold": PERCEPTUAL_SCORE_THRESHOLD,
            "message": f"Perceptual score: {perceptual_score:.2f}"
                       + ("" if ps_ok else f" (need ≥{PERCEPTUAL_SCORE_THRESHOLD})"),
        })
        if not ps_ok:
            passed = False

    # Compute confidence
    confidence = compute_confidence(session)

    # Boost confidence if perceptual validation was strong
    if perceptual_score > 0.95:
        confidence = min(1.0, confidence + 0.1)

    # Compute max deviation
    max_dev = 0.0
    if session.final_corrections:
        max_dev = _max_deviation(session.final_corrections)

    # Soft check: Parameter sanity
    dev_ok = max_dev <= MAX_DEVIATION_FROM_BASELINE
    checks.append({
        "name": "parameter_sanity",
        "passed": dev_ok,
        "value": max_dev,
        "threshold": MAX_DEVIATION_FROM_BASELINE,
        "message": f"Max deviation from baseline: {max_dev:.2f}×"
                   + ("" if dev_ok else " (flagged for review)"),
        "advisory": True,  # Doesn't block promotion
    })
    if not dev_ok:
        confidence = max(0.0, confidence - 0.2)

    # Soft check: Iteration count
    iter_ok = session.iteration_count <= IDEAL_ITERATION_COUNT
    checks.append({
        "name": "iteration_count",
        "passed": iter_ok,
        "value": session.iteration_count,
        "threshold": IDEAL_ITERATION_COUNT,
        "message": f"Converged in {session.iteration_count} iterations"
                   + ("" if iter_ok else f" (ideal ≤{IDEAL_ITERATION_COUNT})"),
        "advisory": True,
    })

    # Human review triggers
    requires_review = False
    review_reason = ""

    if archetype_review_count < AUTO_APPROVE_AFTER:
        requires_review = True
        review_reason = f"First calibrations for archetype (reviewed {archetype_review_count}/{AUTO_APPROVE_AFTER})"

    if confidence < 0.6:
        requires_review = True
        review_reason = f"Low confidence: {confidence:.2f}"

    if max_dev > 2.0:
        requires_review = True
        review_reason = f"High parameter deviation: {max_dev:.2f}×"

    return QualityGateResult(
        passed=passed,
        chain_completion=chain_completion,
        physics_anomalies=physics_anomalies,
        perceptual_score=perceptual_score,
        confidence=confidence,
        iteration_count=session.iteration_count,
        param_deviation_from_baseline=max_dev,
        requires_human_review=requires_review,
        human_review_reason=review_reason,
        checks=checks,
    )
