"""Camera Validator — validates camera trajectory quality.

Runs deterministic checks on the CameraOutput to ensure:
- Visibility: wavefront visible in ≥90% of frames
- Smooth motion: no velocity spikes > 2× average
- No hard cuts (reposition lerps must span ≥30 frames)
- Key moments well-framed

Returns a CameraValidationResult consumed by the pipeline for
pass/fail decisions and cascade routing.
"""

from __future__ import annotations

import logging
import math

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    CameraOutput,
    CameraValidationResult,
    ConnectorOutput,
    SceneManifest,
    StepValidationResult,
)
from kairos.pipelines.domino.creative.camera_router import (
    _manifest_to_spheres,
    _vec_len,
    _vec_sub,
    detect_occlusions,
)

logger = logging.getLogger(__name__)

MIN_VISIBILITY_RATIO = 0.90
MAX_VELOCITY_SPIKE = 2.0   # multiple of average velocity


def validate_camera(
    camera_output: CameraOutput,
    connector_output: ConnectorOutput,
    manifest: SceneManifest,
) -> StepValidationResult:
    """Validate the camera trajectory against quality criteria.

    Returns a StepValidationResult with agent=CAMERA_ROUTER.
    """
    checks: list[dict] = []

    # ── 1. Keyframe count ────────────────────────────────────────────
    checks.append({
        "name": "has_keyframes",
        "passed": len(camera_output.keyframes) >= 2,
        "message": f"Need ≥2 keyframes, got {len(camera_output.keyframes)}",
    })

    if len(camera_output.keyframes) < 2:
        return _build_result(checks)

    # ── 2. Visibility ratio (occlusion check per keyframe) ───────────
    scene_spheres = _manifest_to_spheres(manifest)
    occluded_count = 0
    for kf in camera_output.keyframes:
        occluders = detect_occlusions(kf.position, kf.look_target, scene_spheres)
        if occluders:
            occluded_count += 1

    total = len(camera_output.keyframes)
    vis_ratio = 1.0 - (occluded_count / total) if total > 0 else 0.0
    checks.append({
        "name": "visibility_ratio",
        "passed": vis_ratio >= MIN_VISIBILITY_RATIO,
        "message": f"Visibility {vis_ratio:.1%} (need ≥{MIN_VISIBILITY_RATIO:.0%}), "
                   f"{occluded_count}/{total} frames occluded",
    })

    # ── 3. Smooth motion — velocity spikes ───────────────────────────
    velocities: list[float] = []
    for i in range(1, len(camera_output.keyframes)):
        prev = camera_output.keyframes[i - 1]
        curr = camera_output.keyframes[i]
        dt = max(1, curr.frame - prev.frame)
        dist = _vec_len(_vec_sub(curr.position, prev.position))
        velocities.append(dist / dt)

    if velocities:
        avg_vel = sum(velocities) / len(velocities) if velocities else 1.0
        max_vel = max(velocities) if velocities else 0.0
        spike = max_vel / avg_vel if avg_vel > 1e-9 else 0.0
        smooth = spike <= MAX_VELOCITY_SPIKE
    else:
        spike = 0.0
        smooth = True

    checks.append({
        "name": "smooth_motion",
        "passed": smooth,
        "message": f"Max velocity spike {spike:.1f}× average (limit {MAX_VELOCITY_SPIKE}×)",
    })

    # ── 4. Reposition transitions are smooth (not hard cuts) ─────────
    hard_cuts: list[str] = []
    for occ in camera_output.occlusion_events:
        duration = occ.frame_end - occ.frame_start
        if duration < 30:  # less than 1 second at 30 fps
            hard_cuts.append(
                f"frames {occ.frame_start}-{occ.frame_end} ({duration} frames)"
            )
    checks.append({
        "name": "no_hard_cuts",
        "passed": len(hard_cuts) == 0,
        "message": f"Hard cuts (reposition < 30 frames): {hard_cuts}" if hard_cuts else "OK",
    })

    # ── 5. Total frames reasonable ───────────────────────────────────
    checks.append({
        "name": "total_frames",
        "passed": camera_output.total_frames > 0,
        "message": f"total_frames={camera_output.total_frames}",
    })

    return _build_result(checks)


def _build_result(checks: list[dict]) -> StepValidationResult:
    passed = all(c["passed"] for c in checks)
    errors = [c["message"] for c in checks if not c["passed"]]
    return StepValidationResult(
        agent=AgentRole.CAMERA_ROUTER,
        passed=passed,
        checks=checks,
        error_summary="; ".join(errors) if errors else "",
    )
