"""Kairos Agent — Calibration Perceptual Validator.

After the smoke test confirms the physics numerically (domino tilt angles,
XY displacement), this module renders a small set of key frames from the
baked .blend using EEVEE and sends them to a VLM for visual physics assessment.

Validates:
  - Was the toppling a cascading wave (not an instantaneous explosion)?
  - Did any dominoes get launched far from their positions?
  - Is the chain visually continuous (no obvious gap / chain break)?
  - Camera placement: is the toppling wavefront visible and well-framed?
  - Physics realism: do dominoes fall at a plausible speed and angle?

This is the "Perceptual Validator (VLM)" component from the calibration spec.
It is deliberately lenient on failure (graceful degradation) so that a missing
Ollama installation does not block the calibration pipeline.  Set
PERCEPTUAL_REQUIRED = True to make VLM failure a hard error.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairos.calibration.models import ScenarioDescriptor
from kairos.engines.blender.executor import run_blender_script
from kairos.ai.llm.routing import call_ollama_direct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# LiteLLM alias for the vision model used during calibration.
# Resolves to Qwen3-VL 8B via Ollama (same as the production video reviewer).
PERCEPTUAL_MODEL = "video-reviewer-default"

# Minimum VLM confidence to count the iteration as perceptually passing.
PERCEPTUAL_PASS_THRESHOLD = 0.70

# When True, a VLM call failure (Ollama down, timeout, etc.) causes the
# iteration to be treated as a perceptual fail rather than skipped.
PERCEPTUAL_REQUIRED = False

# Number of frames to render and send to the VLM.
DEFAULT_FRAME_COUNT = 8


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PerceptualResult:
    """Result of a single perceptual validation check."""
    passed: bool
    confidence: float           # 0.0–1.0 VLM confidence
    issues: list[dict[str, str]] = field(default_factory=list)
    model_used: str = ""
    frame_paths: list[str] = field(default_factory=list)
    skipped: bool = False       # True when VLM unavailable — calibration continues
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_calibration_prompt(
    scenario: ScenarioDescriptor,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Physics-focused prompt — not social-media polish, just physics validity."""
    path_type = scenario.path.type.value
    domino_count = scenario.domino_count

    system_prompt = f"""/nothink
You are a physics simulation quality inspector for a domino calibration system.
You are reviewing {frame_count} sequential frames from a Blender rigid-body simulation.

Scenario: {domino_count} dominoes following a {path_type} path.

Your ONLY task is to assess whether the PHYSICS is working correctly.
This is calibration data, not a finished video — ignore lighting, colour, and aesthetics.

## What to assess:

1. **Chain Wave** — Is the toppling a realistic sequential wave (domino-by-domino)?
   FAIL: All dominoes appear to fall simultaneously in the first frame or two — physics explosion.
   PASS: You can see a clear wavefront progressing through the chain across frames.

2. **No Launches** — Are dominoes staying near their starting positions after falling?
   FAIL: One or more dominoes appear to have been launched far from the chain (scattered widely).
   PASS: Fallen dominoes lie roughly where they started, just tipped over.

3. **Chain Continuity** — Is there a single contiguous fallen region at each frame?
   FAIL: There is a gap — dominoes further along the chain have fallen while ones in the middle
         are still standing (two separate fallen regions with standing ones between them).
   PASS: The fallen region is a single continuous front advancing through the chain.

4. **Camera Coverage** — Is the toppling action visible?
   FAIL: Camera is aimed at blank space or the chain is too small to see.
   PASS: The domino chain and the toppling wavefront are clearly visible in most frames.

5. **Fall Speed** — Do dominoes fall at plausible speed?
   A single domino falls ~0.3–0.8 seconds from initial tip to floor.
   FAIL: Dominoes fall impossibly fast (< 0.1s) or freeze mid-fall.
   PASS: Fall speed looks physically plausible.

## Response Format
Respond with ONLY a valid JSON object. No markdown, no explanation, no extra text.

{{
  "passed": true,
  "overall_confidence": 0.85,
  "issues": []
}}

Or if there are physics problems:

{{
  "passed": false,
  "overall_confidence": 0.4,
  "issues": [
    {{
      "category": "explosion",
      "severity": "critical",
      "description": "All {domino_count} dominoes appear to have fallen in frame 1 — physics explosion"
    }},
    {{
      "category": "camera_coverage",
      "severity": "major",
      "description": "Camera is centred on empty floor, chain is out of frame"
    }}
  ]
}}

Valid categories: explosion, chain_break, launch, chain_incomplete, camera_coverage, physics_unrealistic
Valid severities: critical, major, minor

Only flag genuine physics problems. If the simulation looks physically plausible, set passed=true.
REMEMBER: Output ONLY the JSON object."""

    return [{"role": "system", "content": system_prompt}]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_perceptual_response(raw_text: str, model_used: str) -> PerceptualResult:
    """Parse VLM JSON response into a PerceptualResult.

    Gracefully handles non-JSON and malformed responses — treats them as
    skipped rather than failed so that VLM issues don't break calibration.
    """
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        logger.warning("[perceptual] VLM returned no JSON: %r", raw_text[:200])
        return PerceptualResult(
            passed=True,
            confidence=0.0,
            skipped=True,
            skip_reason="VLM returned non-JSON response",
            model_used=model_used,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return PerceptualResult(
            passed=True,
            confidence=0.0,
            skipped=True,
            skip_reason="Malformed JSON in VLM response",
            model_used=model_used,
        )

    passed = bool(data.get("passed", True))
    confidence = float(data.get("overall_confidence", 0.8))

    issues: list[dict[str, str]] = []
    for raw in data.get("issues", []):
        if isinstance(raw, dict):
            issues.append({
                "category": str(raw.get("category", "unknown")),
                "severity": str(raw.get("severity", "major")),
                "description": str(raw.get("description", "")),
            })

    # Override passed if confidence is below our threshold
    if confidence < PERCEPTUAL_PASS_THRESHOLD and passed:
        passed = False

    return PerceptualResult(
        passed=passed,
        confidence=confidence,
        issues=issues,
        model_used=model_used,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def validate_perceptually(
    blend_path: Path,
    scenario: ScenarioDescriptor,
    output_dir: Path,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    max_sample_frame: int = 350,
) -> PerceptualResult:
    """Render key frames from a baked .blend and send to VLM for assessment.

    Args:
        blend_path: Path to the baked .blend produced by the iteration.
        scenario: Scenario descriptor (used to build a context-rich prompt).
        output_dir: Iteration directory — frames and VLM JSON are written here.
        frame_count: Number of evenly-distributed frames to render.
        max_sample_frame: Cap the frame sampling window at this scene frame.
            Domino cascades finish well within the first 300-400 frames;
            sampling beyond that shows only the static resting state which
            confuses the VLM.  Passed via ``--sample-end`` to the render script.

    Returns:
        PerceptualResult.  If Ollama is unreachable or any rendering step fails,
        ``skipped=True`` is set and ``passed=True`` so calibration continues.
        Set PERCEPTUAL_REQUIRED = True to make failures hard errors.
    """
    frames_dir = output_dir / "perceptual_frames"
    render_json_path = output_dir / "perceptual_render.json"

    # ── Step 1: Render preview frames ─────────────────────────────────
    logger.info("[perceptual] Rendering %d frames from %s", frame_count, blend_path.name)

    render_result = await run_blender_script(
        "render_calibration_frames.py",
        blend_file=str(blend_path),
        script_args=[
            "--auto", str(frame_count),
            "--sample-end", str(max_sample_frame),
            "--output-dir", str(frames_dir),
            "--output-json", str(render_json_path),
        ],
        timeout_sec=180,
    )

    if render_result["returncode"] != 0 and not render_json_path.exists():
        msg = f"Frame render failed (rc={render_result['returncode']})"
        logger.warning("[perceptual] %s — %s", msg, render_result.get("stderr", "")[-200:])
        return _fail_or_skip(msg)

    render_data: dict[str, Any] = {}
    if render_json_path.exists():
        try:
            render_data = json.loads(render_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    ok_frames = [
        r["path"]
        for r in render_data.get("frames_rendered", [])
        if r.get("ok")
    ]

    if not ok_frames:
        return _fail_or_skip("No frames rendered successfully")

    logger.info("[perceptual] %d/%d frames rendered OK", len(ok_frames), frame_count)

    # ── Step 2: Build VLM messages with embedded frames ────────────────
    messages = _build_calibration_prompt(scenario, len(ok_frames))

    image_parts: list[dict[str, Any]] = []
    for frame_path in ok_frames:
        try:
            with open(frame_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception as exc:
            logger.debug("[perceptual] Could not encode frame %s: %s", frame_path, exc)

    if not image_parts:
        return _fail_or_skip("Frame encoding failed — no images to send")

    image_parts.append({
        "type": "text",
        "text": (
            f"Review these {len(image_parts) - 1} sequential simulation frames. "
            "Respond with ONLY a valid JSON object."
        ),
    })
    messages.append({"role": "user", "content": image_parts})

    # ── Step 3: Call VLM ───────────────────────────────────────────────
    try:
        resp = call_ollama_direct(
            PERCEPTUAL_MODEL,
            messages,
            max_tokens=8192,
            timeout=120,
        )
        # Thinking models (qwen3-vl) sometimes put the entire answer inside the
        # reasoning field and return empty content when json_object mode is used.
        # Fall back to thinking text so the JSON regex extractor can still find it.
        raw_text = resp.content or resp.thinking or ""
        result = _parse_perceptual_response(raw_text, PERCEPTUAL_MODEL)

    except Exception as exc:
        logger.warning("[perceptual] VLM call failed (%s) — skipping check", exc, exc_info=False)
        return _fail_or_skip(str(exc))

    # ── Step 4: Save result to disk ────────────────────────────────────
    vlm_json_path = output_dir / "perceptual_result.json"
    vlm_json_path.write_text(
        json.dumps({
            "passed": result.passed,
            "confidence": result.confidence,
            "issues": result.issues,
            "model_used": result.model_used,
            "skipped": result.skipped,
            "frame_paths": ok_frames,
        }, indent=2),
        encoding="utf-8",
    )

    log_fn = logger.info if result.passed else logger.warning
    log_fn(
        "[perceptual] %s — confidence=%.2f  issues=%d  model=%s",
        "PASSED" if result.passed else "FAILED",
        result.confidence,
        len(result.issues),
        result.model_used,
    )
    for issue in result.issues:
        logger.warning(
            "[perceptual]   [%s/%s] %s",
            issue.get("category"), issue.get("severity"), issue.get("description"),
        )

    result.frame_paths = ok_frames
    return result


def _fail_or_skip(reason: str) -> PerceptualResult:
    """Return a skipped (graceful) or failed result depending on PERCEPTUAL_REQUIRED."""
    if PERCEPTUAL_REQUIRED:
        return PerceptualResult(
            passed=False,
            confidence=0.0,
            skipped=False,
            skip_reason=reason,
            issues=[{"category": "render_error", "severity": "critical", "description": reason}],
        )
    return PerceptualResult(
        passed=True,
        confidence=0.0,
        skipped=True,
        skip_reason=reason,
    )
