"""Kairos Agent — Screenshot Analyzer (Tier 2 Validation).

Extracts frames from rendered video at key timestamps and analyses
them using a local vision LLM (Moondream2 via Ollama) to verify
simulation quality.

Analysis timestamps:
  - Early:  first 2 seconds (frames showing marble release)
  - Mid:    midway through (marbles traversing the course)
  - Late:   last 3 seconds (marbles reaching bins / finale)

Each frame is sent to the vision model with a targeted prompt.
Results are returned as structured ValidationChecks.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import litellm

from kairos.config import get_settings
from kairos.schemas.contracts import ValidationCheck
from kairos.ai.llm.config import get_step_config, get_ollama_base_url

logger = logging.getLogger(__name__)


def _get_ffmpeg_path() -> str:
    """Get the resolved FFmpeg path from centralised config."""
    return get_settings().ffmpeg_path


def _get_ffprobe_path() -> str:
    """Get the resolved FFprobe path from centralised config."""
    return get_settings().ffprobe_path

# Frame extraction timestamps (seconds from start)
# "first 2 sec, midway, last 3 sec"  — relative to video duration.
SAMPLE_POINTS = {
    "early": 2.0,       # 2 seconds in — marbles should be visible
    "mid": 0.5,         # 50% of duration — marbles traversing course
    "late_offset": 3.0, # 3 seconds before end — finale / bins
}

# Vision prompt — Moondream is a captioning model that only responds
# to descriptive "What/Describe" prompts.  Yes/no questions and structured
# instructions (numbered lists, "you MUST", "Rating: N/10") cause it to
# emit a single EOS token and return empty text.  We therefore ask for a
# plain description and derive quality from keyword analysis.
_VISION_PROMPT = "Describe this image."

STAGE_PROMPTS = {
    "early": _VISION_PROMPT,
    "mid":   _VISION_PROMPT,
    "late":  _VISION_PROMPT,
}


def _extract_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    """Extract a single frame from a video at the given timestamp.

    Uses ffmpeg to seek to the timestamp and extract one PNG frame.
    Returns True if extraction succeeded.
    """
    cmd = [
        _get_ffmpeg_path(), "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and Path(output_path).exists()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Frame extraction failed at %.1fs: %s", timestamp_sec, e)
        return False


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        _get_ffprobe_path(), "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


def _image_to_base64(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vision_model(image_b64: str, prompt: str) -> str:
    """Call the vision LLM (Moondream2) with an image and prompt.

    Uses litellm to route through the configured frame-inspector alias.
    Falls back to direct ollama/moondream call if alias resolution fails.
    """
    # Use direct Ollama /api/generate — more reliable than litellm for
    # vision models (avoids OpenAI-style message-format conversion issues).
    import requests as _requests

    ollama_url = get_ollama_base_url().rstrip("/") + "/api/generate"
    payload = {
        "model": "moondream:latest",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    logger.debug("[vision] POST %s  prompt=%r  image_len=%d",
                 ollama_url, prompt, len(image_b64))

    try:
        r = _requests.post(ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        body = r.json()
        content = body.get("response", "")
        eval_count = body.get("eval_count", "?")
        logger.debug("[vision] eval_count=%s  content_len=%d  preview=%.200s",
                     eval_count, len(content), content)
        return content
    except Exception as e:
        logger.error("Vision model call failed: %s", e)
        return f"ERROR: {e}"


def _check_blank_frame(image_path: str) -> str | None:
    """Fast pixel-level check for blank / solid-colour frames.

    Converts the PNG to raw greyscale via ffmpeg and computes basic
    statistics (mean, min, max) in pure Python.  No heavy deps needed.
    Returns a failure message string if the frame is blank, or None if OK.
    """
    try:
        data = Path(image_path).read_bytes()
        # Very small PNG = likely solid colour
        if len(data) < 2000:
            return (
                "FAIL: Extracted frame is extremely small "
                f"({len(data)} bytes) — likely a solid-colour image."
            )

        # Convert frame → raw 8-bit greyscale via ffmpeg stdout
        cmd = [
            _get_ffmpeg_path(), "-y", "-v", "quiet",
            "-i", image_path,
            "-pix_fmt", "gray",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                pixels = result.stdout
                y_avg = sum(pixels) / len(pixels)
                y_min = min(pixels)
                y_max = max(pixels)
                dyn_range = y_max - y_min

                if dyn_range < 10:
                    return (
                        f"FAIL: Frame is a solid colour "
                        f"(Y range {y_min}–{y_max}, avg {y_avg:.0f})."
                    )
                if y_avg < 15 and dyn_range < 30:
                    return (
                        f"FAIL: Frame is nearly all black "
                        f"(Y avg {y_avg:.0f}, range {dyn_range})."
                    )
                if y_avg > 240 and dyn_range < 30:
                    return (
                        f"FAIL: Frame is nearly all white "
                        f"(Y avg {y_avg:.0f}, range {dyn_range})."
                    )
        except Exception:
            pass  # pixel check is best-effort; fall through to vision model

    except Exception as e:
        logger.debug("Blank-frame check error: %s", e)

    return None


def analyze_frame(
    video_path: str,
    timestamp_sec: float,
    stage: str,
) -> dict[str, Any]:
    """Extract and analyze a single frame from the video.

    Args:
        video_path: Path to the rendered MP4.
        timestamp_sec: Timestamp in seconds to extract.
        stage: One of "early", "mid", "late" — determines the prompt.

    Returns:
        Dict with keys: stage, timestamp, analysis, quality_rating, passed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = str(Path(tmp) / f"frame_{stage}.png")

        if not _extract_frame(video_path, timestamp_sec, frame_path):
            return {
                "stage": stage,
                "timestamp": timestamp_sec,
                "analysis": "Failed to extract frame",
                "quality_rating": 0,
                "passed": False,
            }

        # ── Fast pixel pre-check: catch solid-colour / blank frames ──
        blank_result = _check_blank_frame(frame_path)
        if blank_result is not None:
            return {
                "stage": stage,
                "timestamp": timestamp_sec,
                "analysis": blank_result,
                "quality_rating": 1,
                "passed": False,
            }

        image_b64 = _image_to_base64(frame_path)
        prompt = STAGE_PROMPTS.get(stage, STAGE_PROMPTS["mid"])
        analysis = _call_vision_model(image_b64, prompt)

        # Try to extract a numeric quality rating from the response
        quality_rating = _extract_rating(analysis)

        # Frame passes if it has content (not all-black) and rating >= 4
        passed = quality_rating >= 4 and "ERROR" not in analysis

        return {
            "stage": stage,
            "timestamp": timestamp_sec,
            "analysis": analysis,
            "quality_rating": quality_rating,
            "passed": passed,
        }


# ── Keyword-based quality assessment ────────────────────────────
# Moondream returns a free-text description.  We score it by counting
# positive and negative signal words rather than asking for a rating.

_POSITIVE_SIGNALS = [
    # Scene elements we WANT to see
    "marble", "sphere", "ball", "orb",
    "ramp", "slope", "incline", "slide",
    "course", "track", "path", "lane",
    "wall", "barrier", "rail", "fence",
    "peg", "pin", "obstacle",
    "funnel", "chute", "tube",
    "bin", "container", "basket",
    "gate", "start", "finish",
    # Appearance / quality
    "color", "colour", "colorf", "bright", "vivid", "neon",
    "3d", "render", "scene", "lighting", "lit",
    "depth", "shadow", "reflection",
    "moving", "motion", "rolling", "falling", "bouncing",
]

_NEGATIVE_SIGNALS = [
    # Failure indicators
    "blank", "empty", "nothing", "void",
    "solid color", "solid colour", "single color", "single colour",
    "all black", "all white", "all grey", "all gray",
    "mostly black", "mostly grey", "mostly gray",
    "too dark", "pitch black", "completely dark",
    "no visible", "cannot see", "can't see",
    "upside down", "upside-down", "sideways",
    "broken", "corrupted", "error",
    "featureless", "uniform",
]


def _extract_rating(text: str) -> int:
    """Derive a 1-10 quality rating from a free-text description.

    Moondream is a captioning model — it does not follow instructions to
    output 'Rating: N/10'.  Instead we count positive & negative keyword
    hits in the description and map to a score.
    """
    if not text or "ERROR" in text:
        return 0

    lower = text.lower()

    # First: check for an explicit numeric rating (in case a future model
    # does include one)
    import re
    for pat in [
        r"(\d{1,2})\s*/\s*10",
        r"[Rr]at(?:e|ing)[:\s]+(\d{1,2})",
        r"(\d{1,2})\s+out\s+of\s+10",
    ]:
        m = re.search(pat, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val

    # Count keyword hits
    pos = sum(1 for kw in _POSITIVE_SIGNALS if kw in lower)
    neg = sum(1 for kw in _NEGATIVE_SIGNALS if kw in lower)

    # Very short description (< 30 chars) is suspicious
    if len(text) < 30:
        return max(1, min(3, pos))

    # Strong negative signal → low score
    if neg >= 3:
        return 1
    if neg >= 2:
        return 2
    if neg >= 1 and pos < 2:
        return 3

    # Score by positive hits
    if pos >= 6:
        return 8
    if pos >= 4:
        return 7
    if pos >= 3:
        return 6
    if pos >= 2:
        return 5
    if pos >= 1:
        return 4

    # Non-empty text with no identifiable keywords → mediocre
    return 4 if len(text) > 50 else 3


def analyze_video(video_path: str) -> list[dict[str, Any]]:
    """Analyze a video at three key timestamps.

    Extracts frames at:
      - 2 seconds (early — marble release)
      - midpoint (marbles traversing)
      - 3 seconds before end (late — finale)

    Returns list of analysis dicts, one per stage.
    """
    duration = _get_video_duration(video_path)
    if duration <= 0:
        logger.error("Could not determine video duration for %s", video_path)
        return [{
            "stage": "error",
            "timestamp": 0,
            "analysis": "Could not determine video duration",
            "quality_rating": 0,
            "passed": False,
        }]

    timestamps = {
        "early": min(SAMPLE_POINTS["early"], duration * 0.1),
        "mid": duration * SAMPLE_POINTS["mid"],
        "late": max(0, duration - SAMPLE_POINTS["late_offset"]),
    }

    results = []
    for stage, ts in timestamps.items():
        logger.info(
            "[screenshot_analyzer] Analyzing %s frame at %.1fs / %.1fs",
            stage, ts, duration,
        )
        result = analyze_frame(video_path, ts, stage)
        results.append(result)

        logger.info(
            "[screenshot_analyzer] %s (%.1fs): quality=%d/10 passed=%s",
            stage, ts, result["quality_rating"], result["passed"],
        )
        logger.info(
            "[screenshot_analyzer] %s analysis: %.200s",
            stage, result["analysis"],
        )

    return results


def analyze_to_validation_checks(video_path: str) -> list[ValidationCheck]:
    """Run full screenshot analysis and return ValidationChecks.

    This is the entry point for Tier 2 validation integration.
    """
    results = analyze_video(video_path)

    checks: list[ValidationCheck] = []
    for r in results:
        checks.append(
            ValidationCheck(
                name=f"ai_frame_{r['stage']}",
                passed=r["passed"],
                message=(
                    f"[{r['stage']} @ {r['timestamp']:.1f}s] "
                    f"Quality: {r['quality_rating']}/10 — "
                    f"{r['analysis'][:200]}"
                ),
                value=r["quality_rating"],
                threshold=4,
            )
        )

    # Overall check: at least 2 of 3 stages must pass
    stage_passes = sum(1 for r in results if r["passed"])
    checks.append(
        ValidationCheck(
            name="ai_overall_quality",
            passed=stage_passes >= 2,
            message=f"AI visual check: {stage_passes}/3 stages passed",
            value=stage_passes,
            threshold=2,
        )
    )

    return checks
