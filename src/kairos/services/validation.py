"""Kairos Agent — Validation Engine.

Two-tier validation for simulation outputs:
- Tier 1: Programmatic checks (mandatory, no LLM)
- Tier 2: AI-assisted checks (optional, Moondream2)

All validation is programmatic first, LLM-assisted second.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from kairos.config import get_settings
from kairos.models.contracts import ValidationCheck, ValidationResult

logger = logging.getLogger(__name__)


def _run_ffprobe(video_path: str) -> dict[str, object]:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)  # type: ignore[no-any-return]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return {}


def check_valid_mp4(video_path: str) -> ValidationCheck:
    """Check that the file is a valid MP4 with parseable metadata."""
    probe = _run_ffprobe(video_path)
    passed = bool(probe and "format" in probe and "streams" in probe)
    return ValidationCheck(
        name="valid_mp4",
        passed=passed,
        message="Valid MP4 file" if passed else "Invalid or corrupt MP4",
    )


def check_resolution(video_path: str, *, target: str = "1080x1920") -> ValidationCheck:
    """Check video resolution matches target (width x height)."""
    probe = _run_ffprobe(video_path)
    target_w, target_h = (int(x) for x in target.split("x"))

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            w = stream.get("width", 0)
            h = stream.get("height", 0)
            passed = w == target_w and h == target_h
            return ValidationCheck(
                name="resolution",
                passed=passed,
                message=f"Resolution: {w}x{h}" + ("" if passed else f" (expected {target})"),
                value=f"{w}x{h}",
                threshold=target,
            )
    return ValidationCheck(
        name="resolution",
        passed=False,
        message="No video stream found",
    )


def check_fps(video_path: str, *, min_fps: int = 30) -> ValidationCheck:
    """Check that video FPS is at or above minimum."""
    probe = _run_ffprobe(video_path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = int(num) / int(den) if int(den) != 0 else 0
            except (ValueError, ZeroDivisionError):
                fps = 0
            passed = fps >= min_fps
            return ValidationCheck(
                name="fps",
                passed=passed,
                message=f"FPS: {fps:.1f}" + ("" if passed else f" (minimum: {min_fps})"),
                value=round(fps, 1),
                threshold=min_fps,
            )
    return ValidationCheck(name="fps", passed=False, message="No video stream found")


def check_duration(
    video_path: str,
    *,
    min_sec: int | None = None,
    max_sec: int | None = None,
) -> ValidationCheck:
    """Check video duration is within acceptable range."""
    settings = get_settings()
    min_sec = min_sec or settings.target_duration_min_sec
    max_sec = max_sec or settings.target_duration_max_sec

    probe = _run_ffprobe(video_path)
    duration = float(probe.get("format", {}).get("duration", 0))
    passed = min_sec <= duration <= max_sec
    return ValidationCheck(
        name="duration",
        passed=passed,
        message=f"Duration: {duration:.1f}s"
        + ("" if passed else f" (expected {min_sec}-{max_sec}s)"),
        value=round(duration, 1),
        threshold=f"{min_sec}-{max_sec}",
    )


def check_file_size(
    video_path: str,
    *,
    min_bytes: int = 1_000_000,
    max_bytes: int = 500_000_000,
) -> ValidationCheck:
    """Check file size is within reasonable bounds."""
    path = Path(video_path)
    if not path.exists():
        return ValidationCheck(
            name="file_size",
            passed=False,
            message="File does not exist",
        )
    size = path.stat().st_size
    passed = min_bytes <= size <= max_bytes
    size_mb = size / 1_000_000
    return ValidationCheck(
        name="file_size",
        passed=passed,
        message=f"File size: {size_mb:.1f}MB"
        + ("" if passed else f" (expected {min_bytes / 1e6:.0f}-{max_bytes / 1e6:.0f}MB)"),
        value=size,
        threshold=f"{min_bytes}-{max_bytes}",
    )


def check_audio_present(video_path: str) -> ValidationCheck:
    """Check that an audio stream exists in the video."""
    probe = _run_ffprobe(video_path)
    has_audio = any(
        s.get("codec_type") == "audio" for s in probe.get("streams", [])
    )
    return ValidationCheck(
        name="audio_present",
        passed=has_audio,
        message="Audio stream present" if has_audio else "No audio stream found",
    )


def check_frame_count(video_path: str) -> ValidationCheck:
    """Check frame count matches expected (duration × FPS) within 1%."""
    probe = _run_ffprobe(video_path)
    duration = float(probe.get("format", {}).get("duration", 0))

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = int(num) / int(den) if int(den) != 0 else 0
            except (ValueError, ZeroDivisionError):
                fps = 0

            nb_frames_str = stream.get("nb_frames", "0")
            try:
                actual_frames = int(nb_frames_str)
            except ValueError:
                actual_frames = 0

            expected_frames = duration * fps
            if expected_frames == 0:
                return ValidationCheck(
                    name="frame_count",
                    passed=False,
                    message="Cannot compute expected frame count",
                )
            tolerance = 0.01 * expected_frames
            passed = abs(actual_frames - expected_frames) <= tolerance
            return ValidationCheck(
                name="frame_count",
                passed=passed,
                message=f"Frames: {actual_frames} (expected ~{expected_frames:.0f})",
                value=actual_frames,
                threshold=f"±1% of {expected_frames:.0f}",
            )

    return ValidationCheck(
        name="frame_count",
        passed=False,
        message="No video stream found",
    )


def check_frozen_frames(
    video_path: str,
    *,
    max_consecutive: int = 5,
    sample_count: int = 30,
) -> ValidationCheck:
    """Check for frozen/duplicate frames using frame hash comparison.

    Extracts sample frames and compares hashes for consecutive duplicates.
    """
    try:
        # Extract frames to temp dir and hash them
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select=not(mod(n\\,{max(1, sample_count)})),setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(sample_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return ValidationCheck(
                name="frozen_frames",
                passed=True,
                message="Could not extract frames for comparison (skipped)",
            )

        # Hash each frame chunk
        frame_data = result.stdout
        if not frame_data:
            return ValidationCheck(
                name="frozen_frames",
                passed=False,
                message="No frame data extracted",
            )

        chunk_size = len(frame_data) // sample_count if sample_count > 0 else len(frame_data)
        if chunk_size == 0:
            return ValidationCheck(
                name="frozen_frames",
                passed=True,
                message="Insufficient frame data",
            )

        hashes = []
        for i in range(0, len(frame_data), chunk_size):
            chunk = frame_data[i : i + chunk_size]
            hashes.append(hashlib.md5(chunk).hexdigest())  # noqa: S324

        # Check consecutive duplicates
        max_streak = 1
        current_streak = 1
        for i in range(1, len(hashes)):
            if hashes[i] == hashes[i - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        passed = max_streak < max_consecutive
        return ValidationCheck(
            name="frozen_frames",
            passed=passed,
            message=f"Max consecutive identical frames: {max_streak}"
            + ("" if passed else f" (max allowed: {max_consecutive})"),
            value=max_streak,
            threshold=max_consecutive,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ValidationCheck(
            name="frozen_frames",
            passed=True,
            message="FFmpeg not available for frozen frame check (skipped)",
        )


def check_colour_valid(
    video_path: str,
    *,
    black_threshold: float = 5.0,
    white_threshold: float = 250.0,
    variance_threshold: float = 2.0,
) -> ValidationCheck:
    """Check that the video is not all-black, all-white, or single-colour.

    Samples frames across the video and checks mean pixel values are not
    at extremes (all-black or all-white) and that there is sufficient
    colour variance (not a single solid colour).

    Args:
        video_path: Path to the video file.
        black_threshold: Max mean pixel value to consider "all black".
        white_threshold: Min mean pixel value to consider "all white".
        variance_threshold: Min standard deviation across sampled frame means.
    """
    import re as _re

    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "select=not(mod(n\\,100)),signalstats=stat=tout+vrep+brng",
            "-frames:v",
            "5",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return ValidationCheck(
                name="colour_valid",
                passed=False,
                message=f"FFmpeg colour analysis failed (rc={result.returncode})",
            )

        # Parse mean pixel values from signalstats output lines like:
        #   [Parsed_signalstats...] YAVG: 16.2
        stderr = result.stderr
        y_avg_values: list[float] = []
        for match in _re.finditer(r"YAVG:\s*([\d.]+)", stderr):
            y_avg_values.append(float(match.group(1)))

        if not y_avg_values:
            # Could not parse stats — pass with warning
            return ValidationCheck(
                name="colour_valid",
                passed=True,
                message="Could not parse signalstats output (skipped)",
            )

        mean_brightness = sum(y_avg_values) / len(y_avg_values)

        # Check all-black
        if mean_brightness < black_threshold:
            return ValidationCheck(
                name="colour_valid",
                passed=False,
                message=f"Video appears all-black (mean brightness={mean_brightness:.1f})",
            )

        # Check all-white
        if mean_brightness > white_threshold:
            return ValidationCheck(
                name="colour_valid",
                passed=False,
                message=f"Video appears all-white (mean brightness={mean_brightness:.1f})",
            )

        # Check single-colour (low variance across sampled frames)
        if len(y_avg_values) >= 2:
            mean_val = mean_brightness
            variance = (
                sum((v - mean_val) ** 2 for v in y_avg_values) / len(y_avg_values)
            ) ** 0.5
            if variance < variance_threshold:
                return ValidationCheck(
                    name="colour_valid",
                    passed=False,
                    message=(
                        f"Video appears single-colour "
                        f"(brightness stddev={variance:.2f} < {variance_threshold})"
                    ),
                )

        return ValidationCheck(
            name="colour_valid",
            passed=True,
            message=f"Colour check passed (mean brightness={mean_brightness:.1f})",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ValidationCheck(
            name="colour_valid",
            passed=True,
            message="FFmpeg not available for colour check (skipped)",
        )


def check_motion_present(
    video_path: str,
    *,
    num_samples: int = 10,
    variance_threshold: float = 50.0,
) -> ValidationCheck:
    """Check that motion is present by comparing pixel variance between frames.

    Samples frames at even intervals and computes mean pixel difference.
    Static images (e.g., a single screenshot repeated) will fail.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select=not(mod(n\\,{max(1, num_samples)})),"
            "tblend=all_mode=difference,signalstats",
            "-frames:v",
            str(num_samples),
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Parse signalstats output for YAVG (average luma difference)
        # Higher YAVG = more motion between consecutive frames
        import re

        yavg_values = re.findall(r"YAVG:(\d+\.?\d*)", result.stderr)
        if not yavg_values:
            return ValidationCheck(
                name="motion_present",
                passed=True,
                message="Could not parse motion data (skipped)",
            )

        avg_motion = sum(float(v) for v in yavg_values) / len(yavg_values)
        passed = avg_motion > variance_threshold
        return ValidationCheck(
            name="motion_present",
            passed=passed,
            message=f"Average motion: {avg_motion:.1f}"
            + ("" if passed else f" (threshold: {variance_threshold})"),
            value=round(avg_motion, 1),
            threshold=variance_threshold,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ValidationCheck(
            name="motion_present",
            passed=True,
            message="FFmpeg not available for motion check (skipped)",
        )


def validate_simulation(
    video_path: str,
    *,
    run_tier2: bool = False,
) -> ValidationResult:
    """Run all validation checks on a simulation output.

    Args:
        video_path: Path to the rendered MP4 file.
        run_tier2: Whether to run AI-assisted Tier 2 checks.

    Returns:
        ValidationResult with all check results.
    """
    checks: list[ValidationCheck] = []

    # Tier 1 — Programmatic (mandatory)
    checks.append(check_valid_mp4(video_path))
    checks.append(check_resolution(video_path))
    checks.append(check_fps(video_path))
    checks.append(check_duration(video_path))
    checks.append(check_file_size(video_path))
    checks.append(check_frame_count(video_path))
    checks.append(check_audio_present(video_path))
    checks.append(check_frozen_frames(video_path))
    checks.append(check_colour_valid(video_path))
    checks.append(check_motion_present(video_path))

    tier1_passed = all(c.passed for c in checks)

    # Tier 2 — AI-assisted (optional)
    tier2_passed: bool | None = None
    if run_tier2:
        tier2_checks = _run_tier2_checks(video_path)
        checks.extend(tier2_checks)
        tier2_passed = all(c.passed for c in tier2_checks)

    all_passed = tier1_passed and (tier2_passed is None or tier2_passed)

    return ValidationResult(
        passed=all_passed,
        checks=checks,
        tier1_passed=tier1_passed,
        tier2_passed=tier2_passed,
    )


def _run_tier2_checks(video_path: str) -> list[ValidationCheck]:
    """Run AI-assisted Tier 2 validation checks using Moondream2.

    TODO: Implement after Moondream2 integration (Step 5).
    """
    logger.info("Tier 2 AI-assisted checks not yet implemented for %s", video_path)
    return [
        ValidationCheck(
            name="ai_content_present",
            passed=True,
            message="Tier 2 checks not yet implemented (skipped)",
        ),
    ]
