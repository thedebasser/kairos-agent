"""Kairos Agent — Audio Review Agent Service.

Reviews the final composed audio (music + TTS + SFX mix) for quality issues.
FFmpeg ebur128 loudness analysis always runs regardless of which primary
reviewer is selected.

Default model:  qwen3.5:9b (text+vision) analysing FFmpeg metrics (text-based)
Specialist:     Whisper large-v3 (TTS WER) + BEATs (event detection) + FFmpeg
Escalation:     qwen3.5:35b-a3b (text+vision) analysing FFmpeg metrics

NOTE: No audio-capable LLM is currently available on Ollama. The "omni"
      review mode has been replaced with a metrics-based approach: extended
      FFmpeg analysis (loudness, silence, clipping, duration) is run first,
      then the metrics are sent as TEXT to the LLM for intelligent assessment.
      When a true omni-audio model (e.g. qwen3.5-omni) becomes available on
      Ollama, the raw-audio path can be re-enabled.

Pipeline integration:
  Runs on every pipeline's final.mp4 audio track (extracted via FFmpeg).
  Slots after Video Review → before Human Review in the LangGraph graph.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kairos.agents.base import BaseAudioReviewAgent
from kairos.models.contracts import (
    AudioReviewResult,
    LoudnessMetrics,
    ReviewIssue,
    ReviewIssueSeverity,
)
from kairos.services.llm_config import get_step_config
from kairos.services.llm_routing import _record_llm_call

logger = logging.getLogger(__name__)


def _safe_float(value: str, default: float = 0.0) -> float:
    """Safely convert a string to float, returning default on failure.

    Handles edge cases like '-', 'inf', '-inf', 'N/A', empty strings
    that FFmpeg/FFprobe may output.
    """
    try:
        v = value.strip()
        if not v or v in ('-', 'N/A', 'nan', ''):
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


# =============================================================================
# FFmpeg Loudness Measurement (always runs)
# =============================================================================


def _extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio track from video file as WAV using FFmpeg.

    Returns path to the extracted audio file.
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav", prefix="kairos_ar_")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",           # 16kHz for speech models
        "-ac", "1",               # mono
        output_path,
        "-y", "-loglevel", "warning",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error("Audio extraction failed: %s", exc)
        raise RuntimeError(f"Failed to extract audio from {video_path}: {exc}") from exc

    return output_path


def _measure_loudness_ffmpeg(audio_path: str) -> LoudnessMetrics:
    """Run FFmpeg ebur128 loudness measurement on an audio file.

    Returns standardised LUFS, dBTP, and loudness range metrics.
    """
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        stderr = result.stderr
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error("FFmpeg loudness measurement failed: %s", exc)
        return LoudnessMetrics(
            integrated_lufs=-99.0,
            true_peak_dbtp=-99.0,
            passed=False,
            details=f"Measurement failed: {exc}",
        )

    # Parse ebur128 output from stderr
    integrated_lufs = -99.0
    true_peak = -99.0
    loudness_range = 0.0

    # Look for "Integrated loudness:" section
    i_match = re.search(r"I:\s*([-\d.]+)\s*LUFS", stderr)
    if i_match:
        integrated_lufs = _safe_float(i_match.group(1), -99.0)

    # Look for true peak
    tp_match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", stderr)
    if not tp_match:
        tp_match = re.search(r"True peak:\s*([-\d.]+)", stderr)
    if tp_match:
        true_peak = _safe_float(tp_match.group(1), -99.0)

    # Look for loudness range
    lra_match = re.search(r"LRA:\s*([-\d.]+)\s*LU", stderr)
    if lra_match:
        loudness_range = _safe_float(lra_match.group(1), 0.0)

    return LoudnessMetrics(
        integrated_lufs=integrated_lufs,
        true_peak_dbtp=true_peak,
        loudness_range_lu=loudness_range,
        passed=True,  # will be checked against thresholds by caller
        details=f"Measured via FFmpeg ebur128",
    )


def _detect_silence(audio_path: str, threshold_db: float = -40.0, min_duration: float = 2.0) -> list[dict[str, float]]:
    """Detect silence periods in audio using FFmpeg silencedetect filter.

    Returns list of {start, end, duration} dicts for each silent segment.
    """
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
    except Exception as exc:
        logger.warning("Silence detection failed: %s", exc)
        return []

    segments: list[dict[str, float]] = []
    starts = re.findall(r"silence_start:\s*([-\d.]+)", stderr)
    ends = re.findall(r"silence_end:\s*([-\d.]+).*silence_duration:\s*([-\d.]+)", stderr)

    for i, start_str in enumerate(starts):
        seg: dict[str, float] = {"start": _safe_float(start_str, 0.0)}
        if i < len(ends):
            seg["end"] = _safe_float(ends[i][0], 0.0)
            seg["duration"] = _safe_float(ends[i][1], 0.0)
        segments.append(seg)

    return segments


def _detect_clipping(audio_path: str) -> dict[str, Any]:
    """Detect clipping (samples at 0 dBFS) using FFmpeg astats filter.

    Returns dict with clipping info.
    """
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", "astats=metadata=1:reset=1",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
    except Exception as exc:
        logger.warning("Clipping detection failed: %s", exc)
        return {"error": str(exc)}

    # Count number of clipped samples
    clip_matches = re.findall(r"Number of Clips:\s*(\d+)", stderr)
    total_clips = sum(int(c) for c in clip_matches) if clip_matches else 0

    peak_matches = re.findall(r"Peak level dB:\s*([-\d.]+)", stderr)
    parsed_peaks = [_safe_float(p, -99.0) for p in peak_matches]
    max_peak = max(parsed_peaks, default=-99.0) if parsed_peaks else -99.0

    return {
        "total_clips": total_clips,
        "max_peak_db": max_peak,
        "has_clipping": total_clips > 0 or max_peak >= -0.1,
    }


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via FFprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        raw = result.stdout.strip()
        if not raw or raw in ("-", "N/A", ""):
            return 0.0
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Could not parse audio duration from ffprobe output: %r", result.stdout.strip() if 'result' in dir() else 'N/A')
        return 0.0
    except Exception:
        return 0.0


def _check_loudness_thresholds(
    metrics: LoudnessMetrics,
    lufs_min: float = -16.0,
    lufs_max: float = -14.0,
    peak_max: float = -1.0,
) -> tuple[bool, list[ReviewIssue]]:
    """Check loudness metrics against thresholds.

    Returns (passed, list_of_issues).
    """
    issues: list[ReviewIssue] = []

    if metrics.integrated_lufs < lufs_min:
        issues.append(ReviewIssue(
            category="loudness",
            severity=ReviewIssueSeverity.MAJOR,
            description=f"Integrated loudness too low: {metrics.integrated_lufs:.1f} LUFS (min: {lufs_min})",
            confidence=1.0,
        ))
    elif metrics.integrated_lufs > lufs_max:
        issues.append(ReviewIssue(
            category="loudness",
            severity=ReviewIssueSeverity.MAJOR,
            description=f"Integrated loudness too high: {metrics.integrated_lufs:.1f} LUFS (max: {lufs_max})",
            confidence=1.0,
        ))

    if metrics.true_peak_dbtp > peak_max:
        issues.append(ReviewIssue(
            category="loudness",
            severity=ReviewIssueSeverity.CRITICAL,
            description=f"True peak exceeds limit: {metrics.true_peak_dbtp:.1f} dBTP (max: {peak_max})",
            confidence=1.0,
        ))

    passed = len(issues) == 0
    return passed, issues


# =============================================================================
# Audio Review Prompt
# =============================================================================


def _build_audio_review_prompt(
    expected_transcript: str = "",
) -> list[dict[str, Any]]:
    """Build the omni-modal LLM audio review prompt."""

    transcript_section = ""
    if expected_transcript:
        transcript_section = f"""
## TTS Transcript Check
The narration should say the following:
\"{expected_transcript}\"
Check whether the voiceover accurately says these words (allow minor pronunciation variations).
"""

    system_prompt = f"""You are an audio quality reviewer for a content pipeline.
You are listening to the full audio mix of a ~65-second video (music + TTS voiceover + collision SFX).

## What to check:
1. **Artifacts/Noise** — Any background static, clicks, pops, or noise artifacts?
2. **Unexpected Sounds** — Any weird/unexpected sounds that don't belong?
3. **Theme/Vibe** — Does the audio match a satisfying/relaxing/engaging vibe?
4. **Volume Consistency** — Are volume levels obviously inconsistent across the mix?
5. **Mix Quality** — Is the music-to-voiceover balance acceptable?
{transcript_section}
## CRITICAL: Response Format
You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no extra text.
Do NOT wrap JSON in ```json``` code fences. Output the raw JSON only.

The JSON object MUST follow this exact schema:
{{
  "passed": true,
  "issues": []
}}

Or if there are issues:
{{
  "passed": false,
  "issues": [
    {{
      "category": "audio_artifact",
      "severity": "major",
      "description": "Static noise audible at 10s",
      "timestamp_sec": 10.0,
      "confidence": 0.8
    }}
  ]
}}

Valid categories: audio_artifact, unexpected_sound, vibe_mismatch, volume_issue, tts_error, mix_quality
Valid severities: critical, major, minor, info

Be practical — the bar is "would this audio be acceptable for a social media video?"
Only flag genuine problems, not minor preferences.

REMEMBER: Output ONLY the JSON object, nothing else."""

    return [{"role": "system", "content": system_prompt}]


def _parse_audio_review_response(raw_text: str, model_used: str) -> tuple[bool, list[ReviewIssue]]:
    """Parse the audio LLM's text response into pass/fail + issues."""

    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        logger.warning("Audio reviewer response did not contain JSON")
        return False, [ReviewIssue(
            category="parse_error",
            severity=ReviewIssueSeverity.CRITICAL,
            description="Could not parse audio reviewer response — audio cannot be approved without valid review",
            confidence=0.0,
        )]

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return False, [ReviewIssue(
            category="parse_error",
            severity=ReviewIssueSeverity.CRITICAL,
            description="Malformed JSON in audio reviewer response — audio cannot be approved without valid review",
            confidence=0.0,
        )]

    issues = []
    for raw_issue in data.get("issues", []):
        try:
            severity_str = raw_issue.get("severity", "major").lower()
            try:
                severity = ReviewIssueSeverity(severity_str)
            except ValueError:
                severity = ReviewIssueSeverity.MAJOR
            issues.append(ReviewIssue(
                category=raw_issue.get("category", "unknown"),
                severity=severity,
                description=raw_issue.get("description", ""),
                timestamp_sec=raw_issue.get("timestamp_sec"),
                confidence=float(raw_issue.get("confidence", 1.0)),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed issue in audio review response: %s", exc)

    return data.get("passed", True), issues


# =============================================================================
# Audio Review Agent
# =============================================================================


class AudioReviewAgent(BaseAudioReviewAgent):
    """Concrete Audio Review Agent.

    Default: FFmpeg extended analysis (loudness, silence, clipping) + LLM
             text-based assessment of those metrics.
    Specialist stack: Whisper large-v3 (TTS WER) + BEATs (event detection) + FFmpeg.
    Escalation: qwen3.5:35b-a3b (heavier LLM for metrics analysis).

    FFmpeg loudness analysis always runs regardless of primary reviewer choice.
    """

    def __init__(self) -> None:
        try:
            self._config = get_step_config("audio_reviewer")
            self._config_raw = self._config._raw
        except KeyError:
            logger.warning("audio_reviewer step config not found — using defaults")
            self._config = None
            self._config_raw = {}

        try:
            self._specialist_config = get_step_config("audio_reviewer_specialist")
            self._specialist_raw = self._specialist_config._raw
        except KeyError:
            self._specialist_config = None
            self._specialist_raw = {}

        try:
            self._escalation_config = get_step_config("audio_reviewer_escalation")
        except KeyError:
            self._escalation_config = None

        # Determine which review mode to use
        self._review_mode = "omni"  # default
        if self._specialist_raw.get("stack") == "specialist":
            # Check if specialist stack is explicitly selected
            pass  # stays as "omni" by default, switch via config override

    async def review_audio(
        self,
        audio_path: str,
        expected_transcript: str = "",
    ) -> AudioReviewResult:
        """Review audio quality with model + FFmpeg loudness check.

        1. Extract audio from video (if needed).
        2. Run FFmpeg ebur128 loudness analysis (always).
        3. Run primary audio reviewer (omni LLM or specialist stack).
        4. Combine results into AudioReviewResult.
        """
        # Determine if we need to extract audio
        source_path = audio_path
        extracted_audio_path: str | None = None

        if audio_path.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
            try:
                extracted_audio_path = _extract_audio(audio_path)
                source_path = extracted_audio_path
            except RuntimeError as exc:
                logger.error("[audio_review] Audio extraction failed: %s", exc)
                return AudioReviewResult(
                    passed=False,
                    issues=[ReviewIssue(
                        category="audio_extraction",
                        severity=ReviewIssueSeverity.CRITICAL,
                        description=f"Failed to extract audio: {exc}",
                    )],
                    model_used="none",
                )

        # Step 1: FFmpeg loudness measurement (ALWAYS runs)
        loudness_metrics = _measure_loudness_ffmpeg(source_path)

        # Check loudness thresholds
        loudness_cfg = self._config_raw.get("loudness", {})
        lufs_min = float(loudness_cfg.get("target_lufs_min", -16.0))
        lufs_max = float(loudness_cfg.get("target_lufs_max", -14.0))
        peak_max = float(loudness_cfg.get("true_peak_max_dbtp", -1.0))

        loudness_passed, loudness_issues = _check_loudness_thresholds(
            loudness_metrics, lufs_min, lufs_max, peak_max,
        )

        # Update loudness metrics with pass status
        loudness_metrics = LoudnessMetrics(
            integrated_lufs=loudness_metrics.integrated_lufs,
            true_peak_dbtp=loudness_metrics.true_peak_dbtp,
            loudness_range_lu=loudness_metrics.loudness_range_lu,
            passed=loudness_passed,
            details=loudness_metrics.details,
        )

        # Step 2: Primary audio review
        model_issues: list[ReviewIssue] = []
        model_passed = True
        model_used = "ffmpeg_only"
        tts_wer: float | None = None
        dnsmos_scores: dict[str, float] | None = None

        model_alias = "audio-reviewer-default"
        if self._config:
            try:
                model_alias = self._config.resolve_model()
            except ValueError:
                model_alias = "audio-reviewer-default"

        # Call the LLM for metrics-based audio review
        try:
            model_passed, model_issues = await self._run_omni_review(
                audio_path, model_alias, expected_transcript,
                loudness_metrics=loudness_metrics,
            )
            model_used = model_alias
        except Exception as exc:
            logger.error("[audio_review] Omni model review failed: %s — proceeding with loudness only", exc)
            model_issues.append(ReviewIssue(
                category="reviewer_error",
                severity=ReviewIssueSeverity.INFO,
                description=f"Audio model review failed ({exc}) — loudness check only",
                confidence=0.3,
            ))

        # Optional: DNSMOS speech quality check
        dnsmos_cfg = self._config_raw.get("dnsmos", {})
        if dnsmos_cfg.get("enabled", False) and extracted_audio_path:
            try:
                dnsmos_scores = await self._run_dnsmos(extracted_audio_path)
                threshold = float(dnsmos_cfg.get("threshold_ovrl", 3.0))
                if dnsmos_scores and dnsmos_scores.get("ovrl", 5.0) < threshold:
                    model_issues.append(ReviewIssue(
                        category="speech_quality",
                        severity=ReviewIssueSeverity.MAJOR,
                        description=f"DNSMOS speech quality score {dnsmos_scores['ovrl']:.2f} below threshold {threshold}",
                        confidence=1.0,
                    ))
            except Exception as exc:
                logger.warning("[audio_review] DNSMOS check failed: %s", exc)

        # Combine all issues
        all_issues = loudness_issues + model_issues
        overall_passed = loudness_passed and model_passed and not any(
            i.severity == ReviewIssueSeverity.CRITICAL for i in all_issues
        )

        return AudioReviewResult(
            passed=overall_passed,
            issues=all_issues,
            model_used=model_used,
            loudness=loudness_metrics,
            tts_wer=tts_wer,
            dnsmos_scores=dnsmos_scores,
        )

    async def _run_omni_review(
        self,
        audio_path: str,
        model_alias: str,
        expected_transcript: str,
        loudness_metrics: LoudnessMetrics | None = None,
    ) -> tuple[bool, list[ReviewIssue]]:
        """Run LLM audio review using FFmpeg metrics (text-based analysis).

        No audio-capable LLM is available on Ollama, so we:
        1. Run extended FFmpeg analysis (silence, clipping, duration)
        2. Send ALL metrics as structured text to the LLM
        3. LLM analyses metrics and returns structured JSON assessment

        When a true omni-audio model becomes available, re-enable the
        raw audio path (see commented block below).
        """
        # ── Gather extended FFmpeg metrics ─────────────────────────────
        # Extract audio WAV for analysis if source is video
        analysis_path = audio_path
        extracted_for_analysis: str | None = None
        if audio_path.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
            try:
                extracted_for_analysis = _extract_audio(audio_path)
                analysis_path = extracted_for_analysis
            except RuntimeError:
                analysis_path = audio_path  # try anyway

        duration = _get_audio_duration(analysis_path)
        silence_segments = _detect_silence(analysis_path)
        clipping_info = _detect_clipping(analysis_path)

        # Build metrics summary text for the LLM
        metrics_text = "## FFmpeg Audio Metrics\n\n"
        metrics_text += f"Duration: {duration:.1f}s\n"

        if loudness_metrics:
            metrics_text += f"Integrated Loudness: {loudness_metrics.integrated_lufs:.1f} LUFS\n"
            metrics_text += f"True Peak: {loudness_metrics.true_peak_dbtp:.1f} dBTP\n"
            if loudness_metrics.loudness_range_lu:
                metrics_text += f"Loudness Range: {loudness_metrics.loudness_range_lu:.1f} LU\n"

        if silence_segments:
            metrics_text += f"\nSilence segments detected ({len(silence_segments)}):\n"
            for seg in silence_segments[:10]:  # cap at 10
                start = seg.get('start', 0)
                dur = seg.get('duration', 0)
                metrics_text += f"  - {start:.1f}s\u2013{start+dur:.1f}s ({dur:.1f}s)\n"
        else:
            metrics_text += "\nNo silence segments detected (>2s at -40dB threshold).\n"

        metrics_text += f"\nClipping: {'YES' if clipping_info.get('has_clipping') else 'None detected'}\n"
        if clipping_info.get('has_clipping'):
            metrics_text += f"  Total clipped samples: {clipping_info.get('total_clips', 0)}\n"
            metrics_text += f"  Max peak: {clipping_info.get('max_peak_db', -99):.1f} dB\n"

        # Clean up temp file
        if extracted_for_analysis:
            try:
                Path(extracted_for_analysis).unlink(missing_ok=True)
            except Exception:
                pass

        # ── Build messages for text-based LLM review ──────────────────
        messages = _build_audio_review_prompt(expected_transcript)
        messages.append({
            "role": "user",
            "content": (
                "I cannot send you the audio directly. Instead, here are the "
                "FFmpeg analysis results for the audio track. Please assess "
                "audio quality based on these metrics and respond with ONLY "
                "a valid JSON object.\n\n"
                + metrics_text
            ),
        })

        # ── Structured Output Retry Loop ──────────────────────────────
        import time as _time
        from kairos.services.llm_routing import call_ollama_direct

        MAX_STRUCTURED_RETRIES = 2
        last_raw_text = ""

        for attempt_num in range(1, MAX_STRUCTURED_RETRIES + 1):
            llm_start = _time.monotonic()
            try:
                ollama_resp = call_ollama_direct(
                    model_alias,
                    messages,
                    max_tokens=2048,
                    timeout=120,
                    json_mode=True,
                )
                raw_text = ollama_resp.content
                raw_thinking = ollama_resp.thinking
                last_raw_text = raw_text
                llm_latency = int((_time.monotonic() - llm_start) * 1000)

                _record_llm_call(
                    model_alias=model_alias,
                    model_resolved=ollama_resp.model,
                    call_pattern="audio_review_metrics",
                    routing_outcome="metrics_text_review",
                    tokens_in=ollama_resp.tokens_in,
                    tokens_out=ollama_resp.tokens_out,
                    cost_usd=0.0,
                    latency_ms=llm_latency,
                    status="success",
                    thinking_summary=raw_thinking[:200] if raw_thinking else None,
                    model_type="local",
                    provider="ollama",
                    raw_prompt=messages,
                    raw_response=raw_text,
                    raw_thinking=raw_thinking,
                )

                # Validate: try parse as structured JSON
                passed, issues = _parse_audio_review_response(raw_text, model_alias)
                has_parse_error = any(
                    i.category == "parse_error" for i in issues
                )

                if not has_parse_error:
                    logger.info(
                        "[audio_review] Model returned valid structured JSON on attempt %d/%d",
                        attempt_num, MAX_STRUCTURED_RETRIES,
                    )
                    return passed, issues

                # Parse failed — retry with correction
                if attempt_num < MAX_STRUCTURED_RETRIES:
                    logger.warning(
                        "[audio_review] Response was not valid JSON (attempt %d/%d) — retrying",
                        attempt_num, MAX_STRUCTURED_RETRIES,
                    )
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was NOT valid JSON. I need ONLY a JSON object.\n"
                            "Respond with EXACTLY this format:\n"
                            '{"passed": true, "issues": []}\n'
                            "Or with issues:\n"
                            '{"passed": false, "issues": [{"category": "audio_artifact", '
                            '"severity": "major", "description": "description", "timestamp_sec": null, "confidence": 0.8}]}'
                        ),
                    })
                else:
                    logger.error(
                        "[audio_review] Failed to get valid JSON after %d attempts. Preview: %.300s",
                        MAX_STRUCTURED_RETRIES, raw_text,
                    )
                    return passed, issues  # Return the parse-error result

            except Exception as exc:
                llm_latency = int((_time.monotonic() - llm_start) * 1000)
                _record_llm_call(
                    model_alias=model_alias,
                    model_resolved=model_alias,
                    call_pattern="audio_review_metrics",
                    routing_outcome="metrics_text_review",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=llm_latency,
                    status="error",
                    error=str(exc),
                    model_type="local",
                    provider="ollama",
                )
                raise  # re-raise so caller handles fallback

        # Should not reach here, but safety fallback
        return False, [ReviewIssue(
            category="parse_error",
            severity=ReviewIssueSeverity.CRITICAL,
            description=f"Audio review exhausted retries without valid JSON: {last_raw_text[:300]}",
            confidence=0.0,
        )]

    async def _run_dnsmos(self, audio_path: str) -> dict[str, float]:
        """Run DNSMOS P.835 speech quality assessment.

        Returns sig (signal), bak (background), ovrl (overall) scores (1-5 MOS).
        Requires the dnsmos package to be installed.
        """
        # Placeholder — DNSMOS integration requires the ONNX model
        # When implemented:
        #   from dnsmos import DNSMOS
        #   model = DNSMOS()
        #   scores = model.predict(audio_path)
        #   return {"sig": scores.sig, "bak": scores.bak, "ovrl": scores.ovrl}
        logger.info("[audio_review] DNSMOS not yet installed — skipping")
        return {}

    async def measure_loudness(
        self,
        audio_path: str,
    ) -> dict[str, float]:
        """Run FFmpeg ebur128 loudness measurement.

        This always runs regardless of which primary reviewer is active.
        """
        # Extract audio if needed
        source_path = audio_path
        if audio_path.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
            source_path = _extract_audio(audio_path)

        metrics = _measure_loudness_ffmpeg(source_path)
        return {
            "integrated_lufs": metrics.integrated_lufs,
            "true_peak_dbtp": metrics.true_peak_dbtp,
            "loudness_range_lu": metrics.loudness_range_lu,
        }
