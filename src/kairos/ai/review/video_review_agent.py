"""Kairos Agent — Video Review Agent Service.

Reviews rendered video clips for visual/physics quality issues using a
vision-language model (VLM). Supports escalation to a heavier model for
uncertain clips and optional lightweight pre-check tooling.

Default model:  Qwen3-VL-8B-Instruct (via Ollama)
Escalation:     Qwen3-VL-30B-A3B-Instruct (MoE, for uncertain clips)
Pre-checks:     DOVER (video quality scoring) + LAION aesthetic predictor (optional)

Pipeline integration:
  This agent runs on *every* pipeline's final.mp4 — it is pipeline-agnostic.
  It slots between Video Editor → Audio Review in the LangGraph graph.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kairos.pipelines.contracts import VideoReviewAgent as _VideoReviewAgentBase
from kairos.schemas.contracts import (
    ConceptBrief,
    ReviewIssue,
    ReviewIssueSeverity,
    VideoReviewResult,
)
from kairos.ai.llm.config import get_step_config
from kairos.ai.llm.routing import _record_llm_call

logger = logging.getLogger(__name__)

# Frame sampling defaults
DEFAULT_FRAME_RATE = 0.5       # 0.5 FPS for 8B model (keep under 32K token context)
DEFAULT_MAX_FRAMES = 16        # cap frame count for 8B to stay within context window
ESCALATION_FRAME_RATE = 0.5    # 0.5 FPS for 30B model
ESCALATION_MAX_FRAMES = 32     # cap frame count for 30B to avoid timeouts
DEFAULT_MAX_RESOLUTION = 720   # max height for 8B (balance quality vs context budget)
ESCALATION_MAX_RESOLUTION = 512  # max height for 30B (keep small for speed)


def _extract_frames(
    video_path: str,
    *,
    fps: float = 1.0,
    max_height: int = 1080,
    max_frames: int = 65,
) -> list[str]:
    """Extract frames from a video at a given FPS using FFmpeg.

    Returns a list of paths to extracted JPEG frame files in a temp directory.
    """
    output_dir = tempfile.mkdtemp(prefix="kairos_vr_frames_")
    output_pattern = str(Path(output_dir) / "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=-1:'{min(max_height, 1080)}'",
        "-frames:v", str(max_frames),
        "-q:v", "3",  # JPEG quality (2-5 is good)
        output_pattern,
        "-y", "-loglevel", "warning",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error("Frame extraction failed: %s", exc)
        return []

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def _build_review_prompt(
    concept: ConceptBrief | None = None,
    frame_count: int = 0,
) -> list[dict[str, Any]]:
    """Build the VLM review prompt with context."""

    context_section = ""
    if concept:
        context_section = f"""
## Video Context
- **Title:** {concept.title}
- **Category:** {concept.category.value}
- **Visual Brief:** {concept.visual_brief}
- **Hook Text:** {concept.hook_text}
- **Target Duration:** {concept.target_duration_sec}s
"""

    system_prompt = f"""You are a video quality reviewer for a physics simulation content pipeline.
You are reviewing {frame_count} sequential frames extracted from a rendered video.

Your job is to identify obvious quality issues that would make the video unacceptable for publishing.
{context_section}
## What to check:
1. **Physics Quality** — Do objects move realistically? Any clipping, teleporting, or unrealistic trajectories?
2. **Completion** — Did the simulation run to completion? (e.g., did all dominos fall, did the marble reach the end?)
3. **Object Behavior** — Are objects behaving as expected? Any flying off screen or freezing unexpectedly?
4. **Framing/Camera** — Is the key action visible? Is the camera following the action correctly?
5. **Caption Placement** — Are text overlays visible, readable, and properly positioned?
6. **Visual Polish** — Overall, does this look polished enough for social media?

## CRITICAL: Response Format
You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no extra text.
Do NOT wrap JSON in ```json``` code fences. Output the raw JSON only.

The JSON object MUST follow this exact schema:
{{
  "passed": true,
  "overall_confidence": 0.95,
  "issues": []
}}

Or if there are issues:
{{
  "passed": false,
  "overall_confidence": 0.6,
  "issues": [
    {{
      "category": "broken_physics",
      "severity": "critical",
      "description": "Objects clip through floor at 5s mark",
      "timestamp_sec": 5.0,
      "confidence": 0.9
    }}
  ]
}}

Valid categories: broken_physics, bad_framing, caption_issue, incomplete_simulation, object_behavior, visual_quality
Valid severities: critical, major, minor, info

Be strict but fair. The bar is: "Would a human QA reviewer watching this once say it looks polished?"
If the video looks acceptable, set passed=true with an empty issues list.
Only flag genuine problems, not minor stylistic preferences.

REMEMBER: Output ONLY the JSON object, nothing else."""

    return [{"role": "system", "content": system_prompt}]


def _parse_review_response(raw_text: str, model_used: str) -> VideoReviewResult:
    """Parse the VLM's text response into a structured VideoReviewResult."""
    import json
    import re

    # Try to extract JSON from the response
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        logger.warning("VLM response did not contain JSON — failing review")
        return VideoReviewResult(
            passed=False,
            overall_confidence=0.0,
            issues=[
                ReviewIssue(
                    category="parse_error",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description="Could not parse VLM response — video cannot be approved without valid review",
                    confidence=0.0,
                )
            ],
            model_used=model_used,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("VLM response JSON was malformed — failing review")
        return VideoReviewResult(
            passed=False,
            overall_confidence=0.0,
            issues=[
                ReviewIssue(
                    category="parse_error",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description="Malformed JSON in VLM response — video cannot be approved without valid review",
                    confidence=0.0,
                )
            ],
            model_used=model_used,
        )

    # Parse issues
    issues = []
    for raw_issue in data.get("issues", []):
        try:
            severity_str = raw_issue.get("severity", "major").lower()
            severity = ReviewIssueSeverity(severity_str) if severity_str in ReviewIssueSeverity.__members__.values() else ReviewIssueSeverity.MAJOR
            issues.append(ReviewIssue(
                category=raw_issue.get("category", "unknown"),
                severity=severity,
                description=raw_issue.get("description", ""),
                timestamp_sec=raw_issue.get("timestamp_sec"),
                confidence=float(raw_issue.get("confidence", 1.0)),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed issue in VLM response: %s", exc)

    return VideoReviewResult(
        passed=data.get("passed", True),
        overall_confidence=float(data.get("overall_confidence", 1.0)),
        issues=issues,
        model_used=model_used,
    )


class VideoReviewAgent(_VideoReviewAgentBase):
    """Concrete Video Review Agent using Qwen3-VL via Ollama.

    Default: Qwen3-VL-8B-Instruct reviews all clips.
    Escalation: Qwen3-VL-30B-A3B-Instruct re-reviews uncertain clips.
    Pre-checks: Optional DOVER + LAION aesthetic predictor.
    """

    def __init__(self) -> None:
        # Load config — gracefully handle missing step configs
        try:
            self._config = get_step_config("video_reviewer")
            self._config_raw = self._config._raw
        except KeyError:
            logger.warning("video_reviewer step config not found — using defaults")
            self._config = None
            self._config_raw = {}

        try:
            self._escalation_config = get_step_config("video_reviewer_escalation")
            self._escalation_raw = self._escalation_config._raw
        except KeyError:
            self._escalation_config = None
            self._escalation_raw = {}

    async def review_video(
        self,
        video_path: str,
        concept: ConceptBrief | None = None,
    ) -> VideoReviewResult:
        """Review a rendered video clip using VLM frame analysis.

        1. Optionally run pre-checks (DOVER, aesthetic scoring).
        2. Extract frames from video.
        3. Send frames + prompt to default VLM.
        4. If confidence < threshold, escalate to 30B model.
        5. Return structured review result.
        """
        if not Path(video_path).exists():
            logger.error("[video_review] Video file not found: %s", video_path)
            return VideoReviewResult(
                passed=False,
                overall_confidence=1.0,
                issues=[ReviewIssue(
                    category="missing_file",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description=f"Video file not found: {video_path}",
                )],
                model_used="none",
            )

        # Step 1: Optional pre-checks
        pre_check_scores = await self.run_pre_checks(video_path)

        # Auto-reject if DOVER technical score is below threshold
        pre_check_cfg = self._config_raw.get("pre_checks", {})
        dover_cfg = pre_check_cfg.get("dover", {})
        if dover_cfg.get("enabled", False) and "dover_technical" in pre_check_scores:
            threshold = float(dover_cfg.get("threshold_technical", 0.4))
            if pre_check_scores["dover_technical"] < threshold:
                logger.info("[video_review] DOVER technical score %.2f < threshold %.2f — auto-rejecting",
                            pre_check_scores["dover_technical"], threshold)
                return VideoReviewResult(
                    passed=False,
                    overall_confidence=1.0,
                    issues=[ReviewIssue(
                        category="visual_quality",
                        severity=ReviewIssueSeverity.CRITICAL,
                        description=f"DOVER technical quality score ({pre_check_scores['dover_technical']:.2f}) below threshold ({threshold})",
                    )],
                    model_used="dover_pre_check",
                    pre_check_scores=pre_check_scores,
                )

        # Step 2: Extract frames
        frames = _extract_frames(
            video_path,
            fps=DEFAULT_FRAME_RATE,
            max_height=DEFAULT_MAX_RESOLUTION,
            max_frames=DEFAULT_MAX_FRAMES,
        )
        if not frames:
            logger.error("[video_review] No frames extracted from %s", video_path)
            return VideoReviewResult(
                passed=False,
                overall_confidence=1.0,
                issues=[ReviewIssue(
                    category="frame_extraction",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description="Failed to extract frames from video",
                )],
                model_used="none",
            )

        # Step 3: Build prompt and call default VLM
        model_alias = "video-reviewer-default"
        if self._config:
            model_alias = self._config.resolve_model()

        messages = _build_review_prompt(concept, len(frames))
        # Add frames as image content in the user message
        image_content: list[dict[str, Any]] = []
        for frame_path in frames:
            import base64
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        image_content.append({
            "type": "text",
            "text": (
                f"Review these {len(frames)} sequential frames from the video. "
                "Respond with ONLY a valid JSON object. No markdown, no extra text."
            ),
        })

        messages.append({"role": "user", "content": image_content})

        # ── Structured Output Retry Loop ──────────────────────────────
        # Ask → Validate JSON → If invalid, retry with correction → Fail with reason
        import time as _time
        from kairos.ai.llm.routing import call_ollama_direct

        MAX_STRUCTURED_RETRIES = 2
        result: VideoReviewResult | None = None
        last_raw_text = ""

        for attempt_num in range(1, MAX_STRUCTURED_RETRIES + 1):
            vlm_start = _time.monotonic()
            try:
                ollama_resp = call_ollama_direct(
                    model_alias,
                    messages,
                    max_tokens=8192,
                    timeout=120,
                    json_mode=True,
                )
                raw_text = ollama_resp.content
                raw_thinking = ollama_resp.thinking
                last_raw_text = raw_text
                vlm_latency = int((_time.monotonic() - vlm_start) * 1000)

                _record_llm_call(
                    model_alias=model_alias,
                    model_resolved=ollama_resp.model,
                    call_pattern="vision_review",
                    routing_outcome="direct_vlm",
                    tokens_in=ollama_resp.tokens_in,
                    tokens_out=ollama_resp.tokens_out,
                    cost_usd=0.0,  # local model
                    latency_ms=vlm_latency,
                    status="success",
                    thinking_summary=raw_thinking[:200] if raw_thinking else None,
                    model_type="local",
                    provider="ollama",
                    raw_prompt=[{"role": m["role"], "content": "(image frames omitted)" if isinstance(m.get("content"), list) else m.get("content", "")} for m in messages],
                    raw_response=raw_text,
                    raw_thinking=raw_thinking,
                )

                # Validate: try to parse as structured JSON
                result = _parse_review_response(raw_text, model_alias)

                # Check if parsing actually succeeded (no parse_error issues)
                has_parse_error = any(
                    i.category == "parse_error" for i in result.issues
                )
                if not has_parse_error:
                    logger.info(
                        "[video_review] VLM returned valid structured JSON on attempt %d/%d",
                        attempt_num, MAX_STRUCTURED_RETRIES,
                    )
                    break  # Success — valid JSON received

                # Parse failed — retry with correction message
                if attempt_num < MAX_STRUCTURED_RETRIES:
                    logger.warning(
                        "[video_review] VLM response was not valid JSON (attempt %d/%d) — retrying with correction",
                        attempt_num, MAX_STRUCTURED_RETRIES,
                    )
                    # Add correction message to conversation
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was NOT valid JSON. I need ONLY a JSON object, nothing else.\n"
                            "Please respond with EXACTLY this format (no markdown, no explanation):\n"
                            '{"passed": true, "overall_confidence": 0.95, "issues": []}\n'
                            "Or with issues:\n"
                            '{"passed": false, "overall_confidence": 0.6, "issues": [{"category": "broken_physics", '
                            '"severity": "critical", "description": "description here", "timestamp_sec": null, "confidence": 0.9}]}'
                        ),
                    })
                    result = None  # Reset for retry
                else:
                    logger.error(
                        "[video_review] VLM failed to return valid JSON after %d attempts. "
                        "Last response preview: %.300s",
                        MAX_STRUCTURED_RETRIES, raw_text,
                    )

            except Exception as exc:
                vlm_latency = int((_time.monotonic() - vlm_start) * 1000)
                _record_llm_call(
                    model_alias=model_alias,
                    model_resolved=model_alias,
                    call_pattern="vision_review",
                    routing_outcome="direct_vlm",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=vlm_latency,
                    status="error",
                    error=str(exc),
                    model_type="local",
                    provider="ollama",
                )
                logger.error("[video_review] VLM call failed: %s — FAILING review (no silent pass-through)", exc)
                return VideoReviewResult(
                    passed=False,
                    overall_confidence=0.0,
                    issues=[ReviewIssue(
                        category="reviewer_error",
                        severity=ReviewIssueSeverity.CRITICAL,
                        description=f"VLM review failed ({exc}) — video cannot be approved without review",
                        confidence=0.0,
                    )],
                    model_used=model_alias,
                    pre_check_scores=pre_check_scores,
                )

        # If result is None after retries, the loop exhausted without valid JSON
        if result is None:
            result = VideoReviewResult(
                passed=False,
                overall_confidence=0.0,
                issues=[ReviewIssue(
                    category="parse_error",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description=(
                        f"VLM failed to return valid JSON after {MAX_STRUCTURED_RETRIES} attempts. "
                        f"Last response: {last_raw_text[:300]}"
                    ),
                    confidence=0.0,
                )],
                model_used=model_alias,
                pre_check_scores=pre_check_scores,
            )

        # Attach pre-check scores
        if pre_check_scores:
            result = VideoReviewResult(
                passed=result.passed,
                overall_confidence=result.overall_confidence,
                issues=result.issues,
                model_used=result.model_used,
                escalated=result.escalated,
                pre_check_scores=pre_check_scores,
            )

        # Step 4: Escalation check
        escalation_cfg = self._escalation_raw.get("escalation", {})
        confidence_threshold = float(escalation_cfg.get("confidence_threshold", 0.7))

        if result.overall_confidence < confidence_threshold and self._escalation_config:
            logger.info(
                "[video_review] Confidence %.2f < threshold %.2f — escalating to 30B model",
                result.overall_confidence, confidence_threshold,
            )
            escalation_result = await self._escalate_review(video_path, concept, pre_check_scores)
            return escalation_result

        return result

    async def _escalate_review(
        self,
        video_path: str,
        concept: ConceptBrief | None,
        pre_check_scores: dict[str, float],
    ) -> VideoReviewResult:
        """Re-review with the heavier escalation model (Qwen3-VL-30B-A3B)."""
        escalation_cfg = self._escalation_raw.get("escalation", {})
        frame_rate = float(escalation_cfg.get("max_frame_rate", ESCALATION_FRAME_RATE))
        max_res = int(escalation_cfg.get("max_resolution", ESCALATION_MAX_RESOLUTION))

        frames = _extract_frames(
            video_path,
            fps=frame_rate,
            max_height=max_res,
            max_frames=ESCALATION_MAX_FRAMES,
        )
        if not frames:
            logger.error("[video_review] Escalation: no frames extracted")
            return VideoReviewResult(
                passed=False,
                overall_confidence=0.5,
                issues=[ReviewIssue(
                    category="frame_extraction",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description="Failed to extract frames for escalation review",
                )],
                model_used="video-reviewer-escalation",
                escalated=True,
                pre_check_scores=pre_check_scores,
            )

        model_alias = "video-reviewer-escalation"
        if self._escalation_config:
            model_alias = self._escalation_config.resolve_model()

        messages = _build_review_prompt(concept, len(frames))

        import base64
        image_content: list[dict[str, Any]] = []
        for frame_path in frames:
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        image_content.append({
            "type": "text",
            "text": (
                f"ESCALATED REVIEW: Re-review these {len(frames)} sequential frames. "
                "A previous lighter model flagged this as uncertain. Be thorough. "
                "Respond with ONLY a valid JSON object. No markdown, no extra text."
            ),
        })
        messages.append({"role": "user", "content": image_content})

        import time as _time
        from kairos.ai.llm.routing import call_ollama_direct

        vlm_start = _time.monotonic()
        try:
            ollama_resp = call_ollama_direct(
                model_alias,
                messages,
                max_tokens=8192,
                timeout=600,
                json_mode=True,
            )
            raw_text = ollama_resp.content
            raw_thinking = ollama_resp.thinking
            vlm_latency = int((_time.monotonic() - vlm_start) * 1000)

            _record_llm_call(
                model_alias=model_alias,
                model_resolved=ollama_resp.model,
                call_pattern="vision_review_escalation",
                routing_outcome="direct_vlm",
                tokens_in=ollama_resp.tokens_in,
                tokens_out=ollama_resp.tokens_out,
                cost_usd=0.0,
                latency_ms=vlm_latency,
                status="success",
                thinking_summary=raw_thinking[:200] if raw_thinking else None,
                model_type="local",
                provider="ollama",
                raw_prompt=[{"role": m["role"], "content": "(image frames omitted)" if isinstance(m.get("content"), list) else m.get("content", "")} for m in messages],
                raw_response=raw_text,
                raw_thinking=raw_thinking,
            )

            result = _parse_review_response(raw_text, model_alias)
            return VideoReviewResult(
                passed=result.passed,
                overall_confidence=result.overall_confidence,
                issues=result.issues,
                model_used=model_alias,
                escalated=True,
                pre_check_scores=pre_check_scores,
            )
        except Exception as exc:
            vlm_latency = int((_time.monotonic() - vlm_start) * 1000)
            _record_llm_call(
                model_alias=model_alias,
                model_resolved=model_alias,
                call_pattern="vision_review_escalation",
                routing_outcome="direct_vlm",
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=vlm_latency,
                status="error",
                error=str(exc),
                model_type="local",
                provider="ollama",
            )
            logger.error("[video_review] Escalation VLM call failed: %s", exc)
            return VideoReviewResult(
                passed=False,
                overall_confidence=0.0,
                issues=[ReviewIssue(
                    category="reviewer_error",
                    severity=ReviewIssueSeverity.CRITICAL,
                    description=f"Escalation review failed ({exc}) — video cannot be approved",
                    confidence=0.0,
                )],
                model_used=model_alias,
                escalated=True,
                pre_check_scores=pre_check_scores,
            )

    async def run_pre_checks(
        self,
        video_path: str,
    ) -> dict[str, float]:
        """Run optional lightweight pre-check tools.

        DOVER and LAION aesthetic predictor are optional and disabled by default.
        When enabled, they provide objective quality scores before VLM inference.
        """
        scores: dict[str, float] = {}
        pre_check_cfg = self._config_raw.get("pre_checks", {})

        # DOVER — no-reference video quality assessment
        dover_cfg = pre_check_cfg.get("dover", {})
        if dover_cfg.get("enabled", False):
            try:
                scores.update(await self._run_dover(video_path))
            except Exception as exc:
                logger.warning("[video_review] DOVER pre-check failed: %s", exc)

        # LAION Aesthetic Predictor — keyframe aesthetic scoring
        aesthetic_cfg = pre_check_cfg.get("aesthetic_predictor", {})
        if aesthetic_cfg.get("enabled", False):
            try:
                scores.update(await self._run_aesthetic_predictor(video_path))
            except Exception as exc:
                logger.warning("[video_review] Aesthetic predictor failed: %s", exc)

        return scores

    async def _run_dover(self, video_path: str) -> dict[str, float]:
        """Run DOVER video quality assessment.

        Returns technical and aesthetic quality scores (0-1 scale).
        Requires the `dover` package to be installed.
        """
        # Placeholder — DOVER integration requires the dover package
        # When implemented:
        #   from dover import DOVERModel
        #   model = DOVERModel()
        #   scores = model.predict(video_path)
        #   return {"dover_technical": scores.technical, "dover_aesthetic": scores.aesthetic}
        logger.info("[video_review] DOVER pre-check not yet installed — skipping")
        return {}

    async def _run_aesthetic_predictor(self, video_path: str) -> dict[str, float]:
        """Run LAION aesthetic predictor on keyframes.

        Returns an aesthetic score on a 1-10 scale.
        Requires CLIP + the aesthetic predictor linear model.
        """
        # Placeholder — requires CLIP and the aesthetic predictor model
        # When implemented:
        #   Extract a few keyframes, run CLIP embeddings, predict aesthetic score
        logger.info("[video_review] Aesthetic predictor not yet installed — skipping")
        return {}
