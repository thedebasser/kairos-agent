"""Kairos Agent — Physics Simulation Agent.

Implements SimulationAgent for the Blender 3D physics pipeline.
Orchestrates: config generation → Blender execution → validation → iteration.

Architecture (config-based pipeline):
  1. LLM generates a JSON config matching the category schema
  2. Config drives Blender scene construction via rigid body physics
  3. Validation checks output quality before publishing
  4. On failure, the LLM adjusts the config and re-renders

LLM routing:
  - Generation: ``simulation-first-pass`` (Claude Sonnet) via Instructor
  - Iteration: ``simulation-debugger`` (Claude Sonnet) for config fixes

The agent is stateless — all mutable state lives in PipelineState and the DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from typing import Any

from kairos.pipelines.contracts import SimulationAgent
from kairos.config import get_settings
from kairos.exceptions import (
    SimulationExecutionError,
    SimulationOOMError,
    SimulationTimeoutError,
)
from kairos.schemas.contracts import (
    ConceptBrief,
    SimulationLoopResult,
    SimulationResult,
    SimulationStats,
    ValidationResult,
)
from kairos.ai.llm import routing as llm_routing
from kairos.engines.blender.executor import run_blender_script
from kairos.services import validation
from kairos.ai.llm.config import get_step_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model resolution — driven by llm_config.yaml
# ---------------------------------------------------------------------------

def _generation_model() -> str:
    return get_step_config("simulation_code_generation").resolve_model()


# Regex for extracting simulation stdout markers
_PAYOFF_RE = re.compile(r"PAYOFF_TIMESTAMP[=:]?\s*([\d.]+)")
_PEAK_BODY_RE = re.compile(r"PEAK_BODY_COUNT[=:]?\s*(\d+)")
_COMPLETION_RE = re.compile(r"COMPLETION_RATIO[=:]?\s*([\d.]+)")
_FALLEN_RE = re.compile(r"FALLEN[=:]?\s*(\d+)/(\d+)")


class PhysicsSimulationAgent(SimulationAgent):
    """Simulation agent for the *Oddly Satisfying Physics* pipeline.

    Uses the config-based Blender architecture:
    1. LLM generates a JSON config (creative params only)
    2. Config drives Blender scene construction and rendering
    3. Validation checks output quality
    4. Iteration loop adjusts config (not code) on failure
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._current_config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # ABC: generate_simulation
    # ------------------------------------------------------------------

    async def generate_simulation(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate simulation config via LLM and build a Blender script.

        Returns the Blender Python script as a string.
        """
        category = concept.category.value

        system_prompt = (
            "You are a Blender 3D simulation engineer. Generate a JSON config "
            "for a physics simulation. The config will drive Blender's rigid body "
            "physics engine to create an oddly satisfying short-form video.\n\n"
            f"Category: {category}\n"
            "Output a JSON config with creative parameters for the scene."
        )

        user_prompt = (
            f"Title: {concept.title}\n"
            f"Visual brief: {concept.visual_brief}\n"
            f"Category: {category}\n"
            f"Body count: {concept.simulation_requirements.body_count_initial}"
            f"-{concept.simulation_requirements.body_count_max}\n"
            f"Interaction type: {concept.simulation_requirements.interaction_type}\n"
            f"Colour palette: {json.dumps(concept.simulation_requirements.colour_palette)}\n"
            f"Background: {concept.simulation_requirements.background_colour}\n"
            f"Duration: {concept.target_duration_sec}s\n"
        )

        # --- Learning loop: assemble extra context ---
        try:
            from kairos.ai.learning.learning_loop import (
                format_few_shot_prompt,
                get_category_knowledge_for_prompt,
                get_few_shot_examples,
                get_validation_rules_prompt,
            )

            rules = get_validation_rules_prompt()
            if rules:
                user_prompt += f"\n\n{rules}"

            examples = await get_few_shot_examples(
                pipeline="physics", category=category, limit=2,
            )
            few_shot = format_few_shot_prompt(examples)
            if few_shot:
                user_prompt += f"\n\n{few_shot}"

            ck_text = await get_category_knowledge_for_prompt("physics", category)
            if ck_text:
                user_prompt += f"\n\n{ck_text}"
        except Exception as exc:
            logger.debug("Learning loop context injection skipped: %s", exc)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        from kairos.schemas.simulation import SimulationConfigOutput

        start = time.monotonic()
        result: SimulationConfigOutput = await llm_routing.call_llm(
            model=_generation_model(),
            messages=messages,
            response_model=SimulationConfigOutput,
            cache_step="simulation_config_gen",
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        self._current_config = result.config
        logger.info(
            "Generated simulation config (%d keys) in %dms — %s",
            len(result.config),
            latency_ms,
            result.reasoning[:200],
        )

        return json.dumps(result.config, indent=2)

    # ------------------------------------------------------------------
    # ABC: execute_simulation
    # ------------------------------------------------------------------

    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute simulation via Blender subprocess."""
        return await run_blender_script(
            code,
            timeout=self._settings.sandbox_timeout_sec,
        )

    # ------------------------------------------------------------------
    # ABC: validate_output
    # ------------------------------------------------------------------

    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run Tier 1 (mandatory) + optional Tier 2 validation."""
        return await validation.validate_simulation(
            video_path,
            run_tier2=False,
            skip_checks={"audio_present"},
        )

    # ------------------------------------------------------------------
    # ABC: get_simulation_stats
    # ------------------------------------------------------------------

    async def get_simulation_stats(self, video_path: str) -> SimulationStats:
        """Extract statistics from the rendered video via FFprobe."""
        loop = asyncio.get_running_loop()
        probe = await loop.run_in_executor(None, self._ffprobe, video_path)

        duration = float(probe.get("format", {}).get("duration", 0))
        file_size = int(probe.get("format", {}).get("size", 0))

        fps = 0.0
        total_frames = 0
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    fps = int(num) / int(den) if int(den) else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                try:
                    total_frames = int(stream.get("nb_frames", 0))
                except (ValueError, TypeError):
                    total_frames = int(duration * fps) if fps else 0
                break

        return SimulationStats(
            duration_sec=round(duration, 2),
            peak_body_count=0,
            avg_fps=round(fps, 1),
            min_fps=round(fps, 1),
            payoff_timestamp_sec=0.0,
            total_frames=total_frames,
            file_size_bytes=file_size,
        )

    # ------------------------------------------------------------------
    # ABC: run_loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        concept: ConceptBrief,
    ) -> SimulationLoopResult:
        """Orchestrate the full simulation iteration loop.

        generate → execute → validate → iterate
        Up to ``max_simulation_iterations`` total attempts.
        """
        max_iters = self._settings.max_simulation_iterations
        result = SimulationLoopResult()

        logger.info("Generating initial simulation for '%s'", concept.title)
        try:
            code = await self.generate_simulation(concept)
        except Exception as exc:
            raise SimulationExecutionError(
                f"Simulation generation failed: {exc}"
            ) from exc

        result.simulation_code = code

        for iteration in range(1, max_iters + 1):
            result.simulation_iteration = iteration
            logger.info("Simulation iteration %d/%d", iteration, max_iters)

            try:
                sim_result = await self.execute_simulation(code)
            except (SimulationTimeoutError, SimulationOOMError, SimulationExecutionError) as exc:
                logger.warning("Execution failed (iter %d): %s", iteration, exc)
                result.errors.append(f"iter{iteration}: {exc}")
                if iteration >= max_iters:
                    break
                continue

            result.simulation_result = sim_result

            video_path = self._find_video(sim_result)
            if video_path is None:
                msg = f"iter{iteration}: No MP4 file in output (files: {sim_result.output_files})"
                logger.warning(msg)
                result.errors.append(msg)
                if iteration >= max_iters:
                    break
                continue

            result.raw_video_path = video_path

            stats = await self.get_simulation_stats(video_path)
            stats = self._enrich_stats_from_stdout(stats, sim_result.stdout)
            result.simulation_stats = stats

            completion_ok, completion_ratio, completion_msg = (
                self._check_completion_from_stdout(sim_result.stdout)
            )
            logger.info("Completion check: %s (ratio=%.2f)", completion_msg, completion_ratio)

            if not completion_ok:
                logger.warning("Completion check failed (iter %d): %s", iteration, completion_msg)
                if iteration >= max_iters:
                    result.errors.append(f"Completion check failed: {completion_msg}")
                    break
                continue

            vresult = await self.validate_output(video_path)
            result.validation_result = vresult

            if vresult.passed:
                logger.info("Simulation passed validation on iteration %d", iteration)
                return result

            logger.warning(
                "Validation failed (iter %d): %s — %s",
                iteration, vresult.summary,
                [c.name for c in vresult.failed_checks],
            )

            if iteration >= max_iters:
                result.errors.append(
                    f"Max iterations ({max_iters}) reached without passing validation"
                )
                break

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_video(result: SimulationResult) -> str | None:
        """Find the first .mp4 file in the simulation output."""
        for f in result.output_files:
            if f.lower().endswith(".mp4"):
                return f
        return None

    @staticmethod
    def _ffprobe(video_path: str) -> dict[str, Any]:
        """Run ffprobe and return parsed JSON metadata."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return {}
            return json.loads(result.stdout)  # type: ignore[no-any-return]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return {}

    @staticmethod
    def _enrich_stats_from_stdout(
        stats: SimulationStats,
        stdout: str,
    ) -> SimulationStats:
        """Parse PAYOFF_TIMESTAMP, PEAK_BODY_COUNT from stdout."""
        payoff = 0.0
        peak = 0

        m = _PAYOFF_RE.search(stdout)
        if m:
            try:
                payoff = float(m.group(1))
            except ValueError:
                pass

        m = _PEAK_BODY_RE.search(stdout)
        if m:
            try:
                peak = int(m.group(1))
            except ValueError:
                pass

        if payoff or peak:
            return SimulationStats(
                duration_sec=stats.duration_sec,
                peak_body_count=peak or stats.peak_body_count,
                avg_fps=stats.avg_fps,
                min_fps=stats.min_fps,
                payoff_timestamp_sec=payoff or stats.payoff_timestamp_sec,
                total_frames=stats.total_frames,
                file_size_bytes=stats.file_size_bytes,
            )
        return stats

    @staticmethod
    def _check_completion_from_stdout(stdout: str) -> tuple[bool, float, str]:
        """Check if the simulation completed based on stdout markers."""
        m = _COMPLETION_RE.search(stdout)
        if m:
            try:
                ratio = float(m.group(1))
                passed = ratio >= 0.75
                msg = f"Completion ratio: {ratio:.1%}"
                if not passed:
                    msg += " (below 75% threshold)"
                return passed, ratio, msg
            except ValueError:
                pass

        m = _FALLEN_RE.search(stdout)
        if m:
            try:
                fallen = int(m.group(1))
                total = int(m.group(2))
                ratio = fallen / total if total > 0 else 0
                passed = ratio >= 0.75
                msg = f"Fallen: {fallen}/{total} ({ratio:.0%})"
                if not passed:
                    msg += " (below 75% threshold)"
                return passed, ratio, msg
            except ValueError:
                pass

        if "ERROR:" in stdout and "pre-validation" in stdout.lower():
            return False, 0.0, "Pre-validation failed"

        return True, 1.0, "No completion markers (assumed OK)"
