"""Kairos Agent — Physics Simulation Agent.

Implements BaseSimulationAgent for the Pygame + Pymunk physics pipeline.
Orchestrates: code generation → sandbox execution → validation → adjustment.

LLM routing:
  - Generation: ``simulation-first-pass`` (Claude Sonnet)
  - Adjustment: ``sim-param-adjust`` (Mistral 7B local) with fallback to
    ``simulation-debugger`` (Claude Sonnet)

The agent is stateless — all mutable state lives in PipelineState and the DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from kairos.agents.base import BaseSimulationAgent
from kairos.config import get_settings
from kairos.exceptions import (
    SimulationExecutionError,
    SimulationOOMError,
    SimulationTimeoutError,
)
from kairos.models.contracts import (
    AgentRunStatus,
    ConceptBrief,
    PipelineState,
    PipelineStatus,
    SimulationResult,
    SimulationStats,
    ValidationResult,
)
from kairos.models.simulation import AdjustedSimulationCode, SimulationCode
from kairos.services import llm_routing, sandbox, validation
from kairos.services.llm_config import get_step_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model resolution — driven by llm_config.yaml
# ---------------------------------------------------------------------------

def _generation_model() -> str:
    return get_step_config("simulation_code_generation").resolve_model()

def _adjustment_models() -> tuple[str, str]:
    return get_step_config("simulation_param_adjustment").resolve_primary_and_fallback()

# Regex for extracting simulation stdout markers
_PAYOFF_RE = re.compile(r"PAYOFF_TIMESTAMP[=:]?\s*([\d.]+)")
_PEAK_BODY_RE = re.compile(r"PEAK_BODY_COUNT[=:]?\s*(\d+)")


class PhysicsSimulationAgent(BaseSimulationAgent):
    """Simulation agent for the *Oddly Satisfying Physics* pipeline.

    Implements the full generate → execute → validate → adjust loop with
    configurable max iterations (default 5).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._prompts_dir = (
            Path(__file__).resolve().parent / "prompts"
        )

    # ------------------------------------------------------------------
    # ABC: generate_simulation
    # ------------------------------------------------------------------

    async def generate_simulation(
        self,
        concept: ConceptBrief,
        state: PipelineState,
    ) -> str:
        """Generate initial simulation code via cloud LLM.

        Builds a category-specific prompt from the template, fills in the
        concept details, and calls ``simulation-first-pass``.
        """
        prompt = self._build_generation_prompt(concept)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert Pygame + Pymunk simulation engineer. "
                    "Return ONLY a complete, self-contained Python script. "
                    "The script must be immediately runnable in a headless "
                    "Docker container and produce a valid MP4 file."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        start = time.monotonic()
        result: SimulationCode = await llm_routing.call_llm(
            model=_generation_model(),
            messages=messages,
            response_model=SimulationCode,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "Generated simulation code (%d chars) in %dms",
            len(result.code),
            latency_ms,
        )
        return result.code

    # ------------------------------------------------------------------
    # ABC: execute_simulation
    # ------------------------------------------------------------------

    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute code in the Docker sandbox (blocking call offloaded to executor)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: sandbox.execute_simulation(
                code,
                timeout=self._settings.sandbox_timeout_sec,
                memory_limit=self._settings.sandbox_memory_limit,
                cpu_limit=self._settings.sandbox_cpu_limit,
            ),
        )

    # ------------------------------------------------------------------
    # ABC: validate_output
    # ------------------------------------------------------------------

    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run Tier 1 (mandatory) + optional Tier 2 validation.

        Validation functions are synchronous (FFprobe-based), so we
        offload them to a thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: validation.validate_simulation(video_path, run_tier2=False),
        )

    # ------------------------------------------------------------------
    # ABC: adjust_parameters
    # ------------------------------------------------------------------

    async def adjust_parameters(
        self,
        code: str,
        validation_result: ValidationResult,
        iteration: int,
    ) -> str:
        """Fix code based on failed validation checks.

        Uses local LLM first (Mistral 7B) via ``sim-param-adjust``.
        Falls back to ``simulation-debugger`` (Claude Sonnet) if the
        local model produces code that still looks broken.
        """
        failed_summary = "\n".join(
            f"- {c.name}: {c.message} (value={c.value}, threshold={c.threshold})"
            for c in validation_result.failed_checks
        )
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a debugging specialist for Pygame + Pymunk simulations. "
                    "You will receive Python simulation code and a list of failed "
                    "validation checks. Fix the code so ALL checks pass. "
                    "Return the COMPLETE corrected script."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Iteration {iteration} — Fix Required\n\n"
                    f"### Failed Validation Checks\n{failed_summary}\n\n"
                    f"### Current Code\n```python\n{code}\n```\n\n"
                    "Fix the issues and return the complete corrected script."
                ),
            },
        ]

        def _quality_gate(result: Any) -> bool:
            """Basic quality check: code must contain key markers."""
            if not isinstance(result, AdjustedSimulationCode):
                return False
            c = result.code
            return (
                "pygame" in c
                and "pymunk" in c
                and "simulation.mp4" in c
                and len(c) > 500
            )

        primary, fallback = _adjustment_models()
        start = time.monotonic()
        result: AdjustedSimulationCode = await llm_routing.call_with_quality_fallback(
            primary_model=primary,
            fallback_model=fallback,
            messages=messages,
            validator=_quality_gate,
            response_model=AdjustedSimulationCode,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "Adjusted code (iter %d): %d changes in %dms — %s",
            iteration,
            len(result.changes_made),
            latency_ms,
            "; ".join(result.changes_made[:3]),
        )
        return result.code

    # ------------------------------------------------------------------
    # ABC: get_simulation_stats
    # ------------------------------------------------------------------

    async def get_simulation_stats(self, video_path: str) -> SimulationStats:
        """Extract statistics from the rendered video via FFprobe + stdout parsing."""
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
            peak_body_count=0,  # populated from stdout in run_loop
            avg_fps=round(fps, 1),
            min_fps=round(fps, 1),  # approximation — single FPS from container
            payoff_timestamp_sec=0.0,  # populated from stdout in run_loop
            total_frames=total_frames,
            file_size_bytes=file_size,
        )

    # ------------------------------------------------------------------
    # ABC: run_loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        concept: ConceptBrief,
        state: PipelineState,
    ) -> PipelineState:
        """Orchestrate the full simulation iteration loop.

        generate → execute → validate → (adjust → re-execute → re-validate)*
        Up to ``max_simulation_iterations`` total attempts.
        """
        max_iters = self._settings.max_simulation_iterations
        state.status = PipelineStatus.SIMULATION_PHASE
        state.simulation_iteration = 0

        # ----- Step 1: generate initial code -----
        logger.info("Generating initial simulation code for '%s'", concept.title)
        code = await self.generate_simulation(concept, state)
        state.simulation_code = code

        for iteration in range(1, max_iters + 1):
            state.simulation_iteration = iteration
            logger.info("Simulation iteration %d/%d", iteration, max_iters)

            # ----- Step 2: execute -----
            try:
                sim_result = await self.execute_simulation(code)
            except (SimulationTimeoutError, SimulationOOMError, SimulationExecutionError) as exc:
                logger.warning("Execution failed (iter %d): %s", iteration, exc)
                state.errors.append(f"iter{iteration}: {exc}")
                if iteration >= max_iters:
                    break
                # Synthesise a "failed execution" ValidationResult so adjust_parameters
                # can see the error and fix it.
                fake_validation = ValidationResult(
                    passed=False,
                    checks=[],
                    tier1_passed=False,
                    tier2_passed=None,
                )
                code = await self.adjust_parameters(
                    code, fake_validation, iteration,
                )
                state.simulation_code = code
                continue

            state.simulation_result = sim_result

            # ----- Step 3: find the video file -----
            video_path = self._find_video(sim_result)
            if video_path is None:
                msg = (
                    f"iter{iteration}: No MP4 file in output "
                    f"(files: {sim_result.output_files})"
                )
                logger.warning(msg)
                state.errors.append(msg)
                if iteration >= max_iters:
                    break
                fake_validation = ValidationResult(
                    passed=False,
                    checks=[],
                    tier1_passed=False,
                )
                code = await self.adjust_parameters(
                    code, fake_validation, iteration,
                )
                state.simulation_code = code
                continue

            state.raw_video_path = video_path

            # ----- Step 4: extract stats -----
            stats = await self.get_simulation_stats(video_path)
            # Enrich stats with stdout markers
            stats = self._enrich_stats_from_stdout(stats, sim_result.stdout)
            state.simulation_stats = stats

            # ----- Step 5: validate -----
            vresult = await self.validate_output(video_path)
            state.validation_result = vresult

            if vresult.passed:
                logger.info(
                    "Simulation passed validation on iteration %d: %s",
                    iteration,
                    vresult.summary,
                )
                return state

            logger.warning(
                "Validation failed (iter %d): %s — %s",
                iteration,
                vresult.summary,
                [c.name for c in vresult.failed_checks],
            )

            if iteration >= max_iters:
                state.errors.append(
                    f"Max iterations ({max_iters}) reached without passing validation"
                )
                break

            # ----- Step 6: adjust and retry -----
            code = await self.adjust_parameters(code, vresult, iteration)
            state.simulation_code = code

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_generation_prompt(self, concept: ConceptBrief) -> str:
        """Load the category prompt template and fill in concept details."""
        template_path = self._prompts_dir / f"{concept.category.value}.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}"
            )
        template = template_path.read_text(encoding="utf-8")

        # Simple Jinja-style substitution (double curly braces)
        replacements: dict[str, str] = {
            "title": concept.title,
            "visual_brief": concept.visual_brief,
            "body_count_initial": str(concept.simulation_requirements.body_count_initial),
            "body_count_max": str(concept.simulation_requirements.body_count_max),
            "interaction_type": concept.simulation_requirements.interaction_type,
            "colour_palette": json.dumps(concept.simulation_requirements.colour_palette),
            "background_colour": concept.simulation_requirements.background_colour,
            "special_effects": json.dumps(concept.simulation_requirements.special_effects),
            "target_duration_sec": str(concept.target_duration_sec),
            "seed": str(concept.seed or 42),
        }
        prompt = template
        for key, value in replacements.items():
            prompt = prompt.replace("{{ " + key + " }}", value)
            prompt = prompt.replace("{{" + key + "}}", value)

        return prompt

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
        """Parse PAYOFF_TIMESTAMP and PEAK_BODY_COUNT from simulation stdout."""
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
            # SimulationStats is frozen — rebuild with enriched values
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
