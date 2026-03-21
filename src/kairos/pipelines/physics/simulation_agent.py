"""Kairos Agent — Physics Simulation Agent.

Implements SimulationAgent for the Pygame + Pymunk physics pipeline.
Orchestrates: config generation → template injection → sandbox execution
→ validation → adjustment.

Architecture (config-based pipeline):
  1. LLM generates a JSON config matching the category schema
  2. A fixed template per category consumes the config
  3. Template handles all Pymunk/Pygame logic — LLM controls only creative params
  4. Pre-validation runs headless physics to verify chain completion before render

LLM routing:
  - Generation: ``simulation-first-pass`` (Claude Sonnet) via Instructor
  - Adjustment: ``simulation-debugger`` (Claude Sonnet) for config fixes

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
    AgentRunStatus,
    ConceptBrief,
    PipelineStatus,
    SimulationLoopResult,
    SimulationResult,
    SimulationStats,
    ValidationResult,
)
from kairos.schemas.simulation import (
    AdjustedSimulationConfig,
    SimulationConfigOutput,
)
from kairos.pipelines.physics.configs import CONFIG_REGISTRY
from kairos.pipelines.physics.template_loader import build_simulation_script
from kairos.ai.prompts.physics.builder import (
    build_user_prompt,
    load_system_prompt,
)
from kairos.ai.llm import routing as llm_routing
from kairos.engines.pymunk import sandbox
from kairos.services import validation
from kairos.ai.llm.config import get_step_config

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
_COMPLETION_RE = re.compile(r"COMPLETION_RATIO[=:]?\s*([\d.]+)")
_FALLEN_RE = re.compile(r"FALLEN[=:]?\s*(\d+)/(\d+)")


class PhysicsSimulationAgent(SimulationAgent):
    """Simulation agent for the *Oddly Satisfying Physics* pipeline.

    Uses the config-based template architecture:
    1. LLM generates a JSON config (creative params only)
    2. Fixed template handles physics, rendering, FFmpeg
    3. Headless pre-validation catches chain stalls before render
    4. Validation loop adjusts config (not code) on failure
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._current_config: dict[str, Any] | None = None
        self._iteration_history: list[str] = []  # track what was tried each iter

    # ------------------------------------------------------------------
    # ABC: generate_simulation
    # ------------------------------------------------------------------

    async def generate_simulation(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate simulation config via LLM, then inject into template.

        Learning-loop enhancements (AI review §1–§6):
        - Injects few-shot examples from verified training data
        - Injects category knowledge (parameter ranges, failure modes)
        - Injects static validation rules as prompt instructions
        - Stores reasoning for later training example capture

        Returns the complete runnable Python script (template + config).
        """
        category = concept.category.value

        # Get the config schema for this category
        config_cls = CONFIG_REGISTRY.get(category)
        if config_cls is None:
            raise ValueError(f"No config schema for category '{category}'")

        # Phase 4: config_schema removed from prompt vars — Instructor injects
        # the schema automatically via response_model.

        prompt_vars = {
            "title": concept.title,
            "visual_brief": concept.visual_brief,
            "category": category,
            "body_count_initial": str(concept.simulation_requirements.body_count_initial),
            "body_count_max": str(concept.simulation_requirements.body_count_max),
            "interaction_type": concept.simulation_requirements.interaction_type,
            "colour_palette": json.dumps(concept.simulation_requirements.colour_palette),
            "background_colour": concept.simulation_requirements.background_colour,
            "special_effects": json.dumps(concept.simulation_requirements.special_effects),
            "target_duration_sec": str(concept.target_duration_sec),
            "seed": str(concept.seed or 42),
        }

        # --- Learning loop: assemble extra context for the prompt ---
        extra_context_parts: list[str] = []
        try:
            from kairos.ai.learning.learning_loop import (
                format_few_shot_prompt,
                get_category_knowledge_for_prompt,
                get_few_shot_examples,
                get_validation_rules_prompt,
            )

            # 1. Static validation rules → tell the LLM what NOT to do
            extra_context_parts.append(get_validation_rules_prompt())

            # 2. Few-shot examples (only verified ones)
            examples = await get_few_shot_examples(
                pipeline="physics", category=category, limit=2,
            )
            few_shot_text = format_few_shot_prompt(examples)
            if few_shot_text:
                extra_context_parts.append(few_shot_text)

            # 3. Category knowledge
            ck_text = await get_category_knowledge_for_prompt("physics", category)
            if ck_text:
                extra_context_parts.append(ck_text)
        except Exception as exc:
            logger.debug("Learning loop context injection skipped: %s", exc)

        user_prompt_text = build_user_prompt("simulation_config", prompt_vars).text
        if extra_context_parts:
            user_prompt_text += "\n\n" + "\n\n".join(extra_context_parts)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": load_system_prompt("simulation_config").text},
            {"role": "user", "content": user_prompt_text},
        ]

        start = time.monotonic()
        result: SimulationConfigOutput = await llm_routing.call_llm(
            model=_generation_model(),
            messages=messages,
            response_model=SimulationConfigOutput,
            cache_step="simulation_config_gen",
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        config = result.config
        logger.info(
            "Generated simulation config (%d keys) in %dms — %s",
            len(config),
            latency_ms,
            result.reasoning[:200],
        )

        # Validate config against the Pydantic schema
        try:
            validated = config_cls(**config)
            config = validated.model_dump()
        except Exception as exc:
            logger.warning("Config validation failed, using raw config: %s", exc)
            # Fill in defaults from the schema
            defaults = config_cls.model_construct().model_dump()
            merged = {**defaults, **config}
            config = merged

        self._current_config = config

        # Build the runnable script from template + config
        script = build_simulation_script(category, config)

        return script

    # ------------------------------------------------------------------
    # ABC: execute_simulation
    # ------------------------------------------------------------------

    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute code in the Docker sandbox (async subprocess)."""
        return await sandbox.execute_simulation(
            code,
            timeout=self._settings.sandbox_timeout_sec,
            memory_limit=self._settings.sandbox_memory_limit,
            cpu_limit=self._settings.sandbox_cpu_limit,
        )

    # ------------------------------------------------------------------
    # ABC: validate_output
    # ------------------------------------------------------------------

    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run Tier 1 (mandatory) + optional Tier 2 validation.

        Phase 4: validation is now fully async (no executor needed).
        """
        return await validation.validate_simulation(
            video_path,
            run_tier2=False,
            skip_checks={"audio_present"},  # audio added by video_editor
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
        """Fix config based on failed validation checks, then rebuild script.

        Learning-loop enhancements (AI review §2, §7, §8):
        - Builds structured ValidationFeedback with quantitative deltas
        - Adds urgency signal (minor_tweak → fundamental_rethink)
        - Includes AST-extracted code structure analysis
        - Feeds iteration history so the LLM doesn't repeat failed fixes
        """
        failed_summary = "\n".join(
            f"- {c.name}: {c.message} (value={c.value}, threshold={c.threshold})"
            for c in validation_result.failed_checks
        )

        # Include stdout-based completion info if available
        stdout_info = ""
        if hasattr(validation_result, "stdout") and validation_result.stdout:
            stdout_info = f"\n\n### Simulation Stdout\n```\n{validation_result.stdout[-1000:]}\n```"

        current_config_json = json.dumps(self._current_config or {}, indent=2)

        # Determine category from config
        category = "domino_chain"  # default
        if self._current_config:
            # Try to infer from config keys
            if "domino_count" in self._current_config:
                category = "domino_chain"
            elif "ball_count" in self._current_config:
                category = "ball_pit"
            elif "marble_count" in self._current_config:
                category = "marble_funnel"
            elif "block_mass" in self._current_config:
                category = "destruction"

        config_cls = CONFIG_REGISTRY.get(category)
        # Phase 4: config_schema removed — Instructor injects it via response_model

        # --- Learning loop: structured feedback + AST analysis ---
        extra_adjustment_context = ""
        try:
            from kairos.ai.learning.learning_loop import build_validation_feedback
            from kairos.services.ast_extractor import extract_parameters

            # Structured feedback with quantitative deltas & urgency
            feedback = build_validation_feedback(
                validation_result,
                iteration=iteration,
                max_iterations=self._settings.max_simulation_iterations,
                iteration_history=self._iteration_history,
            )
            extra_adjustment_context += "\n\n" + feedback.to_prompt_text()

            # AST-based code structure analysis
            ast_params = extract_parameters(code)
            ast_text = ast_params.to_feedback_text()
            if ast_text:
                extra_adjustment_context += "\n\n" + ast_text
        except Exception as exc:
            logger.debug("Structured feedback generation skipped: %s", exc)

        user_prompt_text = build_user_prompt("simulation_config_adjust", {
            "iteration": str(iteration),
            "failed_summary": failed_summary,
            "stdout_info": stdout_info,
            "current_config": current_config_json,
            "category": category,
        }).text + extra_adjustment_context

        messages: list[dict[str, str]] = [
            {"role": "system", "content": load_system_prompt("simulation_config_adjust").text},
            {"role": "user", "content": user_prompt_text},
        ]

        _, fallback = _adjustment_models()
        start = time.monotonic()
        result: AdjustedSimulationConfig = await llm_routing.call_llm(
            model=fallback,  # Use cloud model for config adjustment
            messages=messages,
            response_model=AdjustedSimulationConfig,
            cache_step=f"simulation_config_adjust_iter{iteration}",
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "Adjusted config (iter %d): %d changes in %dms — %s",
            iteration,
            len(result.changes_made),
            latency_ms,
            "; ".join(result.changes_made[:3]),
        )

        # Record what was tried this iteration for history feedback
        self._iteration_history.append(
            f"Changes: {'; '.join(result.changes_made[:5])}"
            + (f" — Reasoning: {result.reasoning[:200]}" if result.reasoning else "")
        )

        # Validate and merge the adjusted config
        new_config = result.config
        if config_cls:
            try:
                validated = config_cls(**new_config)
                new_config = validated.model_dump()
            except Exception as exc:
                logger.warning("Adjusted config validation failed: %s", exc)
                # Merge with current config as fallback
                merged = {**(self._current_config or {}), **new_config}
                new_config = merged

        self._current_config = new_config

        # Rebuild script from template + adjusted config
        script = build_simulation_script(category, new_config)
        return script

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
    ) -> SimulationLoopResult:
        """Orchestrate the full simulation iteration loop.

        generate → execute → validate → (adjust → re-execute → re-validate)*
        Up to ``max_simulation_iterations`` total attempts.

        Returns a narrow ``SimulationLoopResult`` (Finding 2.2).
        """
        max_iters = self._settings.max_simulation_iterations
        result = SimulationLoopResult()

        # ----- Step 1: generate initial code -----
        logger.info("Generating initial simulation code for '%s'", concept.title)
        generation_attempts = 0
        max_generation_attempts = 2
        code: str | None = None
        while generation_attempts < max_generation_attempts:
            generation_attempts += 1
            try:
                code = await self.generate_simulation(concept)
                break
            except (TimeoutError, Exception) as exc:
                logger.warning(
                    "Code generation attempt %d/%d failed: %s",
                    generation_attempts,
                    max_generation_attempts,
                    exc,
                )
                result.errors.append(f"code_gen_attempt{generation_attempts}: {exc}")
                if generation_attempts >= max_generation_attempts:
                    raise SimulationExecutionError(
                        f"Simulation code generation failed after "
                        f"{max_generation_attempts} attempts: {exc}"
                    ) from exc
        assert code is not None
        result.simulation_code = code

        for iteration in range(1, max_iters + 1):
            result.simulation_iteration = iteration
            logger.info("Simulation iteration %d/%d", iteration, max_iters)

            # ----- Step 2: execute -----
            try:
                sim_result = await self.execute_simulation(code)
            except (SimulationTimeoutError, SimulationOOMError, SimulationExecutionError) as exc:
                logger.warning("Execution failed (iter %d): %s", iteration, exc)
                result.errors.append(f"iter{iteration}: {exc}")
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
                result.simulation_code = code
                continue

            result.simulation_result = sim_result

            # ----- Step 3: find the video file -----
            video_path = self._find_video(sim_result)
            if video_path is None:
                msg = (
                    f"iter{iteration}: No MP4 file in output "
                    f"(files: {sim_result.output_files})"
                )
                logger.warning(msg)
                if sim_result.stderr:
                    logger.error("iter%d sandbox stderr:\n%s", iteration, sim_result.stderr[-2000:])
                if sim_result.stdout:
                    logger.info("iter%d sandbox stdout:\n%s", iteration, sim_result.stdout[-1000:])
                logger.info("iter%d sandbox returncode: %s", iteration, sim_result.returncode)
                result.errors.append(msg)
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
                result.simulation_code = code
                continue

            result.raw_video_path = video_path

            # ----- Step 4: extract stats -----
            stats = await self.get_simulation_stats(video_path)
            # Enrich stats with stdout markers
            stats = self._enrich_stats_from_stdout(stats, sim_result.stdout)
            result.simulation_stats = stats

            # ----- Step 4.5: stdout-based completion check -----
            completion_ok, completion_ratio, completion_msg = (
                self._check_completion_from_stdout(sim_result.stdout)
            )
            logger.info("Completion check: %s (ratio=%.2f)", completion_msg, completion_ratio)

            if not completion_ok:
                logger.warning(
                    "Simulation failed completion check (iter %d): %s",
                    iteration,
                    completion_msg,
                )
                if iteration >= max_iters:
                    result.errors.append(f"Completion check failed: {completion_msg}")
                    break
                # Create a synthetic validation failure for the adjustment LLM
                from kairos.schemas.contracts import ValidationCheck
                fake_check = ValidationCheck(
                    name="completion_ratio",
                    passed=False,
                    message=completion_msg,
                    value=completion_ratio,
                    threshold=0.75,
                )
                fake_validation = ValidationResult(
                    passed=False,
                    checks=[fake_check],
                    tier1_passed=False,
                )
                code = await self.adjust_parameters(
                    code, fake_validation, iteration,
                )
                result.simulation_code = code
                continue

            # ----- Step 5: validate -----
            vresult = await self.validate_output(video_path)
            result.validation_result = vresult

            if vresult.passed:
                logger.info(
                    "Simulation passed validation on iteration %d: %s",
                    iteration,
                    vresult.summary,
                )
                return result

            logger.warning(
                "Validation failed (iter %d): %s — %s",
                iteration,
                vresult.summary,
                [c.name for c in vresult.failed_checks],
            )

            if iteration >= max_iters:
                result.errors.append(
                    f"Max iterations ({max_iters}) reached without passing validation"
                )
                break

            # ----- Step 6: adjust and retry -----
            code = await self.adjust_parameters(code, vresult, iteration)
            result.simulation_code = code

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_generation_prompt(self, concept: ConceptBrief) -> str:
        """Build the category-specific config generation prompt.

        This is a legacy helper — the main ``generate_simulation`` now
        builds the prompt directly with config schema info.
        """
        variables: dict[str, str] = {
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
        from kairos.ai.prompts.physics.builder import build_simulation_prompt
        return build_simulation_prompt(concept.category.value, variables).text

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
        """Parse PAYOFF_TIMESTAMP, PEAK_BODY_COUNT, COMPLETION_RATIO from stdout."""
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

    @staticmethod
    def _check_completion_from_stdout(stdout: str) -> tuple[bool, float, str]:
        """Check if the simulation completed successfully based on stdout markers.

        Returns (passed, completion_ratio, message).
        """
        # Check completion ratio
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

        # Check fallen count (domino-specific)
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

        # Check for pre-validation failure
        if "ERROR:" in stdout and "pre-validation" in stdout.lower():
            return False, 0.0, "Template pre-validation failed"

        # No completion markers found — assume success
        return True, 1.0, "No completion markers (assumed OK)"
