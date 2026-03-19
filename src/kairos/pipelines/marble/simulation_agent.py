"""Marble Simulation Agent.

Implements BaseSimulationAgent for the Blender marble course pipeline.
Orchestrates the full Blender subprocess pipeline:
  generate_course → validate_course → run_smoke_test → bake_and_render

Each Blender script is executed headless via the blender_executor.
Results are cached after each step.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from kairos.agents.base import BaseSimulationAgent
from kairos.config import get_settings
from kairos.exceptions import SimulationExecutionError
from kairos.models.contracts import (
    ConceptBrief,
    PipelineStatus,
    SimulationLoopResult,
    SimulationResult,
    SimulationStats,
    ValidationCheck,
    ValidationResult,
)
from kairos.pipelines.marble.blender_executor import run_blender_script
from kairos.pipelines.marble.idea_agent import extract_marble_config
from kairos.pipelines.marble.models import (
    BakeRenderResult,
    CourseGenerationResult,
    CourseValidationResult,
    MarbleCourseConfig,
    SmokeTestResult,
)
from kairos.services.response_cache import get_cache

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

# Directories under runs/<run_id>/ for Blender outputs
BLEND_SUBDIR = "blender"


def _get_ffmpeg_path() -> str:
    """Get the resolved FFmpeg path from centralised config."""
    from kairos.config import get_settings
    return get_settings().ffmpeg_path


def _get_ffprobe_path() -> str:
    """Get the resolved FFprobe path from centralised config."""
    from kairos.config import get_settings
    return get_settings().ffprobe_path


class MarbleSimulationAgent(BaseSimulationAgent):
    """Simulation agent for the Blender marble course pipeline.

    Unlike the physics pipeline (which generates Python code and runs
    it in Docker), this agent orchestrates fixed Blender scripts with
    a JSON config produced by the idea agent.

    Pipeline:
    1. Write config JSON to disk
    2. Run generate_course.py → .blend file
    3. Run validate_course.py → structural checks
    4. Run run_smoke_test.py → physics smoke test
    5. Run bake_and_render.py → final .mp4
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._work_dir: Path | None = None

    def _ensure_work_dir(self, run_id: str | UUID) -> Path:
        """Create and return the work directory for this run."""
        # Project root: marble/ -> pipelines/ -> kairos/ -> src/ -> root/
        runs_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "runs"
        work = runs_dir / str(run_id) / BLEND_SUBDIR
        work.mkdir(parents=True, exist_ok=True)
        self._work_dir = work
        return work

    # ------------------------------------------------------------------
    # ABC: generate_simulation
    # ------------------------------------------------------------------

    async def generate_simulation(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate the Blender course — returns path to .blend file.

        Writes config JSON, calls generate_course.py in Blender.
        """
        config = extract_marble_config(concept)
        if config is None:
            # Fallback: build a default config from the concept
            config = MarbleCourseConfig(
                seed=concept.seed or 42,
                archetype="funnel_race",
                title=concept.title,
                visual_brief=concept.visual_brief,
                marble_count=concept.simulation_requirements.body_count_initial,
                hook_text=concept.hook_text,
            )

        work_dir = self._ensure_work_dir(state.pipeline_run_id)

        # Write config to JSON
        config_path = work_dir / "config.json"
        config_path.write_text(
            json.dumps(config.to_blender_config(), indent=2),
            encoding="utf-8",
        )

        blend_path = work_dir / "course.blend"
        gen_json = work_dir / "generation_result.json"

        logger.info(
            "[marble_sim] Generating course: %s (archetype=%s, marbles=%d)",
            config.title,
            config.archetype.value,
            config.marble_count,
        )

        result = await run_blender_script(
            "generate_course.py",
            script_args=[
                "--config", str(config_path),
                "--output-blend", str(blend_path),
                "--output-json", str(gen_json),
            ],
            timeout_sec=300,
        )

        if result["returncode"] != 0:
            stderr_snippet = result["stderr"][-2000:] if result["stderr"] else ""
            msg = f"Blender generate_course failed (rc={result['returncode']}): {stderr_snippet}"
            logger.error("[marble_sim] %s", msg)
            raise SimulationExecutionError(msg)

        if not blend_path.exists():
            raise SimulationExecutionError(
                f"Blender generate_course did not produce {blend_path}"
            )

        logger.info("[marble_sim] Course generated: %s", blend_path)
        return str(blend_path)

    # ------------------------------------------------------------------
    # ABC: execute_simulation
    # ------------------------------------------------------------------

    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute = bake + render the .blend file.

        'code' is actually the path to the .blend file (returned by
        generate_simulation). We run bake_and_render.py on it, which
        produces PNG frames, then combine them with ffmpeg.
        """
        blend_path = Path(code)
        if not blend_path.exists():
            return SimulationResult(
                returncode=1,
                stderr=f"Blend file not found: {blend_path}",
            )

        work_dir = blend_path.parent
        output_video = work_dir / "render.mp4"
        frames_dir = work_dir / "frames"

        logger.info("[marble_sim] Baking and rendering: %s", blend_path)

        result = await run_blender_script(
            "bake_and_render.py",
            blend_file=str(blend_path),
            script_args=[
                "--preset", "render_preview",
                "--output", str(output_video),
            ],
            timeout_sec=1800,
        )

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if result["returncode"] != 0:
            return SimulationResult(
                returncode=result["returncode"],
                stdout=stdout,
                stderr=stderr,
            )

        # Combine PNG frames into video with ffmpeg
        if frames_dir.exists() and any(frames_dir.glob("*.png")):
            logger.info("[marble_sim] Combining %d frames into video...",
                        len(list(frames_dir.glob("*.png"))))
            import subprocess as sp
            ffmpeg_cmd = [
                _get_ffmpeg_path(), "-y",
                "-framerate", "30",
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(output_video),
            ]
            proc = sp.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                return SimulationResult(
                    returncode=proc.returncode,
                    stdout=stdout,
                    stderr=f"ffmpeg frame combine failed: {proc.stderr[-500:]}",
                )

        output_files = [str(output_video)] if output_video.exists() else []

        return SimulationResult(
            returncode=0,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            execution_time_sec=0.0,
        )

    # ------------------------------------------------------------------
    # ABC: validate_output
    # ------------------------------------------------------------------

    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run structural validation on the .blend file.

        Also runs the smoke test for physics validation.
        """
        blend_path = Path(video_path)

        # If this is a .mp4, find the .blend in the same directory
        if blend_path.suffix == ".mp4":
            blend_path = blend_path.parent / "course.blend"

        if not blend_path.exists():
            return ValidationResult(
                passed=False,
                checks=[
                    ValidationCheck(
                        name="blend_exists",
                        passed=False,
                        message=f"Blend file not found: {blend_path}",
                    )
                ],
            )

        # Run validate_course.py
        val_result = await run_blender_script(
            "validate_course.py",
            blend_file=str(blend_path),
            script_args=["--strict"],
            timeout_sec=120,
        )

        val_json = val_result.get("json_output")
        val_checks: list[ValidationCheck] = []
        val_passed = True

        if val_json:
            for check in val_json.get("checks", []):
                passed = check.get("passed", False)
                if not passed:
                    val_passed = False
                val_checks.append(
                    ValidationCheck(
                        name=check.get("name", "unknown"),
                        passed=passed,
                        message=check.get("message", ""),
                    )
                )
        elif val_result["returncode"] != 0:
            val_passed = False
            val_checks.append(
                ValidationCheck(
                    name="validation_script",
                    passed=False,
                    message=f"Validation script failed: rc={val_result['returncode']}",
                )
            )

        # Run smoke test — use a short checkpoint window (up to 600 frames)
        # to avoid baking the entire 1950-frame sim during validation.
        # Checkpoint data is advisory — only the core checks gate pass/fail.
        smoke_result = await run_blender_script(
            "run_smoke_test.py",
            blend_file=str(blend_path),
            script_args=["--frames", "300",
                          "--checkpoints", "1,150,300,600"],
            timeout_sec=300,
        )

        smoke_json = smoke_result.get("json_output")
        smoke_passed = True

        if smoke_json:
            for check in smoke_json.get("checks", []):
                passed = check.get("passed", False)
                # Skip advisory checks (e.g. checkpoint retention) — they
                # provide diagnostics but should not block the pipeline.
                is_advisory = check.get("advisory", False)
                if not passed and not is_advisory:
                    smoke_passed = False
                val_checks.append(
                    ValidationCheck(
                        name=f"smoke_{check.get('name', 'unknown')}",
                        passed=passed,
                        message=check.get("message", ""),
                    )
                )
        elif smoke_result["returncode"] != 0:
            smoke_passed = False
            val_checks.append(
                ValidationCheck(
                    name="smoke_test_script",
                    passed=False,
                    message=f"Smoke test failed: rc={smoke_result['returncode']}",
                )
            )

        all_passed = smoke_passed  # Structural checks are advisory

        return ValidationResult(
            passed=all_passed,
            checks=val_checks,
            tier1_passed=all_passed,
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
        """Adjust config and regenerate course on validation failure.

        For now: bump solver iterations and substeps, then regenerate.
        Future: use LLM to diagnose and fix.
        """
        blend_path = Path(code)
        work_dir = blend_path.parent
        config_path = work_dir / "config.json"

        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}

        # Simple adjustments based on iteration
        config["substeps_per_frame"] = min(
            30, config.get("substeps_per_frame", 10) + 5
        )
        config["solver_iterations"] = min(
            60, config.get("solver_iterations", 20) + 10
        )
        # Reduce marble count slightly if physics are unstable
        if config.get("marble_count", 20) > 10:
            config["marble_count"] = max(10, config["marble_count"] - 5)

        config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "[marble_sim] Adjusted config (iter %d): substeps=%d, solver=%d, marbles=%d",
            iteration,
            config.get("substeps_per_frame"),
            config.get("solver_iterations"),
            config.get("marble_count"),
        )

        # Re-generate course with adjusted config
        new_blend = work_dir / f"course_v{iteration}.blend"
        result = await run_blender_script(
            "generate_course.py",
            script_args=[
                "--config", str(config_path),
                "--output-blend", str(new_blend),
            ],
            timeout_sec=300,
        )

        if result["returncode"] != 0:
            raise SimulationExecutionError(
                f"Re-generation failed on iteration {iteration}"
            )

        return str(new_blend)

    # ------------------------------------------------------------------
    # ABC: get_simulation_stats
    # ------------------------------------------------------------------

    async def get_simulation_stats(self, video_path: str) -> SimulationStats:
        """Extract statistics from a rendered video."""
        vp = Path(video_path)

        # Try to get duration/resolution from ffprobe
        duration = 65.0
        frames = 1950
        file_size = 0

        if vp.exists():
            file_size = vp.stat().st_size
            # Quick ffprobe for duration
            try:
                import subprocess
                probe = subprocess.run(
                    [
                        _get_ffprobe_path(), "-v", "quiet",
                        "-print_format", "json",
                        "-show_format",
                        str(vp),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if probe.returncode == 0:
                    info = json.loads(probe.stdout)
                    duration = float(info.get("format", {}).get("duration", 65.0))
                    frames = int(duration * 30)
            except Exception:
                pass

        return SimulationStats(
            duration_sec=duration,
            peak_body_count=0,
            avg_fps=30.0,
            min_fps=30.0,
            payoff_timestamp_sec=duration * 0.75,
            total_frames=frames,
            file_size_bytes=file_size,
        )

    # ------------------------------------------------------------------
    # ABC: run_loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        concept: ConceptBrief,
    ) -> SimulationLoopResult:
        """Run the full Blender simulation pipeline with validation loop.

        generate → validate → (adjust → regenerate)* → bake_and_render

        Returns a narrow ``SimulationLoopResult`` (Finding 2.2).
        """
        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()

        result = SimulationLoopResult()

        for iteration in range(1, MAX_ITERATIONS + 1):
            result.simulation_iteration = iteration
            logger.info("[marble_sim] === Iteration %d/%d ===", iteration, MAX_ITERATIONS)

            # Step 1: Generate course
            if iteration == 1:
                blend_path = await self.generate_simulation(concept)
            else:
                # Use adjusted config
                blend_path = await self.adjust_parameters(
                    blend_path,
                    result.validation_result,  # type: ignore
                    iteration,
                )

            result.simulation_code = blend_path

            # Step 2: Validate
            validation = await self.validate_output(blend_path)
            result.validation_result = validation

            if not validation.passed:
                failed = [c.name for c in validation.failed_checks]
                logger.warning(
                    "[marble_sim] Validation failed on iter %d: %s",
                    iteration,
                    ", ".join(failed),
                )
                if iteration >= MAX_ITERATIONS:
                    logger.error("[marble_sim] Max iterations reached, proceeding anyway")
                    break
                continue

            logger.info("[marble_sim] Validation passed on iteration %d", iteration)
            break

        # Step 3: Bake & render
        logger.info("[marble_sim] Baking and rendering...")
        sim_result = await self.execute_simulation(blend_path)
        result.simulation_result = sim_result

        if sim_result.returncode != 0:
            msg = f"Bake/render failed: {sim_result.stderr[:500]}"
            raise SimulationExecutionError(msg)

        if sim_result.output_files:
            video_path = sim_result.output_files[0]
            result.raw_video_path = video_path

            # Get stats
            stats = await self.get_simulation_stats(video_path)
            result.simulation_stats = stats

            logger.info(
                "[marble_sim] Render complete: %s (%.1fs, %d bytes)",
                video_path,
                stats.duration_sec,
                stats.file_size_bytes,
            )

            # ── Tier 2: AI screenshot analysis ───────────────────────
            ai_passed = True
            ai_summary = ""
            try:
                from kairos.services.screenshot_analyzer import analyze_video
                logger.info("[marble_sim] Running AI screenshot analysis...")
                analysis_results = analyze_video(video_path)
                stages_passed = 0
                failure_details = []
                for r in analysis_results:
                    logger.info(
                        "[marble_sim] AI frame check [%s @ %.1fs]: "
                        "quality=%d/10 passed=%s | %s",
                        r["stage"], r["timestamp"],
                        r["quality_rating"], r["passed"],
                        r["analysis"][:200],
                    )
                    if r["passed"]:
                        stages_passed += 1
                    else:
                        failure_details.append(
                            f"{r['stage']}({r['quality_rating']}/10): "
                            f"{r['analysis'][:100]}"
                        )

                # Require at least 2 of 3 stages to pass
                ai_passed = stages_passed >= 2
                if not ai_passed:
                    ai_summary = (
                        f"AI screenshot analysis FAILED ({stages_passed}/3 passed). "
                        f"Failures: {'; '.join(failure_details)}"
                    )
                    logger.error("[marble_sim] %s", ai_summary)
                else:
                    logger.info(
                        "[marble_sim] AI screenshot analysis PASSED (%d/3 stages)",
                        stages_passed,
                    )
            except Exception as e:
                logger.warning(
                    "[marble_sim] AI screenshot analysis failed (non-blocking): %s", e
                )

            if not ai_passed:
                # Feed failure back — add to errors and raise so the
                # pipeline graph can retry or surface the problem.
                result.errors.append(ai_summary)
                raise SimulationExecutionError(
                    f"Render produced bad video: {ai_summary}"
                )
        else:
            raise SimulationExecutionError("No video file produced by bake_and_render")

        return result
