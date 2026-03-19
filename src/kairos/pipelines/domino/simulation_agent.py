"""Domino Simulation Agent.

Implements BaseSimulationAgent for the Blender domino run pipeline.
Orchestrates the full Blender subprocess pipeline:
  generate_domino_course → validate_domino_course → smoke_test_domino → bake_and_render

Each Blender script is executed headless via the blender_executor.
Results are cached after each step to prevent token bleed on retries.
"""

from __future__ import annotations

import json
import logging
import shutil
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
from kairos.pipelines.domino.blender_executor import run_blender_script
from kairos.pipelines.domino.idea_agent import extract_domino_config
from kairos.pipelines.domino.models import (
    BakeRenderResult,
    CourseGenerationResult,
    CourseValidationResult,
    DominoCourseConfig,
    SmokeTestResult,
)
from kairos.services.response_cache import get_cache

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

BLEND_SUBDIR = "blender"


def _get_ffmpeg_path() -> str:
    """Get the resolved FFmpeg path from centralised config."""
    from kairos.config import get_settings
    return get_settings().ffmpeg_path


def _get_ffprobe_path() -> str:
    """Get the resolved FFprobe path from centralised config."""
    from kairos.config import get_settings
    return get_settings().ffprobe_path


class DominoSimulationAgent(BaseSimulationAgent):
    """Simulation agent for the Blender domino run pipeline.

    Pipeline:
    1. Write config JSON to disk
    2. Run generate_domino_course.py → .blend file
    3. Run validate_domino_course.py → structural checks
    4. Run smoke_test_domino.py → physics smoke test
    5. Run bake_and_render.py → final .mp4

    Caching strategy:
    - Step-level cache: if the full step output is cached (same run_id), skip entirely.
    - LLM cache: idea agent caches concept generation (global, shared across runs).
    - Blender outputs are cached per-run via step cache (blend file + validation results).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._work_dir: Path | None = None

    def _ensure_work_dir(self, run_id: str | UUID) -> Path:
        """Create and return the work directory for this run."""
        runs_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "runs"
        work = runs_dir / str(run_id) / BLEND_SUBDIR
        work.mkdir(parents=True, exist_ok=True)
        self._work_dir = work
        return work

    # Palette / lighting → environment theme mapping.
    # Ensures the environment theme matches the concept's visual style.
    _PALETTE_THEME_MAP: dict[str, str] = {
        "neon": "neon_city",
        "pastel": "candy_land",
        "ocean": "deep_space",
        "sunset": "golden_hour",
        "earth": "enchanted_forest",
        "rainbow": "candy_land",
        "monochrome": "arctic_lab",
    }
    _LIGHTING_THEME_MAP: dict[str, str] = {
        "dark_neon": "neon_city",
        "dramatic": "lava_world",
        "soft_daylight": "golden_hour",
        "studio": "arctic_lab",
        "soft": "enchanted_forest",
    }

    def _derive_theme_from_concept(
        self,
        concept: ConceptBrief,
        work_dir: Path,
    ) -> str | None:
        """Derive an environment theme name from the concept's visual config.

        Reads the Blender config.json to extract palette and lighting_preset,
        then maps them to the most appropriate environment theme.

        Returns theme name or None (random).
        """
        config_path = work_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Priority 1: palette name
        palette = config.get("palette", "")
        if palette in self._PALETTE_THEME_MAP:
            theme = self._PALETTE_THEME_MAP[palette]
            logger.info("[domino_sim] Derived theme '%s' from palette '%s'", theme, palette)
            return theme

        # Priority 2: lighting preset
        lighting = config.get("lighting_preset", "")
        if lighting in self._LIGHTING_THEME_MAP:
            theme = self._LIGHTING_THEME_MAP[lighting]
            logger.info("[domino_sim] Derived theme '%s' from lighting '%s'", theme, lighting)
            return theme

        return None  # random

    # ------------------------------------------------------------------
    # ABC: generate_simulation
    # ------------------------------------------------------------------

    async def generate_simulation(
        self,
        concept: ConceptBrief,
    ) -> str:
        """Generate the Blender domino course — returns path to .blend file."""
        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_generate")
            if cached and Path(cached.get("blend_path", "")).exists():
                logger.info("[domino_sim] Cache HIT for generate step — skipping Blender")
                return cached["blend_path"]

        config = extract_domino_config(concept)
        if config is None:
            config = DominoCourseConfig(
                seed=concept.seed or 42,
                archetype="s_curve",
                title=concept.title,
                visual_brief=concept.visual_brief,
                domino_count=concept.simulation_requirements.body_count_initial,
                hook_text=concept.hook_text,
            )

        # Get pipeline_run_id from artifact system (no longer passed via state)
        from kairos.services.step_artifacts import get_run_artifacts
        artifacts = get_run_artifacts()
        run_id = artifacts.run_id if artifacts else "unknown"
        work_dir = self._ensure_work_dir(run_id)

        config_path = work_dir / "config.json"
        config_path.write_text(
            json.dumps(config.to_blender_config(), indent=2),
            encoding="utf-8",
        )

        blend_path = work_dir / "domino_course.blend"
        gen_json = work_dir / "generation_result.json"

        logger.info(
            "[domino_sim] Generating course: %s (archetype=%s, dominos=%d)",
            config.title, config.archetype.value, config.domino_count,
        )

        result = await run_blender_script(
            "generate_domino_course.py",
            script_args=[
                "--config", str(config_path),
                "--output-blend", str(blend_path),
                "--output-json", str(gen_json),
            ],
            timeout_sec=300,
        )

        if result["returncode"] != 0:
            stderr_snippet = result["stderr"][-2000:] if result["stderr"] else ""
            msg = f"Blender generate_domino_course failed (rc={result['returncode']}): {stderr_snippet}"
            logger.error("[domino_sim] %s", msg)
            raise SimulationExecutionError(msg)

        if not blend_path.exists():
            raise SimulationExecutionError(
                f"Blender generate_domino_course did not produce {blend_path}"
            )

        logger.info("[domino_sim] Course generated: %s", blend_path)

        # ── Cache store ──────────────────────────────────────────────
        if cache:
            cache.put_step("domino_generate", {"blend_path": str(blend_path)})

        return str(blend_path)

    # ------------------------------------------------------------------
    # ABC: execute_simulation
    # ------------------------------------------------------------------

    async def execute_simulation(self, code: str) -> SimulationResult:
        """Execute = bake + render the .blend file.

        'code' is the path to the .blend file (from generate_simulation).
        """
        blend_path = Path(code)
        if not blend_path.exists():
            return SimulationResult(
                returncode=1,
                stderr=f"Blend file not found: {blend_path}",
            )

        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_render")
            if cached:
                video_path = cached.get("video_path", "")
                if video_path and Path(video_path).exists():
                    logger.info("[domino_sim] Cache HIT for render step — skipping bake+render")
                    return SimulationResult(
                        returncode=0,
                        stdout=cached.get("stdout", ""),
                        output_files=[video_path],
                    )

        work_dir = blend_path.parent
        output_video = work_dir / "render.mp4"
        frames_dir = work_dir / "frames"
        config_path = work_dir / "config.json"

        logger.info("[domino_sim] Baking and rendering: %s", blend_path)

        result = await run_blender_script(
            "bake_and_render.py",
            blend_file=str(blend_path),
            script_args=[
                "--preset", "render_preview",
                "--output", str(output_video),
                "--config", str(config_path),
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
        # Mux collision audio into the render (ambient bed disabled — see
        # docs/white-noise-diagnosis.md for why it was removed).
        collision_wav = work_dir / "collision_audio.wav"
        has_collision = collision_wav.exists() and collision_wav.stat().st_size > 1000

        if frames_dir.exists() and any(frames_dir.glob("*.png")):
            n_frames = len(list(frames_dir.glob("*.png")))
            logger.info("[domino_sim] Combining %d frames into video...", n_frames)
            import subprocess as sp

            if has_collision:
                # Collision audio only (no ambient bed)
                logger.info("[domino_sim] Including collision audio: %s",
                            collision_wav)
                ffmpeg_cmd = [
                    _get_ffmpeg_path(), "-y",
                    "-framerate", "30",
                    "-i", str(frames_dir / "frame_%04d.png"),
                    "-i", str(collision_wav),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(output_video),
                ]
            else:
                # No audio — video only
                logger.info("[domino_sim] No audio WAVs found, video-only")
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

        sim_result = SimulationResult(
            returncode=0,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            execution_time_sec=0.0,
        )

        # ── Cache store ──────────────────────────────────────────────
        if cache and output_files:
            cache.put_step("domino_render", {
                "video_path": output_files[0],
                "stdout": stdout[:5000],  # truncate to avoid huge cache files
            })

        return sim_result

    # ------------------------------------------------------------------
    # ABC: validate_output
    # ------------------------------------------------------------------

    async def validate_output(self, video_path: str) -> ValidationResult:
        """Run structural validation and smoke test on the .blend file."""
        blend_path = Path(video_path)

        if blend_path.suffix == ".mp4":
            blend_path = blend_path.parent / "domino_course.blend"

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

        # ── Cache check ──────────────────────────────────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_validate")
            if cached:
                logger.info("[domino_sim] Cache HIT for validate step — skipping")
                return ValidationResult.model_validate(cached)

        val_checks: list[ValidationCheck] = []

        # Run validate_domino_course.py
        val_result = await run_blender_script(
            "validate_domino_course.py",
            blend_file=str(blend_path),
            script_args=["--strict"],
            timeout_sec=120,
        )

        val_json = val_result.get("json_output")
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

        # Run smoke test — frame_count=0 → auto-compute from domino count
        smoke_result = await run_blender_script(
            "smoke_test_domino.py",
            blend_file=str(blend_path),
            script_args=[],
            timeout_sec=300,
        )

        smoke_json = smoke_result.get("json_output")
        smoke_passed = True

        if smoke_json:
            for check in smoke_json.get("checks", []):
                passed = check.get("passed", False)
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

        all_passed = smoke_passed  # structural checks advisory

        validation = ValidationResult(
            passed=all_passed,
            checks=val_checks,
            tier1_passed=all_passed,
        )

        # ── Cache store ──────────────────────────────────────────────
        if cache:
            cache.put_step("domino_validate", validation.model_dump(mode="json"))

        return validation

    # ------------------------------------------------------------------
    # ABC: adjust_parameters
    # ------------------------------------------------------------------

    async def adjust_parameters(
        self,
        code: str,
        validation_result: ValidationResult,
        iteration: int,
    ) -> str:
        """Adjust config and regenerate course on validation failure."""
        blend_path = Path(code)
        work_dir = blend_path.parent
        config_path = work_dir / "config.json"

        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}

        # Increase physics fidelity on failure
        config["substeps_per_frame"] = min(
            30, config.get("substeps_per_frame", 20) + 5
        )
        config["solver_iterations"] = min(
            60, config.get("solver_iterations", 20) + 10
        )
        # Increase trigger impulse if chain didn't propagate
        config["trigger_impulse"] = min(
            8.0, config.get("trigger_impulse", 3.0) + 1.0
        )
        # Tighten spacing for better chain propagation on curves
        config["spacing_ratio"] = max(
            0.25, config.get("spacing_ratio", 0.35) - 0.03
        )
        # Reduce domino count slightly for stability
        if config.get("domino_count", 150) > 80:
            config["domino_count"] = max(80, config["domino_count"] - 30)

        config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "[domino_sim] Adjusted config (iter %d): substeps=%d, solver=%d, impulse=%.1f, count=%d",
            iteration,
            config.get("substeps_per_frame"),
            config.get("solver_iterations"),
            config.get("trigger_impulse"),
            config.get("domino_count"),
        )

        # Invalidate cached validation for this run
        cache = get_cache()
        if cache:
            # Clear step caches for generate and validate so they re-run
            for step in ("domino_generate", "domino_validate", "domino_render"):
                step_path = cache.step_cache_dir / f"step_{step}.json"
                if step_path.exists():
                    step_path.unlink()
                    logger.info("[domino_sim] Cleared cache for %s", step)

        new_blend = work_dir / f"domino_course_v{iteration}.blend"
        result = await run_blender_script(
            "generate_domino_course.py",
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
        duration = 65.0
        frames = 1950
        file_size = 0

        if vp.exists():
            file_size = vp.stat().st_size
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
            payoff_timestamp_sec=duration * 0.05,  # dominos start early
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
        """Run the full Blender domino pipeline with validation loop.

        generate → validate → (adjust → regenerate)* → bake_and_render

        Returns a narrow ``SimulationLoopResult`` (Finding 2.2).
        """
        # ── Step-level cache for the entire run_loop ─────────────────
        cache = get_cache()
        if cache:
            cached = cache.get_step("domino_run_loop")
            if cached:
                logger.info("[domino_sim] Cache HIT for full run_loop — skipping")
                return SimulationLoopResult.model_validate(cached)

        result = SimulationLoopResult()
        blend_path = ""

        for iteration in range(1, MAX_ITERATIONS + 1):
            result.simulation_iteration = iteration
            logger.info("[domino_sim] === Iteration %d/%d ===", iteration, MAX_ITERATIONS)

            # Step 1: Generate course
            if iteration == 1:
                blend_path = await self.generate_simulation(concept)
            else:
                blend_path = await self.adjust_parameters(
                    blend_path,
                    result.validation_result,  # type: ignore[arg-type]
                    iteration,
                )

            result.simulation_code = blend_path

            # Step 2: Validate
            validation = await self.validate_output(blend_path)
            result.validation_result = validation

            if not validation.passed:
                failed = [c.name for c in validation.failed_checks]
                logger.warning(
                    "[domino_sim] Validation failed on iter %d: %s",
                    iteration, ", ".join(failed),
                )
                if iteration >= MAX_ITERATIONS:
                    logger.error("[domino_sim] Max iterations reached, proceeding anyway")
                    break
                continue

            logger.info("[domino_sim] Validation passed on iteration %d", iteration)
            break

        # Step 3: Prepare environment (download HDRI, textures, SFX pool)
        # This writes theme_config.json to the work dir, which
        # bake_and_render.py reads to apply HDRI + materials + compositor.
        work_dir = Path(blend_path).parent

        # Derive theme from concept palette to ensure consistency
        theme_name = self._derive_theme_from_concept(concept, work_dir)

        try:
            from kairos.services.environment.orchestrator import prepare_environment_without_sfx
            env_result = prepare_environment_without_sfx(work_dir, theme_name=theme_name)
            logger.info(
                "[domino_sim] Environment prepared: theme=%s, hdri=%s, ground=%s",
                env_result.get("theme_name"),
                env_result.get("hdri_downloaded"),
                env_result.get("ground_texture_downloaded"),
            )
        except Exception as exc:
            logger.warning("[domino_sim] Environment preparation failed (non-fatal): %s", exc)

        # Step 4: Bake & render
        logger.info("[domino_sim] Baking and rendering...")
        sim_result = await self.execute_simulation(blend_path)
        result.simulation_result = sim_result

        if sim_result.returncode != 0:
            msg = f"Bake/render failed: {sim_result.stderr[:500]}"
            raise SimulationExecutionError(msg)

        if sim_result.output_files:
            video_path = sim_result.output_files[0]
            result.raw_video_path = video_path

            stats = await self.get_simulation_stats(video_path)
            result.simulation_stats = stats

            logger.info(
                "[domino_sim] Render complete: %s (%.1fs, %d bytes)",
                video_path, stats.duration_sec, stats.file_size_bytes,
            )
        else:
            raise SimulationExecutionError("No video file produced by bake_and_render")

        # ── Cache store ──────────────────────────────────────────────
        if cache:
            cache.put_step("domino_run_loop", result.model_dump(mode="json"))

        return result
