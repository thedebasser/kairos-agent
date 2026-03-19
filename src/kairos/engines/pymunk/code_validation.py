"""Kairos Agent — Static Code Validation for Simulation Scripts.

Checks generated simulation code for common Pymunk/Pygame failure modes
BEFORE sending it to the Docker sandbox. Catches ~80% of broken code
without needing to run it.

Based on analysis of 558+ incorrect LLM code generation attempts.
"""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger(__name__)


class CodeValidationResult:
    """Result of static code validation."""

    def __init__(self) -> None:
        self.passed = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def summary(self) -> str:
        if self.passed and not self.warnings:
            return "All checks passed"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return "; ".join(parts)


def validate_simulation_code(code: str) -> CodeValidationResult:
    """Run all static validation checks on generated simulation code.

    Checks (based on the 6 most common LLM failure modes):
    1. Syntax: valid Python
    2. Coordinate system: positive_y_is_up = False or Y-down gravity
    3. Physics loop: space.step() present
    4. Display init: pygame.init() present
    5. Output: FFmpeg pipe or video output path
    6. Moment of inertia: uses helper functions, not raw numbers

    Args:
        code: The generated Python script as a string.

    Returns:
        CodeValidationResult with errors and warnings.
    """
    result = CodeValidationResult()

    # 1. Syntax check
    _check_syntax(code, result)
    if not result.passed:
        return result  # No point continuing if syntax is broken

    # 2. Coordinate system
    _check_coordinate_system(code, result)

    # 3. Physics stepping
    _check_physics_step(code, result)

    # 4. Pygame initialisation
    _check_pygame_init(code, result)

    # 5. Video output
    _check_video_output(code, result)

    # 6. Moment of inertia
    _check_moment_usage(code, result)

    # 7. Event loop (not needed for headless but check for display.set_mode)
    _check_headless(code, result)

    # 8. Required stdout markers
    _check_stdout_markers(code, result)

    logger.info(
        "Code validation: %s (%d errors, %d warnings)",
        "PASSED" if result.passed else "FAILED",
        len(result.errors),
        len(result.warnings),
    )
    for err in result.errors:
        logger.warning("  [ERROR] %s", err)
    for warn in result.warnings:
        logger.info("  [WARN] %s", warn)

    return result


def _check_syntax(code: str, result: CodeValidationResult) -> None:
    """Verify code is syntactically valid Python."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        result.add_error(f"Syntax error at line {e.lineno}: {e.msg}")


def _check_coordinate_system(code: str, result: CodeValidationResult) -> None:
    """Verify Y-down coordinate convention is used."""
    has_y_down_flag = "positive_y_is_up" in code and (
        "positive_y_is_up = False" in code
        or "positive_y_is_up=False" in code
    )
    has_y_down_gravity = bool(
        re.search(r"gravity\s*=\s*\(\s*0\s*,\s*[89]\d{2}\s*\)", code)
    )
    has_y_up_gravity = bool(
        re.search(r"gravity\s*=\s*\(\s*0\s*,\s*-\s*[89]\d{2}\s*\)", code)
    )

    if has_y_up_gravity and not has_y_down_flag:
        result.add_warning(
            "Using Y-up gravity without positive_y_is_up=False — "
            "coordinate conversion bugs likely. Prefer Y-down convention."
        )
    elif not has_y_down_gravity and not has_y_up_gravity:
        result.add_error(
            "No gravity setting found. Must set space.gravity = (0, 900) "
            "for Y-down convention."
        )

    if not has_y_down_flag and not has_y_up_gravity:
        result.add_warning(
            "positive_y_is_up = False not found. Recommended for Y-down coords."
        )


def _check_physics_step(code: str, result: CodeValidationResult) -> None:
    """Verify space.step() is called in the code."""
    if "space.step(" not in code:
        result.add_error(
            "No space.step() call found — physics will never advance. "
            "Must call space.step(1/60) in the render loop."
        )


def _check_pygame_init(code: str, result: CodeValidationResult) -> None:
    """Verify pygame is initialised."""
    if "pygame.init()" not in code:
        result.add_error(
            "pygame.init() not found — Pygame must be initialised "
            "before creating surfaces."
        )


def _check_video_output(code: str, result: CodeValidationResult) -> None:
    """Verify video output mechanism exists."""
    has_ffmpeg = "ffmpeg" in code.lower()
    has_output_path = "simulation.mp4" in code or "/workspace/output" in code
    has_pipe = "subprocess.Popen" in code or "subprocess.run" in code

    if not has_ffmpeg and not has_pipe:
        result.add_error(
            "No FFmpeg pipe or video output found. Must pipe frames to "
            "ffmpeg subprocess to produce MP4 output."
        )
    elif not has_output_path:
        result.add_warning(
            "Output path '/workspace/output/simulation.mp4' not found. "
            "Video may be written to unexpected location."
        )


def _check_moment_usage(code: str, result: CodeValidationResult) -> None:
    """Verify moment of inertia uses helper functions."""
    has_body_creation = "pymunk.Body(" in code
    has_moment_helper = (
        "moment_for_circle" in code
        or "moment_for_box" in code
        or "moment_for_poly" in code
        or "shape.mass" in code
        or "shape.density" in code
    )

    # Check for hardcoded moment values (common LLM mistake)
    hardcoded_moment = re.findall(
        r"pymunk\.Body\(\s*[\d.]+\s*,\s*[\d.]+\s*\)", code
    )

    if has_body_creation and not has_moment_helper and hardcoded_moment:
        result.add_warning(
            f"Found {len(hardcoded_moment)} Body() calls with hardcoded moment values. "
            "Use pymunk.moment_for_circle/box() instead for correct physics."
        )


def _check_headless(code: str, result: CodeValidationResult) -> None:
    """Verify headless rendering (no display window)."""
    if "pygame.display.set_mode" in code:
        result.add_warning(
            "pygame.display.set_mode() found — this requires a display server. "
            "Use pygame.Surface((WIDTH, HEIGHT)) for headless rendering."
        )
    if "pygame.display.flip" in code or "pygame.display.update" in code:
        result.add_warning(
            "pygame.display.flip/update found — not needed in headless mode. "
            "Pipe frames directly to FFmpeg."
        )
    if "SDL_VIDEODRIVER" not in code:
        result.add_warning(
            "SDL_VIDEODRIVER not set to 'dummy'. Required for headless rendering."
        )


def _check_stdout_markers(code: str, result: CodeValidationResult) -> None:
    """Verify required stdout markers are printed."""
    if "PAYOFF_TIMESTAMP" not in code:
        result.add_warning("PAYOFF_TIMESTAMP marker not found in stdout prints.")
    if "PEAK_BODY_COUNT" not in code:
        result.add_warning("PEAK_BODY_COUNT marker not found in stdout prints.")
