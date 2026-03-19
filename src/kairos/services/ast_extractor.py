"""Kairos Agent — AST-Based Parameter Extraction.

Walks the Python AST of generated simulation code to extract structural
information that regex cannot reliably capture (AI Architecture Review §7).

Extracted data is used for:
1. More precise adjustment feedback ("your loop runs for 55s, need 65s")
2. Enriching ValidationFeedback with quantitative context
3. Populating category knowledge with parameter ranges

All public functions are safe to call on any string — parse failures
return empty/default results rather than raising.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExtractedParameters:
    """Simulation parameters extracted from the AST."""

    def __init__(self) -> None:
        self.body_count: int = 0
        self.loop_iterations: int = 0
        self.step_size: float = 0.0
        self.estimated_duration_sec: float = 0.0
        self.gravity: tuple[float, float] | None = None
        self.output_path: str = ""
        self.has_ffmpeg_pipe: bool = False
        self.has_pygame_init: bool = False
        self.has_space_step: bool = False
        self.canvas_size: tuple[int, int] | None = None
        self.raw_assignments: dict[str, Any] = {}

    def to_feedback_text(self) -> str:
        """Render as a concise feedback section for adjustment prompts.

        Returns empty string when no meaningful data was extracted
        (e.g. empty or unparseable code).
        """
        # If nothing was found at all, don't emit noise
        has_any_data = (
            self.body_count
            or self.estimated_duration_sec
            or self.gravity
            or self.canvas_size
            or self.has_ffmpeg_pipe
            or self.has_pygame_init
            or self.has_space_step
        )
        if not has_any_data:
            return ""

        lines = ["### Code Structure Analysis (AST)"]
        if self.body_count:
            lines.append(f"- Body/object creation calls: {self.body_count}")
        if self.estimated_duration_sec:
            lines.append(f"- Estimated simulation duration: {self.estimated_duration_sec:.1f}s")
        if self.loop_iterations and self.step_size:
            lines.append(
                f"- Main loop: {self.loop_iterations} iterations × "
                f"{self.step_size:.4f}s step = "
                f"{self.loop_iterations * self.step_size:.1f}s"
            )
        if self.gravity:
            lines.append(f"- Gravity: ({self.gravity[0]}, {self.gravity[1]})")
        if self.canvas_size:
            lines.append(f"- Canvas: {self.canvas_size[0]}×{self.canvas_size[1]}")
        if self.has_ffmpeg_pipe:
            lines.append("- FFmpeg pipe: present ✓")
        else:
            lines.append("- FFmpeg pipe: NOT FOUND ✗")
        if not self.has_pygame_init:
            lines.append("- pygame.init(): NOT FOUND ✗")
        if not self.has_space_step:
            lines.append("- space.step(): NOT FOUND ✗")
        if len(lines) <= 1:
            return ""
        return "\n".join(lines)


def extract_parameters(code: str) -> ExtractedParameters:
    """Extract simulation parameters from code using AST walking.

    Falls back gracefully on parse errors — returns an empty
    ExtractedParameters rather than raising.
    """
    params = ExtractedParameters()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("AST parse failed — returning empty parameters")
        return params

    visitor = _ParameterVisitor(params)
    visitor.visit(tree)

    # Estimate duration from loop range and step size
    if params.loop_iterations and params.step_size:
        params.estimated_duration_sec = params.loop_iterations * params.step_size
    elif params.loop_iterations:
        # Default Pymunk step is 1/60
        params.estimated_duration_sec = params.loop_iterations / 60.0

    return params


class _ParameterVisitor(ast.NodeVisitor):
    """AST visitor that extracts simulation-relevant parameters."""

    def __init__(self, params: ExtractedParameters) -> None:
        self.params = params

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func_name = self._call_name(node)

        # Count pymunk.Body() calls
        if func_name in ("pymunk.Body", "Body"):
            self.params.body_count += 1

        # Detect space.step() calls and extract step size
        if func_name in ("space.step", "self.space.step"):
            self.params.has_space_step = True
            if node.args:
                step_val = self._eval_num(node.args[0])
                if step_val is not None:
                    self.params.step_size = step_val

        # Detect pygame.init()
        if func_name in ("pygame.init", "pg.init"):
            self.params.has_pygame_init = True

        # Detect subprocess.Popen (FFmpeg pipe)
        if func_name in ("subprocess.Popen", "Popen"):
            self.params.has_ffmpeg_pipe = True

        # Detect pymunk.moment_for_circle etc. as body count proxy
        if func_name and "moment_for" in func_name:
            self.params.body_count += 1

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        """Extract loop iteration count from `for i in range(N)` patterns."""
        if isinstance(node.iter, ast.Call):
            func_name = self._call_name(node.iter)
            if func_name == "range":
                range_val = self._eval_range(node.iter)
                if range_val is not None and range_val > self.params.loop_iterations:
                    self.params.loop_iterations = range_val
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        """Detect while loops with frame counters."""
        # Look for `while frame < N` or `while frame_count < N`
        if isinstance(node.test, ast.Compare):
            if len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Lt):
                right = self._eval_num(node.test.comparators[0])
                if right is not None and right > self.params.loop_iterations:
                    self.params.loop_iterations = int(right)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        """Extract named assignments (gravity, canvas size, etc.)."""
        for target in node.targets:
            name = self._target_name(target)
            if name is None:
                continue
            val = self._eval_num(node.value)
            name_lower = name.lower()

            # Capture raw numeric assignments
            if val is not None:
                self.params.raw_assignments[name] = val

            # Gravity: space.gravity = (0, 900)
            if "gravity" in name_lower and isinstance(node.value, ast.Tuple):
                grav = self._eval_tuple(node.value)
                if grav and len(grav) == 2:
                    self.params.gravity = (grav[0], grav[1])

            # Canvas / resolution
            if name_lower in ("width", "w", "screen_width"):
                if val is not None:
                    w = int(val)
                    existing_h = self.params.canvas_size[1] if self.params.canvas_size else 0
                    self.params.canvas_size = (w, existing_h)
            if name_lower in ("height", "h", "screen_height"):
                if val is not None:
                    h = int(val)
                    existing_w = self.params.canvas_size[0] if self.params.canvas_size else 0
                    self.params.canvas_size = (existing_w, h)

            # SIMULATION_TIME, TOTAL_FRAMES, NUM_FRAMES etc.
            if name_lower in ("simulation_time", "total_time", "duration"):
                if val is not None:
                    self.params.estimated_duration_sec = val
            if name_lower in ("total_frames", "num_frames", "frame_count", "max_frames"):
                if val is not None:
                    self.params.loop_iterations = max(
                        self.params.loop_iterations, int(val)
                    )

        self.generic_visit(node)

    # --- Helpers ---

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        """Extract a dotted function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts: list[str] = [node.func.attr]
            val = node.func.value
            while isinstance(val, ast.Attribute):
                parts.append(val.attr)
                val = val.value
            if isinstance(val, ast.Name):
                parts.append(val.id)
            return ".".join(reversed(parts))
        return ""

    @staticmethod
    def _eval_num(node: ast.expr) -> float | None:
        """Try to statically evaluate a node as a number."""
        # Constant literal
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        # Unary minus: -900
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _ParameterVisitor._eval_num(node.operand)
            if inner is not None:
                return -inner
        # BinOp: 60 * 65, 1/60
        if isinstance(node, ast.BinOp):
            left = _ParameterVisitor._eval_num(node.left)
            right = _ParameterVisitor._eval_num(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div) and right != 0:
                    return left / right
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
        return None

    @staticmethod
    def _eval_tuple(node: ast.Tuple) -> list[float] | None:
        """Evaluate a tuple of numeric constants."""
        vals: list[float] = []
        for elt in node.elts:
            v = _ParameterVisitor._eval_num(elt)
            if v is None:
                return None
            vals.append(v)
        return vals

    @staticmethod
    def _eval_range(node: ast.Call) -> int | None:
        """Evaluate range(N) or range(start, end) to iteration count."""
        args = node.args
        if len(args) == 1:
            val = _ParameterVisitor._eval_num(args[0])
            return int(val) if val is not None else None
        if len(args) >= 2:
            start = _ParameterVisitor._eval_num(args[0])
            end = _ParameterVisitor._eval_num(args[1])
            if start is not None and end is not None:
                return int(end - start)
        return None

    @staticmethod
    def _target_name(node: ast.expr) -> str | None:
        """Get the name string from an assignment target."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = [node.attr]
            val = node.value
            while isinstance(val, ast.Attribute):
                parts.append(val.attr)
                val = val.value
            if isinstance(val, ast.Name):
                parts.append(val.id)
            return ".".join(reversed(parts))
        return None
