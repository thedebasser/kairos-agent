"""Blender executor — runs Blender scripts in background mode.

Handles finding the Blender binary, executing scripts with arguments,
capturing JSON output, and managing temp files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Blender binary search paths (platform-specific)
_BLENDER_SEARCH_PATHS = {
    "Windows": [
        Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
    ],
    "Darwin": [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    ],
    "Linux": [
        Path("/usr/bin/blender"),
        Path("/snap/bin/blender"),
    ],
}

# Project root is 5 levels up: marble/ -> pipelines/ -> kairos/ -> src/ -> project_root/
BLEND_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "blend" / "scripts"


def find_blender() -> Path | None:
    """Find the Blender executable on the system."""
    # Check PATH first
    blender_path = shutil.which("blender")
    if blender_path:
        return Path(blender_path)

    # Platform-specific search
    system = platform.system()
    for p in _BLENDER_SEARCH_PATHS.get(system, []):
        if p.exists():
            return p

    return None


async def run_blender_script(
    script_name: str,
    *,
    blend_file: str | None = None,
    script_args: list[str] | None = None,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    """Run a Blender Python script in background mode.

    Args:
        script_name: Name of the script in blend/scripts/ (e.g. 'generate_course.py').
        blend_file: Optional .blend file to open before running script.
        script_args: Arguments to pass after '--' separator.
        timeout_sec: Maximum execution time.

    Returns:
        Dict with 'returncode', 'stdout', 'stderr', and optionally parsed 'json_output'.
    """
    blender = find_blender()
    if blender is None:
        raise FileNotFoundError(
            "Blender not found. Install Blender and ensure it's on PATH or "
            "in a standard location."
        )

    script_path = BLEND_SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Blender script not found: {script_path}")

    cmd = [str(blender), "--background"]
    if blend_file:
        cmd.append(str(Path(blend_file).resolve()))
    cmd.extend(["--python", str(script_path)])

    if script_args:
        cmd.append("--")
        cmd.extend(script_args)

    logger.info("Running Blender: %s", " ".join(cmd[:6]) + " ...")
    logger.debug("Full command: %s", cmd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        proc.kill()
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Blender script timed out after {timeout_sec}s",
            "json_output": None,
        }

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Try to extract JSON from stdout (last JSON block)
    json_output = _extract_json(stdout)

    if proc.returncode != 0:
        logger.warning("Blender script %s exited with code %d", script_name, proc.returncode)
        logger.debug("Blender stderr:\n%s", stderr[-2000:] if len(stderr) > 2000 else stderr)

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "json_output": json_output,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the last valid JSON object/block from text."""
    # Find the last '{' and try to parse from there
    lines = text.strip().split("\n")

    # Try parsing the last N lines as JSON
    json_lines: list[str] = []
    in_json = False
    brace_depth = 0

    for line in reversed(lines):
        stripped = line.strip()
        if not in_json:
            if stripped.endswith("}"):
                in_json = True
                brace_depth = stripped.count("}") - stripped.count("{")
                json_lines.insert(0, line)
                if brace_depth <= 0:
                    break
            continue
        json_lines.insert(0, line)
        brace_depth += stripped.count("}") - stripped.count("{")
        if brace_depth <= 0:
            break

    if json_lines:
        try:
            return json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            pass

    return None
