"""Kairos Agent — Async Subprocess Utilities.

Provides async wrappers for FFprobe and FFmpeg subprocess execution,
replacing blocking ``subprocess.run`` calls with
``asyncio.create_subprocess_exec``.

All media analysis functions (validation, video review, audio review,
screenshot analysis) should use these helpers instead of ``subprocess.run``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_async(
    cmd: list[str],
    *,
    timeout: int = 60,
    text: bool = True,
    check: bool = False,
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously.

    Args:
        cmd: Command and arguments.
        timeout: Timeout in seconds.
        text: If True, decode stdout/stderr as UTF-8.
        check: If True, raise on non-zero exit code.

    Returns:
        Tuple of (returncode, stdout, stderr).

    Raises:
        asyncio.TimeoutError: If the process exceeds the timeout.
        RuntimeError: If check=True and the process returns non-zero.
        FileNotFoundError: If the executable is not found.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise

    returncode = process.returncode or 0
    if text:
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
    else:
        stdout = stdout_bytes  # type: ignore[assignment]
        stderr = stderr_bytes  # type: ignore[assignment]

    if check and returncode != 0:
        raise RuntimeError(
            f"Command {cmd[0]} failed (rc={returncode}): {stderr[:500] if isinstance(stderr, str) else ''}"
        )

    return returncode, stdout, stderr


async def run_ffprobe_json(
    ffprobe_path: str,
    video_path: str,
    *,
    timeout: int = 30,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output.

    Args:
        ffprobe_path: Path to the ffprobe executable.
        video_path: Path to the media file.
        timeout: Timeout in seconds.
        extra_args: Additional ffprobe arguments.

    Returns:
        Parsed JSON dict, or empty dict on failure.
    """
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        *(extra_args or ["-show_format", "-show_streams"]),
        video_path,
    ]

    try:
        rc, stdout, _ = await run_async(cmd, timeout=timeout)
        if rc != 0:
            return {}
        return json.loads(stdout)
    except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError):
        return {}
