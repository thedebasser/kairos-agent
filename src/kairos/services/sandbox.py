"""Kairos Agent — Simulation Sandbox Executor.

Executes agent-generated simulation code inside an isolated Docker container
with strict resource limits (no network, 4GB RAM, 2 CPUs, 5min timeout).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from kairos.config import get_settings
from kairos.exceptions import (
    SimulationExecutionError,
    SimulationOOMError,
    SimulationTimeoutError,
)
from kairos.models.contracts import SimulationResult

logger = logging.getLogger(__name__)


def execute_simulation(
    code: str,
    *,
    timeout: int | None = None,
    memory_limit: str | None = None,
    cpu_limit: int | None = None,
) -> SimulationResult:
    """Execute simulation code in Docker sandbox.

    Args:
        code: Complete Python source code for the simulation.
        timeout: Max execution time in seconds (default from settings).
        memory_limit: Docker memory limit (default from settings).
        cpu_limit: Docker CPU core limit (default from settings).

    Returns:
        SimulationResult with return code, stdout, stderr, and output files.

    Raises:
        SimulationTimeoutError: Execution exceeded timeout.
        SimulationOOMError: Container killed by OOM.
        SimulationExecutionError: Any other sandbox failure.
    """
    settings = get_settings()
    timeout = timeout or settings.sandbox_timeout_sec
    memory_limit = memory_limit or settings.sandbox_memory_limit
    cpu_limit = cpu_limit or settings.sandbox_cpu_limit

    # Write code to a temp workspace directory
    workspace = Path(tempfile.mkdtemp(prefix="kairos_sim_"))
    code_file = workspace / "simulation.py"
    code_file.write_text(code, encoding="utf-8")
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        f"--memory={memory_limit}",
        f"--cpus={cpu_limit}",
        f"--stop-timeout={timeout}",
        "--read-only",
        "--tmpfs",
        "/tmp:size=512m",
        "-v",
        f"{workspace}:/workspace",
        settings.sandbox_image,
        "python",
        "/workspace/simulation.py",
    ]

    logger.info("Executing simulation in sandbox (timeout=%ds)", timeout)

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Grace period beyond Docker timeout
        )
    except subprocess.TimeoutExpired as e:
        msg = f"Simulation exceeded {timeout}s timeout"
        raise SimulationTimeoutError(msg) from e
    except FileNotFoundError as e:
        msg = "Docker not found. Is Docker installed and running?"
        raise SimulationExecutionError(msg) from e

    # Check for OOM kill (Docker exit code 137)
    if result.returncode == 137:
        msg = f"Simulation killed by OOM (memory limit: {memory_limit})"
        raise SimulationOOMError(msg)

    output_files = [str(f) for f in output_dir.iterdir()] if output_dir.exists() else []

    return SimulationResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_files=output_files,
    )
