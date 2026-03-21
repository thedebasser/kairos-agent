"""Unit tests for async subprocess helpers.

Phase 4: tests for the shared async subprocess module.
"""

import asyncio

import pytest

from kairos.services.async_subprocess import run_async, run_ffprobe_json

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class TestRunAsync:
    """Tests for the general-purpose run_async helper."""

    async def test_simple_echo(self):
        """Basic command should return stdout."""
        rc, stdout, stderr = await run_async(
            ["python", "-c", "print('hello')"], timeout=10, text=True,
        )
        assert rc == 0
        assert "hello" in stdout

    async def test_nonexistent_command(self):
        """Missing executable should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await run_async(
                ["__nonexistent_command_xyz__"], timeout=5, text=True,
            )

    async def test_timeout(self):
        """Long-running command should be killed after timeout."""
        with pytest.raises(asyncio.TimeoutError):
            await run_async(
                ["python", "-c", "import time; time.sleep(30)"],
                timeout=1,
                text=True,
            )

    async def test_check_nonzero_raises(self):
        """check=True with a failing command should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="failed \\(rc=1\\)"):
            await run_async(
                ["python", "-c", "raise SystemExit(1)"],
                timeout=5,
                text=True,
                check=True,
            )

    async def test_binary_mode(self):
        """text=False should return bytes."""
        rc, stdout, _ = await run_async(
            ["python", "-c", "import sys; sys.stdout.buffer.write(b'\\x00\\x01\\x02')"],
            timeout=5,
            text=False,
        )
        assert rc == 0
        assert isinstance(stdout, bytes)
        assert b"\x00\x01\x02" in stdout

    async def test_stderr_capture(self):
        """Should capture stderr output."""
        rc, _, stderr = await run_async(
            ["python", "-c", "import sys; sys.stderr.write('err')"],
            timeout=5,
            text=True,
        )
        assert rc == 0
        assert "err" in stderr
