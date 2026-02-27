"""Kairos Agent — Session Logging.

Configures structured logging for every pipeline run:
  1. Rich console output (coloured, timestamped, shows module + level)
  2. Per-run log file in ``logs/<run_id>.log`` (full DEBUG trace)

Usage (called once at CLI entry):
    from kairos.services.session_logging import init_logging
    run_id = init_logging()            # auto-generates run ID
    run_id = init_logging("my-run-id") # explicit run ID
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"

CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%H:%M:%S"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Colour formatter for console
# ---------------------------------------------------------------------------

class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour codes to console log output."""

    COLOURS = {
        logging.DEBUG: "\033[90m",     # grey
        logging.INFO: "\033[36m",      # cyan
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, "")
        formatted = super().format(record)
        return f"{colour}{formatted}{self.RESET}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_initialised = False


def init_logging(
    run_id: str | None = None,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> str:
    """Initialise console + file logging for a pipeline run.

    Args:
        run_id: Optional explicit run ID (used in log filename).
                Defaults to a new UUID.
        console_level: Minimum level for console output.
        file_level: Minimum level for the log file.

    Returns:
        The run ID used for the log file.
    """
    global _initialised  # noqa: PLW0603
    if _initialised:
        # Re-entry guard — only configure once per process
        return run_id or "already-initialised"

    if run_id is None:
        run_id = uuid4().hex[:12]

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{run_id}.log"
    log_path = LOG_DIR / log_filename

    # ── Root logger ──────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Remove any pre-existing handlers (e.g. from basicConfig in tests)
    root.handlers.clear()

    # ── Console handler ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(_ColourFormatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (flush every record so file always matches console) ──
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    )
    # Force immediate flush so a Ctrl+C never loses buffered entries
    _orig_emit = file_handler.emit

    def _flushing_emit(record: logging.LogRecord) -> None:
        _orig_emit(record)
        file_handler.flush()

    file_handler.emit = _flushing_emit  # type: ignore[method-assign]
    root.addHandler(file_handler)

    # ── Quieten noisy libraries ──────────────────────────────────────
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "openai",
        "litellm",
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
        "instructor",
        "langsmith",
        "docker",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # LiteLLM has its own verbose flag that bypasses Python logging
    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except ImportError:
        pass

    _initialised = True

    logger = logging.getLogger(__name__)
    logger.info("=" * 72)
    logger.info("  Kairos Agent — Session started")
    logger.info("  Run ID:   %s", run_id)
    logger.info("  Log file: %s", log_path)
    logger.info("=" * 72)

    return run_id
