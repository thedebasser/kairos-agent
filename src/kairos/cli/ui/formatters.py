"""Kairos Agent -- Console Output Formatters.

Pure functions that transform trace events into Rich renderables
(Text, Table, Panel) with consistent styling.  No I/O -- all side
effects live in ``run_display.py``.

Style guide:
  - No emojis.
  - Level colours: info=cyan, success=green, warning=yellow, error=red,
    debug=dim.
  - Step names are bold white.
  - Timestamps are dim.
  - Cost figures use 6 decimal places (micro-dollar precision).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.style import Style
from rich.text import Text

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

LEVEL_STYLES: dict[str, Style] = {
    "info": Style(color="cyan"),
    "success": Style(color="green", bold=True),
    "warning": Style(color="yellow"),
    "error": Style(color="red", bold=True),
    "debug": Style(dim=True),
}

STEP_STYLE = Style(color="white", bold=True)
TIMESTAMP_STYLE = Style(dim=True)
COST_STYLE = Style(color="magenta")
TOKENS_STYLE = Style(color="blue")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_timestamp(ts: datetime | str) -> Text:
    """Format a UTC timestamp as ``HH:MM:SS``."""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return Text(str(ts)[:8], style=TIMESTAMP_STYLE)
    return Text(ts.strftime("%H:%M:%S"), style=TIMESTAMP_STYLE)


def format_level(level: str) -> Text:
    """Render a level tag like ``[INFO]``."""
    style = LEVEL_STYLES.get(level, LEVEL_STYLES["info"])
    return Text(f"[{level.upper():^7s}]", style=style)


def format_step_name(name: str) -> Text:
    """Render a step name in bold white."""
    return Text(name, style=STEP_STYLE) if name else Text("")


def format_duration(duration_ms: int) -> str:
    """Human-friendly duration: ``1.23s`` or ``456ms``."""
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{duration_ms}ms"


def format_cost(cost_usd: float) -> Text:
    """Render a cost value with 6dp."""
    return Text(f"${cost_usd:.6f}", style=COST_STYLE)


def format_tokens(tokens_in: int, tokens_out: int) -> Text:
    """Render token counts as ``123 -> 456``."""
    return Text(f"{tokens_in} -> {tokens_out}", style=TOKENS_STYLE)


# ---------------------------------------------------------------------------
# Event -> Rich Text line
# ---------------------------------------------------------------------------

def format_console_event(event: dict[str, Any]) -> Text:
    """Turn a ``console.jsonl`` dict into a single Rich Text line."""
    ts = format_timestamp(event.get("timestamp", ""))
    level = format_level(event.get("level", "info"))
    step = format_step_name(event.get("step_name", ""))
    msg = event.get("message", "")

    line = Text()
    line.append_text(ts)
    line.append(" ")
    line.append_text(level)
    line.append(" ")
    if step.plain:
        line.append_text(step)
        line.append(" | ")
    line.append(msg)
    return line


def format_step_completed(event: dict[str, Any]) -> Text:
    """One-line summary for a ``step_completed`` event."""
    name = event.get("step_name", "?")
    status = event.get("status", "?")
    duration = format_duration(event.get("duration_ms", 0))

    style = LEVEL_STYLES.get("success" if status == "success" else "error")
    line = Text()
    line.append_text(format_timestamp(event.get("timestamp", "")))
    line.append(" ")
    line.append(f"Step {name} ", style=STEP_STYLE)
    line.append(status.upper(), style=style)
    line.append(f" ({duration})")
    return line


def format_run_completed(event: dict[str, Any]) -> Text:
    """One-line summary for a ``run_completed`` event."""
    status = event.get("status", "?")
    duration = format_duration(event.get("total_duration_ms", 0))
    cost = event.get("total_cost_usd", 0.0)
    calls = event.get("total_llm_calls", 0)

    style = LEVEL_STYLES.get("success" if status == "success" else "error")
    line = Text()
    line.append("Run ", style=Style(bold=True))
    line.append(status.upper(), style=style)
    line.append(f" | {duration} | {calls} LLM calls | ")
    line.append_text(format_cost(cost))
    return line


def format_llm_call_completed(event: dict[str, Any]) -> Text:
    """One-line summary for an ``llm_call_completed`` event."""
    model = event.get("model_resolved", event.get("model_alias", "?"))
    tokens_in = event.get("tokens_in", 0)
    tokens_out = event.get("tokens_out", 0)
    latency = format_duration(event.get("latency_ms", 0))
    cost = event.get("cost_usd", 0.0)
    status = event.get("status", "success")

    line = Text()
    line.append_text(format_timestamp(event.get("timestamp", "")))
    line.append(" LLM ")
    line.append(model, style=Style(color="bright_blue"))
    line.append(f" {status} ")
    line.append_text(format_tokens(tokens_in, tokens_out))
    line.append(f" {latency} ")
    line.append_text(format_cost(cost))
    return line
