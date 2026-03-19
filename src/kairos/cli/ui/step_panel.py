"""Kairos Agent -- Step Detail Panels.

Rich Panel widgets that render completed-step summaries inside the
live display.  Each panel shows step name, status, duration, LLM call
count, cost, and (optionally) a decisions table.

These are pure renderables -- no I/O.
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kairos.cli.ui.formatters import (
    LEVEL_STYLES,
    STEP_STYLE,
    format_cost,
    format_duration,
    format_tokens,
)


def _status_style(status: str) -> str:
    """Map step status to a Rich border colour."""
    if status == "success":
        return "green"
    if status in ("error", "failed"):
        return "red"
    return "yellow"


def build_step_panel(
    step_event: dict[str, Any],
    llm_calls: list[dict[str, Any]] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> Panel:
    """Build a Rich Panel summarising a completed step.

    Parameters
    ----------
    step_event:
        A ``step_completed`` event dict.
    llm_calls:
        Optional list of ``llm_call_completed`` events for this step.
    decisions:
        Optional list of ``decision`` events for this step.
    """
    name = step_event.get("step_name", "?")
    status = step_event.get("status", "?")
    duration = format_duration(step_event.get("duration_ms", 0))
    attempt = step_event.get("attempt", 1)

    # -- Header text -------------------------------------------------------
    header = Text()
    header.append(f"Step {step_event.get('step_number', '?')}: ", style="bold")
    header.append(name, style=STEP_STYLE)

    # -- Status line -------------------------------------------------------
    body = Text()
    status_style = LEVEL_STYLES.get("success" if status == "success" else "error")
    body.append("Status: ")
    body.append(status.upper(), style=status_style)
    body.append(f"  Duration: {duration}")
    if attempt > 1:
        body.append(f"  Attempt: {attempt}")
    body.append("\n")

    # -- Errors (if any) ---------------------------------------------------
    errors = step_event.get("errors", [])
    if errors:
        body.append("\nErrors:\n", style="bold red")
        for err in errors:
            body.append(f"  - {err}\n", style="red")

    # -- Build panel -------------------------------------------------------
    content_parts: list[Any] = [body]

    # LLM calls table
    if llm_calls:
        content_parts.append(_llm_calls_table(llm_calls))

    # Decisions table
    if decisions:
        content_parts.append(_decisions_table(decisions))

    from rich.console import Group

    return Panel(
        Group(*content_parts),
        title=header,
        border_style=_status_style(status),
        expand=True,
    )


def _llm_calls_table(calls: list[dict[str, Any]]) -> Table:
    """Render LLM calls as a compact table."""
    table = Table(
        title="LLM Calls",
        show_header=True,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Model", style="bright_blue", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")

    for call in calls:
        model = call.get("model_resolved", call.get("model_alias", "?"))
        status = call.get("status", "?")
        tok = format_tokens(call.get("tokens_in", 0), call.get("tokens_out", 0))
        latency = format_duration(call.get("latency_ms", 0))
        cost = format_cost(call.get("cost_usd", 0.0))

        status_text = Text(status, style="green" if status == "success" else "red")
        table.add_row(model, status_text, tok, latency, cost)

    return table


def _decisions_table(decisions: list[dict[str, Any]]) -> Table:
    """Render decision events as a compact table."""
    table = Table(
        title="Decisions",
        show_header=True,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Saw", ratio=3)
    table.add_column("Decided", ratio=3)
    table.add_column("Action", ratio=3)

    for d in decisions:
        saw = _truncate(d.get("saw", ""), 80)
        decided = _truncate(d.get("decided", ""), 80)
        action = _truncate(d.get("action", ""), 80)
        table.add_row(saw, decided, action)

    return table


def _truncate(s: str, max_len: int) -> str:
    """Truncate long strings with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
