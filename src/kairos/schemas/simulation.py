"""Kairos Agent — Simulation Response Models.

Pydantic models used as structured output targets for LLM calls
during the simulation generation and adjustment phases.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulationCode(BaseModel, frozen=True):
    """Structured output from the simulation-first-pass LLM call.

    The LLM returns a complete Python script as a string.
    """

    code: str = Field(
        description=(
            "Complete, self-contained Blender simulation config "
            "that drives rigid body physics rendering to an output MP4"
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of design choices and physics setup",
    )


class AdjustedSimulationCode(BaseModel, frozen=True):
    """Structured output from the parameter adjustment LLM call.

    The LLM returns fixed code plus an explanation of changes.
    """

    code: str = Field(
        description="The corrected Python simulation script",
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of changes applied to fix the validation failures",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these changes should fix the issues",
    )


# ---------------------------------------------------------------------------
# JSON-config architecture models (template-based pipeline)
# ---------------------------------------------------------------------------


class SimulationConfigOutput(BaseModel, frozen=True):
    """Structured output from the LLM config-generation call.

    The LLM returns a JSON configuration matching the category schema.
    A fixed template consumes this config to produce the simulation.
    """

    config: dict[str, Any] = Field(
        description=(
            "JSON configuration object matching the category-specific schema. "
            "Controls visual layout, object counts, colours, and creative "
            "choices. Physics parameters are locked in the template."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of creative choices (path type, palette rationale, timing)",
    )


class AdjustedSimulationConfig(BaseModel, frozen=True):
    """Structured output from the config-adjustment LLM call.

    The LLM fixes config parameters based on validation or physics failures.
    """

    config: dict[str, Any] = Field(
        description="The corrected configuration object",
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of parameter changes applied",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these changes should fix the issues",
    )
