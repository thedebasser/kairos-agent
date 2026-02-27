"""Kairos Agent — Simulation Response Models.

Pydantic models used as structured output targets for LLM calls
during the simulation generation and adjustment phases.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimulationCode(BaseModel, frozen=True):
    """Structured output from the simulation-first-pass LLM call.

    The LLM returns a complete Python script as a string.
    """

    code: str = Field(
        description=(
            "Complete, self-contained Python script using Pygame + Pymunk "
            "that renders a simulation video to /workspace/output/simulation.mp4"
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
