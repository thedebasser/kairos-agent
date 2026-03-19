"""Kairos Agent — Exception Hierarchy.

Fail-fast error handling with typed exceptions. Reading the exception type alone
tells you what went wrong. PipelineError subclasses are caught and handled
(retry, escalate, re-route). InfrastructureError is never caught — it crashes
the process.

All exceptions include the pipeline_run_id for tracing.
"""

from __future__ import annotations

from uuid import UUID


class PipelineError(Exception):
    """Base for all pipeline errors."""

    def __init__(self, message: str, pipeline_run_id: UUID | None = None) -> None:
        self.pipeline_run_id = pipeline_run_id
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.pipeline_run_id:
            return f"[run={self.pipeline_run_id}] {base}"
        return base


class ConceptGenerationError(PipelineError):
    """Idea Agent failed to produce a valid concept."""


class SimulationExecutionError(PipelineError):
    """Simulation code failed to execute in sandbox."""


class SimulationTimeoutError(SimulationExecutionError):
    """Simulation exceeded maximum execution time."""


class SimulationOOMError(SimulationExecutionError):
    """Simulation exceeded memory limit."""


class ValidationError(PipelineError):
    """Produced output failed validation checks."""


class VideoAssemblyError(PipelineError):
    """FFmpeg composition failed."""


class LLMRoutingError(PipelineError):
    """Both local and cloud LLM calls failed."""


class PublishError(PipelineError):
    """Upload to platform failed after all retries."""


class InfrastructureError(Exception):
    """Database, Redis, Docker, or other infrastructure failure. Always fatal.

    Inherits from Exception (NOT PipelineError) so that ``except PipelineError``
    handlers cannot accidentally swallow infrastructure failures.
    (Finding 4.2 — kairos_architectural_review)
    """

    def __init__(self, message: str, pipeline_run_id: UUID | None = None) -> None:
        self.pipeline_run_id = pipeline_run_id
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.pipeline_run_id:
            return f"[run={self.pipeline_run_id}] {base}"
        return base
