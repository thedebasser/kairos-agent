"""Kairos Agent -- RunTracer: single source of truth for observability.

Replaces the triple-logging of ``session_logging`` + ``step_artifacts``
+ ``monitoring`` with one unified API.  Every pipeline event flows
through ``RunTracer`` to registered ``TracingSink`` implementations
and the per-run JSONL file writers.

Usage in graph.py::

    tracer = get_tracer()              # module-level singleton
    tracer.init_run(run_id, pipeline)  # call once at pipeline start

    with tracer.step("idea_agent", step_number=1) as span:
        span.log("Generating concept...")
        with span.llm_call("concept_developer", resolved_model) as llm:
            result = await call_llm(...)
            llm.complete(tokens_in=..., tokens_out=..., cost_usd=...)
        span.decide(saw="...", decided="...", action="...")

    tracer.complete_run(status, final_state)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from kairos.ai.tracing.events import (
    ActionTaken,
    ConsoleMessage,
    Decision,
    LLMCallCompleted,
    LLMCallStarted,
    PromptRendered,
    RunCompleted,
    RunStarted,
    StepCompleted,
    StepStarted,
    TraceEvent,
)
from kairos.ai.tracing.writers import RunFileWriter

logger = logging.getLogger(__name__)

# Root directory for run artifacts (project_root/runs/)
_RUNS_DIR = Path(__file__).resolve().parents[4] / "runs"


# ---------------------------------------------------------------------------
# LLMCallSpan
# ---------------------------------------------------------------------------

class LLMCallSpan:
    """Context-managed span for a single LLM call."""

    def __init__(
        self,
        tracer: RunTracer,
        step_name: str,
        call_id: str,
        model_alias: str,
        model_resolved: str,
        call_pattern: str = "direct",
    ) -> None:
        self._tracer = tracer
        self._step_name = step_name
        self.call_id = call_id
        self.model_alias = model_alias
        self.model_resolved = model_resolved
        self.call_pattern = call_pattern
        self._start = time.monotonic()

        # Published data (set by caller via complete/fail)
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.cost_usd: float = 0.0
        self.status: str = "success"
        self.error: str | None = None
        self.model_type: str = "local"
        self.provider: str = ""
        self.routing_outcome: str = "direct"
        self.has_thinking: bool = False

    def complete(
        self,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        model_type: str = "local",
        provider: str = "",
        routing_outcome: str = "direct",
        has_thinking: bool = False,
    ) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd
        self.model_type = model_type
        self.provider = provider
        self.routing_outcome = routing_outcome
        self.has_thinking = has_thinking
        self.status = "success"

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error

    @property
    def latency_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)


# ---------------------------------------------------------------------------
# StepSpan
# ---------------------------------------------------------------------------

class StepSpan:
    """Context-managed span for a pipeline step (graph node)."""

    def __init__(
        self,
        tracer: RunTracer,
        step_name: str,
        step_number: int,
        attempt: int = 1,
    ) -> None:
        self._tracer = tracer
        self.step_name = step_name
        self.step_number = step_number
        self.attempt = attempt
        self._start = time.monotonic()
        self.status: str = "success"
        self.outputs: dict[str, Any] = {}
        self.errors: list[str] = []
        self._llm_call_counter = 0

    # -- Logging -----------------------------------------------------------

    def log(
        self,
        message: str,
        level: str = "info",
    ) -> None:
        """Emit a human-readable console message."""
        self._tracer._emit(
            ConsoleMessage(
                run_id=self._tracer._run_id,
                level=level,  # type: ignore[arg-type]
                step_name=self.step_name,
                message=message,
            ),
            console=True,
        )

    # -- Decisions ---------------------------------------------------------

    def decide(
        self,
        *,
        saw: str = "",
        decided: str = "",
        action: str = "",
        reasoning: str = "",
    ) -> None:
        """Record an agent reasoning entry."""
        event = Decision(
            run_id=self._tracer._run_id,
            step_name=self.step_name,
            saw=saw,
            decided=decided,
            action=action,
            reasoning=reasoning,
        )
        self._tracer._emit(event)
        if self._tracer._writer:
            self._tracer._writer.write_decision(
                self.step_name, self.step_number, event
            )

    # -- LLM calls ---------------------------------------------------------

    @contextmanager
    def llm_call(
        self,
        model_alias: str,
        model_resolved: str,
        *,
        call_pattern: str = "direct",
    ) -> Generator[LLMCallSpan, None, None]:
        """Context manager for an LLM call within this step."""
        from uuid import uuid4

        call_id = uuid4().hex[:16]
        self._llm_call_counter += 1

        start_event = LLMCallStarted(
            run_id=self._tracer._run_id,
            call_id=call_id,
            step_name=self.step_name,
            model_alias=model_alias,
            model_resolved=model_resolved,
            call_pattern=call_pattern,
        )
        self._tracer._emit(start_event)

        span = LLMCallSpan(
            tracer=self._tracer,
            step_name=self.step_name,
            call_id=call_id,
            model_alias=model_alias,
            model_resolved=model_resolved,
            call_pattern=call_pattern,
        )

        try:
            yield span
        except Exception as exc:
            span.fail(str(exc))
            raise
        finally:
            completed_event = LLMCallCompleted(
                run_id=self._tracer._run_id,
                call_id=call_id,
                step_name=self.step_name,
                model_alias=model_alias,
                model_resolved=span.model_resolved,
                model_type=span.model_type,
                provider=span.provider,
                call_pattern=span.call_pattern,
                routing_outcome=span.routing_outcome,
                tokens_in=span.tokens_in,
                tokens_out=span.tokens_out,
                cost_usd=span.cost_usd,
                latency_ms=span.latency_ms,
                status=span.status,
                error=span.error,
                has_thinking=span.has_thinking,
            )
            self._tracer._emit(completed_event)
            self._tracer._total_cost += span.cost_usd
            self._tracer._total_llm_calls += 1

    # -- Prompt lineage ----------------------------------------------------

    def record_prompt(
        self,
        call_id: str,
        template_name: str,
        *,
        template_version: int = 0,
        template_hash: str = "",
        pipeline: str = "",
        variables: dict[str, str] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a prompt render event and write the request file."""
        event = PromptRendered(
            run_id=self._tracer._run_id,
            call_id=call_id,
            step_name=self.step_name,
            template_name=template_name,
            template_version=template_version,
            template_hash=template_hash,
            pipeline=pipeline,
            variables=variables or {},
        )
        self._tracer._emit(event)

        if self._tracer._writer and messages:
            self._tracer._writer.write_prompt_request(
                self.step_name,
                self.step_number,
                messages,
                lineage=event.model_dump(mode="json", exclude={"event_id", "run_id", "timestamp", "event_type"}),
            )

    def record_response(
        self,
        call_id: str,
        response: Any,
    ) -> None:
        """Write the LLM response artifact file."""
        if self._tracer._writer:
            self._tracer._writer.write_prompt_response(
                self.step_name,
                self.step_number,
                response,
                call_id=call_id,
            )

    # -- Completion --------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    def fail(self, error: str) -> None:
        self.status = "error"
        self.errors.append(error)

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


# ---------------------------------------------------------------------------
# RunTracer
# ---------------------------------------------------------------------------

class RunTracer:
    """Single source of truth for all pipeline observability.

    Manages the run lifecycle, dispatches events to sinks, and writes
    the per-run file structure.
    """

    def __init__(self, runs_dir: Path | None = None) -> None:
        self._runs_dir = runs_dir or _RUNS_DIR
        self._run_id: str = ""
        self._pipeline: str = ""
        self._writer: RunFileWriter | None = None
        self._sinks: list[Any] = []  # TracingSink instances
        self._start: float = 0.0
        self._total_cost: float = 0.0
        self._total_llm_calls: int = 0
        self._initialised: bool = False
        self._current_step_name: str = ""   # active step, used by tracer.action()

    # -- Sink registration -------------------------------------------------

    def add_sink(self, sink: Any) -> None:
        """Register a TracingSink consumer."""
        self._sinks.append(sink)

    # -- Run lifecycle -----------------------------------------------------

    def init_run(self, run_id: str, pipeline: str) -> None:
        """Initialise tracing for a new pipeline run."""
        self._run_id = run_id
        self._pipeline = pipeline
        self._start = time.monotonic()
        self._total_cost = 0.0
        self._total_llm_calls = 0
        self._initialised = True

        # Create file writer
        self._writer = RunFileWriter(self._runs_dir, run_id)

        # Emit run_started
        self._emit(
            RunStarted(
                run_id=run_id,
                pipeline=pipeline,
                pipeline_run_id=run_id,
            ),
            console=True,
        )

        logger.info("RunTracer initialised: run=%s pipeline=%s dir=%s", run_id, pipeline, self._writer.run_dir)

    def complete_run(
        self,
        status: str,
        *,
        errors: list[str] | None = None,
        final_video_path: str | None = None,
        concept_title: str | None = None,
    ) -> None:
        """Finalise the run, write summary, flush sinks."""
        duration_ms = int((time.monotonic() - self._start) * 1000) if self._start else 0

        event = RunCompleted(
            run_id=self._run_id,
            pipeline=self._pipeline,
            pipeline_run_id=self._run_id,
            status=status,
            total_duration_ms=duration_ms,
            total_cost_usd=self._total_cost,
            total_llm_calls=self._total_llm_calls,
            errors=errors or [],
            final_video_path=final_video_path,
            concept_title=concept_title,
        )
        self._emit(event, console=True)

        # Write run_summary.json
        from kairos.ai.tracing.summary import write_run_summary

        if self._writer:
            write_run_summary(self._writer.run_dir, event)

        # Flush and close
        for sink in self._sinks:
            try:
                sink.flush()
                sink.close()
            except Exception as exc:
                logger.warning("Sink cleanup failed: %s", exc)

        if self._writer:
            self._writer.close()

        self._initialised = False

    # -- Step context manager ----------------------------------------------

    @contextmanager
    def step(
        self,
        step_name: str,
        step_number: int,
        *,
        attempt: int = 1,
    ) -> Generator[StepSpan, None, None]:
        """Context manager wrapping a pipeline step."""
        start_event = StepStarted(
            run_id=self._run_id,
            step_name=step_name,
            step_number=step_number,
            attempt=attempt,
        )
        self._emit(start_event, console=True)

        span = StepSpan(self, step_name, step_number, attempt)
        self._current_step_name = step_name

        try:
            yield span
        except Exception as exc:
            span.fail(str(exc))
            raise
        finally:
            self._current_step_name = ""
            completed_event = StepCompleted(
                run_id=self._run_id,
                step_name=step_name,
                step_number=step_number,
                attempt=span.attempt,
                status=span.status,
                duration_ms=span.duration_ms,
                outputs=span.outputs,
                errors=span.errors,
            )
            self._emit(completed_event, console=True)

    # -- Tool / sub-operation logging ------------------------------------

    def action(
        self,
        tool: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        status: str = "success",
        duration_ms: int = 0,
        step_name: str = "",
    ) -> None:
        """Emit an ActionTaken event for a discrete sub-operation.

        Can be called from anywhere that has access to the tracer singleton.
        Uses the active step name if ``step_name`` is not provided.

        Example::

            tracer.action("blender:generate_course",
                          input_summary="archetype=s_curve",
                          output_summary="domino_course.blend (1.2 MB)",
                          status="success", duration_ms=8340)
        """
        effective_step = step_name or self._current_step_name
        self._emit(
            ActionTaken(
                run_id=self._run_id,
                step_name=effective_step,
                tool=tool,
                input_summary=input_summary,
                output_summary=output_summary,
                status=status,  # type: ignore[arg-type]
                duration_ms=duration_ms,
            ),
        )

    # -- Console logging ---------------------------------------------------

    def console(
        self,
        message: str,
        *,
        level: str = "info",
        step_name: str = "",
    ) -> None:
        """Emit a human-readable console message."""
        self._emit(
            ConsoleMessage(
                run_id=self._run_id,
                level=level,  # type: ignore[arg-type]
                step_name=step_name,
                message=message,
            ),
            console=True,
        )

    # -- Output directory --------------------------------------------------

    def get_output_dir(self, version: int = 1) -> Path:
        """Return the versioned assets output directory."""
        if self._writer:
            return self._writer.get_output_dir(version)
        return self._runs_dir / self._run_id / "assets" / f"v{version}"

    @property
    def run_dir(self) -> Path | None:
        return self._writer.run_dir if self._writer else None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_llm_calls(self) -> int:
        return self._total_llm_calls

    # -- Internal ----------------------------------------------------------

    def _emit(self, event: TraceEvent, *, console: bool = False) -> None:
        """Dispatch an event to file writers and sinks."""
        # File writers
        if self._writer:
            self._writer.write_event(event)
            if console:
                self._writer.write_console(event)

        # Sinks
        for sink in self._sinks:
            try:
                sink.on_event(event)
            except Exception as exc:
                logger.warning("Sink %s failed on %s: %s", type(sink).__name__, event.event_type, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracer: RunTracer | None = None


def init_tracer(runs_dir: Path | None = None) -> RunTracer:
    """Create (or replace) the global RunTracer singleton."""
    global _tracer  # noqa: PLW0603
    _tracer = RunTracer(runs_dir=runs_dir)
    return _tracer


def get_tracer() -> RunTracer:
    """Return the global RunTracer, creating one if needed."""
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        _tracer = RunTracer()
    return _tracer
