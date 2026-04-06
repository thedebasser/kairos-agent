# ADR-004: Pluggable Tracing Sinks

**Status:** Accepted
**Date:** 2026-03-01

## Context

Before the refactor, logging went to three disconnected places: `logs/` text files, `runs/steps/` JSON files, and the database. There was no cross-referencing between them. You couldn't follow a single run's lifecycle across all three. Reviewing a run meant opening 5+ files and mentally joining them by timestamp.

We needed unified observability that:
- Produces human-readable audit trails (for debugging)
- Feeds dashboards and analytics (for monitoring)
- Persists to the database (for querying)
- Doesn't create a single point of failure (if Langfuse is down, the run shouldn't fail)

## Decision

Implement a single `RunTracer` that emits strongly-typed events (`TraceEvent` subclasses). Events flow through a list of `TracingSink` implementations. Each sink handles its own persistence independently.

```
RunTracer
  ├── RunFileWriter    → runs/<id>/events.jsonl + console.jsonl + step artifacts
  ├── LangfuseSink     → Langfuse traces + MetricsStore + AlertManager
  └── DatabaseSink     → PostgreSQL (pipeline_runs, agent_runs)
```

Key design choices:
- **Events are the atomic unit.** 10 event types cover the full lifecycle. Each carries `run_id` + `event_id`.
- **Sinks are fire-and-forget.** If a sink raises, the error is logged but the pipeline continues.
- **Context managers for scope.** `tracer.step("simulation")` and `tracer.llm_call(...)` are context managers that automatically emit start/end events.
- **No emojis in events.** Plain ASCII for machine readability and reliable JSON serialization.

## Consequences

**Positive:**
- Adding a new sink (e.g., CloudWatch, Datadog) = implementing 3 methods (`on_event`, `flush`, `close`).
- Run artifacts are self-contained — `runs/<id>/` has everything needed to replay a run.
- Events can be filtered cheaply by `event_type` string (no import needed).
- Failure isolation — Langfuse outage doesn't break pipeline execution.

**Negative:**
- Event serialization cost on every emit (mitigated by Pydantic's fast serialization).
- Three copies of every event in different formats (file, Langfuse, DB). Disk/storage overhead.
- Sink ordering is not guaranteed. Events may appear slightly out of order across sinks.

**Alternative considered:** Single unified log with multiple formatters. Rejected because different consumers need fundamentally different data shapes (JSONL vs Langfuse spans vs SQL rows).
