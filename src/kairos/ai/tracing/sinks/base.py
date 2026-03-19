"""Kairos Agent -- TracingSink ABC.

All observability consumers implement this interface.  The ``RunTracer``
dispatches events to registered sinks; each sink decides what to do with
them (write to files, send to Langfuse, persist to database, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kairos.ai.tracing.events import TraceEvent


class TracingSink(ABC):
    """Interface for tracing event consumers."""

    @abstractmethod
    def on_event(self, event: TraceEvent) -> None:
        """Handle a single tracing event.

        Implementations MUST NOT raise -- failures should be logged
        and swallowed to avoid disrupting the pipeline.
        """

    def flush(self) -> None:
        """Flush any buffered data.  Called at run end."""

    def close(self) -> None:
        """Release resources.  Called at run end after flush."""
