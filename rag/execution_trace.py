"""Structured per-request execution events for RAG observability."""

from dataclasses import dataclass, field
from collections.abc import Callable
from time import perf_counter
from typing import Literal


TraceStatus = Literal["completed", "failed", "skipped"]


@dataclass(frozen=True)
class TraceEvent:
    """One completed, failed, or skipped stage in request order."""

    stage: str
    label: str
    status: TraceStatus
    duration_ms: float
    details: dict[str, object] = field(default_factory=dict)


TraceEventCallback = Callable[[TraceEvent], None]


class TraceRecorder:
    """Collect ordered trace events during one RAG request."""

    def __init__(self, on_event: TraceEventCallback | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._on_event = on_event

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return an immutable snapshot of recorded events."""
        return tuple(self._events)

    def record(
        self,
        *,
        stage: str,
        label: str,
        status: TraceStatus = "completed",
        started_at: float | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append one event, calculating elapsed time when a start is provided."""
        duration_ms = 0.0
        if started_at is not None:
            duration_ms = max(0.0, (perf_counter() - started_at) * 1_000)
        event = TraceEvent(
            stage=stage,
            label=label,
            status=status,
            duration_ms=duration_ms,
            details=details or {},
        )
        self._events.append(event)
        if self._on_event is not None:
            self._on_event(event)


def start_timer() -> float:
    """Return a monotonic timestamp for a later trace event."""
    return perf_counter()
