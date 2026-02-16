"""Replay engine: step through traces interactively."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .trace import Event, Span, Trace


@dataclass
class ReplayState:
    """Current position in a trace replay."""
    span_index: int = 0
    event_index: int = 0
    paused: bool = True


class ReplayEngine:
    """Step-through replay of a recorded trace.

    Example::

        engine = ReplayEngine.from_file("trace.jsonl")
        while engine.has_next():
            span, event = engine.step()
            print(f"[{span.name}] {event.event_type}: {event.data}")
    """

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self._flat: list[tuple[Span, Event]] = []
        self._position: int = 0
        self._build_flat_list()

    def _build_flat_list(self) -> None:
        """Flatten all spans/events into a sequential list sorted by timestamp."""
        pairs: list[tuple[Span, Event]] = []
        for span in self.trace.spans:
            for event in span.events:
                pairs.append((span, event))
        pairs.sort(key=lambda p: p[1].timestamp)
        self._flat = pairs

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayEngine:
        return cls(Trace.load(path))

    @property
    def total_steps(self) -> int:
        return len(self._flat)

    @property
    def position(self) -> int:
        return self._position

    def has_next(self) -> bool:
        return self._position < len(self._flat)

    def has_prev(self) -> bool:
        return self._position > 0

    def step(self) -> tuple[Span, Event] | None:
        """Advance one step and return (span, event), or None if at end."""
        if not self.has_next():
            return None
        result = self._flat[self._position]
        self._position += 1
        return result

    def step_back(self) -> tuple[Span, Event] | None:
        """Go back one step."""
        if not self.has_prev():
            return None
        self._position -= 1
        return self._flat[self._position]

    def peek(self) -> tuple[Span, Event] | None:
        """Look at current step without advancing."""
        if not self.has_next():
            return None
        return self._flat[self._position]

    def jump(self, position: int) -> tuple[Span, Event] | None:
        """Jump to a specific position."""
        if 0 <= position < len(self._flat):
            self._position = position
            return self._flat[self._position]
        return None

    def reset(self) -> None:
        """Reset to the beginning."""
        self._position = 0

    def current_span_events(self) -> list[tuple[Span, Event]]:
        """Get all events in the current span."""
        if not self._flat or self._position >= len(self._flat):
            return []
        current_span = self._flat[min(self._position, len(self._flat) - 1)][0]
        return [(s, e) for s, e in self._flat if s.span_id == current_span.span_id]

    def search(self, query: str) -> list[int]:
        """Find positions where event data contains the query string."""
        results: list[int] = []
        query_lower = query.lower()
        for i, (span, event) in enumerate(self._flat):
            searchable = f"{span.name} {event.event_type.value} {event.data}"
            if query_lower in searchable.lower():
                results.append(i)
        return results
