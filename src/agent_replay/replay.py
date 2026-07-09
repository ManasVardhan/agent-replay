"""Replay engine: step through traces interactively."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .trace import Event, Span, Trace


@dataclass
class ReplayState:
    """Current position in a trace replay."""
    span_index: int = 0
    event_index: int = 0
    paused: bool = True


@dataclass
class PlaybackStep:
    """One step of a timed playback: how long to wait, then what to show.

    Attributes
    ----------
    delay : seconds to pause before showing this event (already speed-adjusted)
    elapsed : seconds since the first event in the original trace timeline
    span : the span the event belongs to
    event : the event itself
    """
    delay: float
    elapsed: float
    span: Span
    event: Event


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

    def playback_plan(
        self,
        speed: float = 1.0,
        max_delay: float | None = None,
    ) -> list[PlaybackStep]:
        """Build a timed playback plan preserving the trace's original pacing.

        Parameters
        ----------
        speed : playback speed multiplier; 2.0 plays twice as fast. Must be > 0.
        max_delay : optional cap in seconds for the pause before each event,
                    applied after speed adjustment. Useful for traces with
                    long idle gaps. Must be >= 0 when given.

        Returns
        -------
        List of PlaybackStep in timestamp order. The first step always has
        delay 0.0. Negative timestamp gaps are clamped to 0.
        """
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        if max_delay is not None and max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")

        steps: list[PlaybackStep] = []
        if not self._flat:
            return steps

        first_ts = self._flat[0][1].timestamp
        prev_ts = first_ts
        for span, event in self._flat:
            gap = max(0.0, event.timestamp - prev_ts)
            delay = gap / speed
            if max_delay is not None:
                delay = min(delay, max_delay)
            steps.append(
                PlaybackStep(
                    delay=delay,
                    elapsed=max(0.0, event.timestamp - first_ts),
                    span=span,
                    event=event,
                )
            )
            prev_ts = event.timestamp
        return steps

    def search(self, query: str) -> list[int]:
        """Find positions where event data contains the query string."""
        results: list[int] = []
        query_lower = query.lower()
        for i, (span, event) in enumerate(self._flat):
            searchable = f"{span.name} {event.event_type.value} {event.data}"
            if query_lower in searchable.lower():
                results.append(i)
        return results
