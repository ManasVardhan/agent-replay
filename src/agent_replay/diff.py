"""Diff tool: compare two traces and find divergence points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace import Event, EventType, Span, Trace


@dataclass
class Divergence:
    """A point where two traces differ."""
    position: int
    description: str
    trace_a_event: Event | None = None
    trace_b_event: Event | None = None
    trace_a_span: str = ""
    trace_b_span: str = ""
    severity: str = "info"  # info, warning, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "description": self.description,
            "severity": self.severity,
            "trace_a_span": self.trace_a_span,
            "trace_b_span": self.trace_b_span,
            "trace_a_event": self.trace_a_event.to_dict() if self.trace_a_event else None,
            "trace_b_event": self.trace_b_event.to_dict() if self.trace_b_event else None,
        }


@dataclass
class DiffResult:
    """Result of comparing two traces."""
    trace_a_id: str
    trace_b_id: str
    divergences: list[Divergence] = field(default_factory=list)
    summary: str = ""

    @property
    def identical(self) -> bool:
        return len(self.divergences) == 0

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.divergences if d.severity == "critical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_a_id": self.trace_a_id,
            "trace_b_id": self.trace_b_id,
            "identical": self.identical,
            "divergence_count": len(self.divergences),
            "critical_count": self.critical_count,
            "summary": self.summary,
            "divergences": [d.to_dict() for d in self.divergences],
        }


def diff_traces(trace_a: Trace, trace_b: Trace) -> DiffResult:
    """Compare two traces and return divergence points.

    Compares event sequences across both traces, checking for:
    - Different event types at the same position
    - Different tool calls
    - Different LLM responses
    - Missing or extra events
    """
    result = DiffResult(trace_a_id=trace_a.trace_id, trace_b_id=trace_b.trace_id)
    events_a = trace_a.all_events()
    events_b = trace_b.all_events()

    # Build span lookup
    span_lookup_a = {s.span_id: s.name for s in trace_a.spans}
    span_lookup_b = {s.span_id: s.name for s in trace_b.spans}

    def _span_for_event(trace: Trace, event: Event) -> str:
        for span in trace.spans:
            if any(e.event_id == event.event_id for e in span.events):
                return span.name
        return "unknown"

    max_len = max(len(events_a), len(events_b))

    for i in range(max_len):
        ea = events_a[i] if i < len(events_a) else None
        eb = events_b[i] if i < len(events_b) else None

        if ea is None:
            result.divergences.append(Divergence(
                position=i,
                description=f"Trace B has extra event: {eb.event_type.value}",  # type: ignore
                trace_b_event=eb,
                trace_b_span=_span_for_event(trace_b, eb),  # type: ignore
                severity="warning",
            ))
            continue

        if eb is None:
            result.divergences.append(Divergence(
                position=i,
                description=f"Trace A has extra event: {ea.event_type.value}",
                trace_a_event=ea,
                trace_a_span=_span_for_event(trace_a, ea),
                severity="warning",
            ))
            continue

        # Compare event types
        if ea.event_type != eb.event_type:
            result.divergences.append(Divergence(
                position=i,
                description=f"Event type divergence: {ea.event_type.value} vs {eb.event_type.value}",
                trace_a_event=ea,
                trace_b_event=eb,
                trace_a_span=_span_for_event(trace_a, ea),
                trace_b_span=_span_for_event(trace_b, eb),
                severity="critical",
            ))
            continue

        # Compare event data for meaningful differences
        if ea.event_type == EventType.TOOL_CALL:
            if ea.data.get("tool") != eb.data.get("tool"):
                result.divergences.append(Divergence(
                    position=i,
                    description=f"Different tool called: {ea.data.get('tool')} vs {eb.data.get('tool')}",
                    trace_a_event=ea,
                    trace_b_event=eb,
                    trace_a_span=_span_for_event(trace_a, ea),
                    trace_b_span=_span_for_event(trace_b, eb),
                    severity="critical",
                ))

        elif ea.event_type == EventType.LLM_RESPONSE:
            content_a = ea.data.get("content", "")
            content_b = eb.data.get("content", "")
            if content_a != content_b:
                result.divergences.append(Divergence(
                    position=i,
                    description="LLM response content differs",
                    trace_a_event=ea,
                    trace_b_event=eb,
                    trace_a_span=_span_for_event(trace_a, ea),
                    trace_b_span=_span_for_event(trace_b, eb),
                    severity="info",
                ))

        elif ea.event_type == EventType.DECISION:
            if ea.data.get("choice") != eb.data.get("choice"):
                result.divergences.append(Divergence(
                    position=i,
                    description=f"Decision divergence: '{ea.data.get('choice')}' vs '{eb.data.get('choice')}'",
                    trace_a_event=ea,
                    trace_b_event=eb,
                    trace_a_span=_span_for_event(trace_a, ea),
                    trace_b_span=_span_for_event(trace_b, eb),
                    severity="critical",
                ))

    # Generate summary
    n = len(result.divergences)
    if n == 0:
        result.summary = "Traces are identical in structure and content."
    else:
        result.summary = (
            f"Found {n} divergence(s): "
            f"{result.critical_count} critical, "
            f"{n - result.critical_count} informational."
        )

    return result
