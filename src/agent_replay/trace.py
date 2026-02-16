"""Trace data model: spans, events, tool calls, and full traces."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    """Types of events that can occur in an agent trace."""
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DECISION = "decision"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    LOG = "log"


@dataclass
class Event:
    """A single event within a span."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        return cls(
            event_type=EventType(d["event_type"]),
            timestamp=d["timestamp"],
            data=d.get("data", {}),
            event_id=d.get("event_id", uuid.uuid4().hex[:12]),
        )


@dataclass
class Span:
    """A named execution span containing events. Spans can nest."""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def add_event(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        event = Event(event_type=event_type, data=data or {})
        self.events.append(event)
        return event

    def close(self) -> None:
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        return cls(
            name=d["name"],
            span_id=d["span_id"],
            parent_id=d.get("parent_id"),
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            events=[Event.from_dict(e) for e in d.get("events", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass
class Trace:
    """A complete execution trace consisting of ordered spans."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = "unnamed"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def event_count(self) -> int:
        return sum(len(s.events) for s in self.spans)

    def add_span(self, name: str, parent_id: str | None = None, **kwargs: Any) -> Span:
        span = Span(name=name, parent_id=parent_id, **kwargs)
        self.spans.append(span)
        return span

    def close(self) -> None:
        self.end_time = time.time()
        for span in self.spans:
            if span.end_time is None:
                span.close()

    def get_span(self, span_id: str) -> Span | None:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def all_events(self) -> list[Event]:
        """Return all events across spans, sorted by timestamp."""
        events = []
        for span in self.spans:
            events.extend(span.events)
        return sorted(events, key=lambda e: e.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        return cls(
            trace_id=d["trace_id"],
            name=d.get("name", "unnamed"),
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        """Save trace as JSONL (one line per span, header first)."""
        path = Path(path)
        with open(path, "w") as f:
            header = {
                "type": "trace_header",
                "trace_id": self.trace_id,
                "name": self.name,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "metadata": self.metadata,
            }
            f.write(json.dumps(header) + "\n")
            for span in self.spans:
                record = {"type": "span", **span.to_dict()}
                f.write(json.dumps(record) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Load trace from a JSONL file."""
        path = Path(path)
        spans: list[Span] = []
        header: dict[str, Any] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "trace_header":
                    header = record
                elif record.get("type") == "span":
                    record.pop("type", None)
                    spans.append(Span.from_dict(record))
        return cls(
            trace_id=header.get("trace_id", uuid.uuid4().hex[:16]),
            name=header.get("name", "unnamed"),
            start_time=header.get("start_time", 0),
            end_time=header.get("end_time"),
            spans=spans,
            metadata=header.get("metadata", {}),
        )
