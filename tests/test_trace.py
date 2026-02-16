"""Tests for the trace data model."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_replay.trace import Event, EventType, Span, Trace


def test_event_creation():
    event = Event(event_type=EventType.LLM_REQUEST, data={"model": "gpt-4"})
    assert event.event_type == EventType.LLM_REQUEST
    assert event.data["model"] == "gpt-4"
    assert event.event_id


def test_event_roundtrip():
    event = Event(event_type=EventType.TOOL_CALL, data={"tool": "search", "args": {"q": "test"}})
    d = event.to_dict()
    restored = Event.from_dict(d)
    assert restored.event_type == event.event_type
    assert restored.data == event.data


def test_span_add_event():
    span = Span(name="test-span")
    event = span.add_event(EventType.LOG, {"message": "hello"})
    assert len(span.events) == 1
    assert span.events[0].data["message"] == "hello"


def test_span_duration():
    span = Span(name="test", start_time=100.0)
    assert span.duration is None
    span.end_time = 102.5
    assert span.duration == 2.5


def test_trace_save_load(tmp_path: Path):
    trace = Trace(name="test-trace")
    span = trace.add_span("step-1")
    span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4"})
    span.add_event(EventType.LLM_RESPONSE, {"content": "Hello!", "tokens": 10})
    span.close()
    trace.close()

    path = tmp_path / "trace.jsonl"
    trace.save(path)

    loaded = Trace.load(path)
    assert loaded.trace_id == trace.trace_id
    assert loaded.name == "test-trace"
    assert len(loaded.spans) == 1
    assert len(loaded.spans[0].events) == 2


def test_trace_all_events():
    trace = Trace(name="multi")
    s1 = trace.add_span("a")
    s1.add_event(EventType.LOG, {"message": "first"})
    s2 = trace.add_span("b")
    s2.add_event(EventType.LOG, {"message": "second"})
    events = trace.all_events()
    assert len(events) == 2


def test_trace_event_count():
    trace = Trace(name="count")
    s = trace.add_span("s")
    s.add_event(EventType.LOG, {})
    s.add_event(EventType.ERROR, {})
    assert trace.event_count == 2
