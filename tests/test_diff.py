"""Tests for the diff tool."""

from agent_replay.diff import diff_traces
from agent_replay.recorder import Recorder
from agent_replay.trace import EventType


def test_diff_identical():
    with Recorder("a") as r:
        with r.span("s"):
            r.llm_request(model="gpt-4")
            r.llm_response(content="hi")
    trace_a = r.trace

    with Recorder("b") as r:
        with r.span("s"):
            r.llm_request(model="gpt-4")
            r.llm_response(content="hi")
    trace_b = r.trace

    result = diff_traces(trace_a, trace_b)
    assert result.identical


def test_diff_different_tools():
    with Recorder("a") as r:
        with r.span("s"):
            r.tool_call("search", {})
    trace_a = r.trace

    with Recorder("b") as r:
        with r.span("s"):
            r.tool_call("browse", {})
    trace_b = r.trace

    result = diff_traces(trace_a, trace_b)
    assert not result.identical
    assert result.critical_count >= 1


def test_diff_extra_events():
    with Recorder("a") as r:
        with r.span("s"):
            r.llm_request(model="gpt-4")
    trace_a = r.trace

    with Recorder("b") as r:
        with r.span("s"):
            r.llm_request(model="gpt-4")
            r.llm_response(content="extra")
    trace_b = r.trace

    result = diff_traces(trace_a, trace_b)
    assert not result.identical
    assert len(result.divergences) == 1
