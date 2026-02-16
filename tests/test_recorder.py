"""Tests for the recorder."""

import tempfile
from pathlib import Path

from agent_replay.recorder import Recorder, record_trace
from agent_replay.trace import EventType, Trace


def test_recorder_basic():
    with Recorder("test") as rec:
        with rec.span("step-1"):
            rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            rec.llm_response(content="hello!", tokens=5)

    trace = rec.trace
    assert trace.name == "test"
    assert len(trace.spans) == 1
    assert len(trace.spans[0].events) == 2
    assert trace.spans[0].events[0].event_type == EventType.LLM_REQUEST


def test_recorder_tool_call():
    with Recorder("tools") as rec:
        with rec.span("tool-use"):
            rec.tool_call("search", {"query": "python"})
            rec.tool_result("search", {"results": ["python.org"]})

    events = rec.trace.all_events()
    assert events[0].event_type == EventType.TOOL_CALL
    assert events[1].event_type == EventType.TOOL_RESULT


def test_recorder_nested_spans():
    with Recorder("nested") as rec:
        with rec.span("outer"):
            rec.log("starting")
            with rec.span("inner"):
                rec.decision("choose tool", "search")

    assert len(rec.trace.spans) == 2
    inner = rec.trace.spans[1]
    assert inner.parent_id == rec.trace.spans[0].span_id


def test_recorder_saves_output(tmp_path: Path):
    output = tmp_path / "out.jsonl"
    with Recorder("save-test", output_path=output) as rec:
        with rec.span("s"):
            rec.log("test")

    assert output.exists()
    loaded = Trace.load(output)
    assert loaded.name == "save-test"


def test_record_trace_decorator(tmp_path: Path):
    output = tmp_path / "decorated.jsonl"

    @record_trace("decorated", output_path=output)
    def my_agent(task: str, recorder=None):
        with recorder.span("work"):
            recorder.llm_request(model="gpt-4")
            recorder.llm_response(content="done")
        return "result"

    result = my_agent("do stuff")
    assert result == "result"
    assert output.exists()
