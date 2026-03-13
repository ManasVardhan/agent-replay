"""Extended tests for the recorder module."""

from __future__ import annotations

from pathlib import Path

from agent_replay.recorder import Recorder, record_trace
from agent_replay.trace import EventType


class TestRecorderContextManager:
    def test_context_manager_closes_trace(self):
        with Recorder("test") as rec:
            with rec.span("s1"):
                rec.log("hello")
        assert rec.trace.end_time is not None

    def test_finish_saves_output(self, tmp_path: Path):
        rec = Recorder("test", output_path=tmp_path / "out.jsonl")
        with rec.span("s1"):
            rec.log("hello")
        rec.finish()
        assert (tmp_path / "out.jsonl").exists()

    def test_finish_returns_trace(self):
        rec = Recorder("test")
        with rec.span("s1"):
            rec.log("hello")
        trace = rec.finish()
        assert trace.name == "test"
        assert trace.end_time is not None


class TestRecorderEvents:
    def test_all_event_types(self):
        with Recorder("full-test") as rec:
            with rec.span("main"):
                rec.llm_request(model="gpt-4", messages=[])
                rec.llm_response(content="reply", tokens=10)
                rec.tool_call("search", {"q": "test"})
                rec.tool_result("search", {"results": []})
                rec.decision("next", choice="stop")
                rec.state_change("status", old="running", new="done")
                rec.log("completed", level="info")
                rec.error("oops", exception="ValueError")
                rec.event(EventType.LOG, {"custom": "data"})
        events = rec.trace.all_events()
        types = {e.event_type for e in events}
        assert EventType.LLM_REQUEST in types
        assert EventType.LLM_RESPONSE in types
        assert EventType.TOOL_CALL in types
        assert EventType.TOOL_RESULT in types
        assert EventType.DECISION in types
        assert EventType.STATE_CHANGE in types
        assert EventType.LOG in types
        assert EventType.ERROR in types

    def test_llm_request_extra_kwargs(self):
        with Recorder("test") as rec:
            with rec.span("main"):
                rec.llm_request(model="gpt-4", messages=[], temperature=0.7)
        event = rec.trace.all_events()[0]
        assert event.data["temperature"] == 0.7

    def test_tool_call_empty_args(self):
        with Recorder("test") as rec:
            with rec.span("main"):
                rec.tool_call("tool_name")
        event = rec.trace.all_events()[0]
        assert event.data["args"] == {}

    def test_error_no_exception(self):
        with Recorder("test") as rec:
            with rec.span("main"):
                rec.error("something went wrong")
        event = rec.trace.all_events()[0]
        assert event.data["exception"] is None


class TestRecorderSpans:
    def test_auto_span_created(self):
        """When no explicit span exists, a default span is created."""
        with Recorder("test") as rec:
            rec.log("no span yet")
        assert len(rec.trace.spans) == 1
        assert rec.trace.spans[0].name == "default"

    def test_nested_spans(self):
        with Recorder("test") as rec:
            with rec.span("outer"):
                rec.log("outer event")
                with rec.span("inner"):
                    rec.log("inner event")
                rec.log("back to outer")

        assert len(rec.trace.spans) == 2
        inner = [s for s in rec.trace.spans if s.name == "inner"][0]
        outer = [s for s in rec.trace.spans if s.name == "outer"][0]
        assert inner.parent_id == outer.span_id

    def test_deeply_nested_spans(self):
        with Recorder("test") as rec:
            with rec.span("l1"):
                with rec.span("l2"):
                    with rec.span("l3"):
                        rec.log("deep")
        assert len(rec.trace.spans) == 3

    def test_span_metadata(self):
        with Recorder("test") as rec:
            with rec.span("s1", metadata={"key": "value"}) as span:
                rec.log("hello")
        assert span.metadata["key"] == "value"

    def test_sequential_spans(self):
        with Recorder("test") as rec:
            with rec.span("first"):
                rec.log("a")
            with rec.span("second"):
                rec.log("b")
            with rec.span("third"):
                rec.log("c")
        assert len(rec.trace.spans) == 3
        # Each should have no parent (sequential, not nested)
        for span in rec.trace.spans:
            assert span.parent_id is None

    def test_span_closes_on_exit(self):
        with Recorder("test") as rec:
            with rec.span("s1") as span:
                rec.log("hello")
        assert span.end_time is not None
        assert span.duration is not None
        assert span.duration >= 0


class TestRecordTraceDecorator:
    def test_basic_decorator(self, tmp_path: Path):
        output = tmp_path / "decorated.jsonl"

        @record_trace("decorated-run", output_path=str(output))
        def my_agent(task: str, recorder=None):
            recorder.llm_request(model="gpt-4", messages=[{"role": "user", "content": task}])
            recorder.llm_response(content="Done!", tokens=5)
            return "result"

        result = my_agent("test task")
        assert result == "result"
        assert output.exists()

    def test_decorator_default_name(self):
        @record_trace()
        def agent_fn(recorder=None):
            recorder.log("hello")

        agent_fn()  # Should not raise

    def test_decorator_preserves_function_name(self):
        @record_trace("test")
        def my_named_function(recorder=None):
            pass

        assert my_named_function.__name__ == "my_named_function"


class TestRecorderMetadata:
    def test_trace_metadata(self):
        rec = Recorder("test", metadata={"env": "prod", "version": "1.0"})
        with rec.span("s1"):
            rec.log("hello")
        rec.finish()
        assert rec.trace.metadata["env"] == "prod"
        assert rec.trace.metadata["version"] == "1.0"

    def test_default_empty_metadata(self):
        rec = Recorder("test")
        assert rec.trace.metadata == {}
