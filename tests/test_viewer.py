"""Tests for the Rich terminal viewer."""

from __future__ import annotations


from rich.console import Console

from agent_replay.diff import DiffResult, diff_traces
from agent_replay.recorder import Recorder
from agent_replay.replay import ReplayEngine
from agent_replay.trace import EventType, Trace
from agent_replay.viewer import TraceViewer


def _make_trace(name: str = "test-trace") -> Trace:
    """Build a small trace with diverse event types for viewer testing."""
    with Recorder(name) as rec:
        with rec.span("planning"):
            rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            rec.llm_response(content="I'll search for that.", tokens=12)
            rec.decision("next action", choice="search")
        with rec.span("execution"):
            rec.tool_call("web_search", {"query": "python docs"})
            rec.tool_result("web_search", {"url": "https://docs.python.org"})
            rec.state_change("status", old="searching", new="done")
            rec.log("Search completed successfully")
        with rec.span("error-span"):
            rec.error("Connection timeout", exception="TimeoutError")
    return rec.trace


class TestTraceViewerShowTrace:
    """Tests for the show_trace method."""

    def test_show_trace_runs_without_error(self):
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        trace = _make_trace()
        trace.close()
        # Should not raise
        viewer.show_trace(trace)

    def test_show_trace_empty(self):
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        trace = Trace(name="empty")
        trace.close()
        viewer.show_trace(trace)

    def test_show_trace_running(self):
        """Trace without end_time should show 'running' for duration."""
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        trace = _make_trace()
        # Don't close, so duration is None
        viewer.show_trace(trace)


class TestTraceViewerShowTree:
    """Tests for the show_tree method."""

    def test_show_tree_runs_without_error(self):
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        trace = _make_trace()
        trace.close()
        viewer.show_tree(trace)

    def test_show_tree_empty_trace(self):
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        trace = Trace(name="empty")
        viewer.show_tree(trace)

    def test_show_tree_nested_spans(self):
        """Nested spans should render as tree children."""
        trace = Trace(name="nested")
        parent = trace.add_span("parent")
        child = trace.add_span("child", parent_id=parent.span_id)
        child.add_event(EventType.LOG, {"message": "nested event"})
        parent.close()
        child.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_tree(trace)


class TestTraceViewerShowDiff:
    """Tests for the show_diff method."""

    def test_show_diff_identical(self):
        trace_a = _make_trace("a")
        trace_a.close()
        trace_b = _make_trace("b")
        trace_b.close()
        result = diff_traces(trace_a, trace_b)

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_diff(result)

    def test_show_diff_with_divergences(self):
        trace_a = Trace(name="a")
        s_a = trace_a.add_span("step")
        s_a.add_event(EventType.TOOL_CALL, {"tool": "search", "args": {}})
        s_a.close()
        trace_a.close()

        trace_b = Trace(name="b")
        s_b = trace_b.add_span("step")
        s_b.add_event(EventType.TOOL_CALL, {"tool": "fetch", "args": {}})
        s_b.close()
        trace_b.close()

        result = diff_traces(trace_a, trace_b)
        assert not result.identical

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_diff(result)

    def test_show_diff_empty(self):
        result = DiffResult(trace_a_id="a", trace_b_id="b")
        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_diff(result)


class TestTraceViewerShowStep:
    """Tests for the show_step method (replay view)."""

    def test_show_step_at_beginning(self):
        trace = _make_trace()
        trace.close()
        engine = ReplayEngine(trace)

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_step(engine)

    def test_show_step_at_end(self):
        trace = _make_trace()
        trace.close()
        engine = ReplayEngine(trace)
        # Step through all events
        while engine.has_next():
            engine.step()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_step(engine)  # Should print "End of trace"

    def test_show_step_all_event_types(self):
        """Verify each event type renders without error."""
        trace = _make_trace()
        trace.close()
        engine = ReplayEngine(trace)

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        while engine.has_next():
            viewer.show_step(engine)
            engine.step()


class TestTraceViewerDefaultConsole:
    """Test that viewer works with default console."""

    def test_default_console(self):
        viewer = TraceViewer()
        assert viewer.console is not None


class TestViewerEventRendering:
    """Test specific event type rendering paths in _show_span."""

    def test_long_llm_response_truncated(self):
        """LLM responses longer than 80 chars should be truncated."""
        trace = Trace(name="long-response")
        span = trace.add_span("test")
        span.add_event(EventType.LLM_RESPONSE, {
            "content": "A" * 200,
            "tokens": 50,
        })
        span.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_trace(trace)

    def test_long_tool_result_truncated(self):
        """Tool results longer than 60 chars should be truncated."""
        trace = Trace(name="long-result")
        span = trace.add_span("test")
        span.add_event(EventType.TOOL_RESULT, {
            "tool": "fetch",
            "result": "B" * 200,
        })
        span.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_trace(trace)

    def test_llm_response_no_tokens(self):
        """LLM response without tokens field."""
        trace = Trace(name="no-tokens")
        span = trace.add_span("test")
        span.add_event(EventType.LLM_RESPONSE, {"content": "Hello"})
        span.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_trace(trace)

    def test_state_change_rendering(self):
        trace = Trace(name="state-change")
        span = trace.add_span("test")
        span.add_event(EventType.STATE_CHANGE, {"key": "status", "old": "a", "new": "b"})
        span.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_trace(trace)

    def test_generic_event_rendering(self):
        """Events like LOG that just show data/message."""
        trace = Trace(name="log-event")
        span = trace.add_span("test")
        span.add_event(EventType.LOG, {"message": "Something happened", "level": "info"})
        span.close()
        trace.close()

        console = Console(file=None, force_terminal=True, width=120)
        viewer = TraceViewer(console)
        viewer.show_trace(trace)
