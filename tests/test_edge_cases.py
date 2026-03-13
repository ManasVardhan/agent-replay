"""Edge case and robustness tests for agent-replay."""

from __future__ import annotations

import json
from pathlib import Path


from agent_replay.recorder import Recorder
from agent_replay.replay import ReplayEngine
from agent_replay.trace import EventType, Span, Trace
from agent_replay.diff import diff_traces
from agent_replay.exporters import export_html, export_json


class TestEmptyTrace:
    """Tests for traces with no spans or events."""

    def test_empty_trace_properties(self) -> None:
        trace = Trace(name="empty")
        assert trace.event_count == 0
        assert len(trace.spans) == 0
        assert trace.duration is None

    def test_empty_trace_all_events(self) -> None:
        trace = Trace(name="empty")
        assert trace.all_events() == []

    def test_empty_trace_save_load(self, tmp_path: Path) -> None:
        trace = Trace(name="empty")
        path = tmp_path / "empty.jsonl"
        trace.save(path)
        loaded = Trace.load(path)
        assert loaded.name == "empty"
        assert len(loaded.spans) == 0

    def test_empty_trace_export_json(self, tmp_path: Path) -> None:
        trace = Trace(name="empty")
        out = export_json(trace, tmp_path / "empty.json")
        data = json.loads(out.read_text())
        assert data["spans"] == []

    def test_empty_trace_export_html(self, tmp_path: Path) -> None:
        trace = Trace(name="empty")
        out = export_html(trace, tmp_path / "empty.html")
        html = out.read_text()
        assert "empty" in html

    def test_empty_trace_diff(self) -> None:
        a = Trace(name="a")
        b = Trace(name="b")
        result = diff_traces(a, b)
        assert result.identical

    def test_empty_replay(self) -> None:
        trace = Trace(name="empty")
        engine = ReplayEngine(trace)
        assert engine.total_steps == 0
        assert not engine.has_next()
        assert engine.step() is None


class TestSpanEdgeCases:
    """Edge cases for span handling."""

    def test_span_without_events(self) -> None:
        trace = Trace(name="no-events")
        trace.add_span("empty-span")
        assert trace.event_count == 0
        assert len(trace.spans) == 1

    def test_span_roundtrip(self) -> None:
        span = Span(name="test", metadata={"key": "value"})
        span.add_event(EventType.LOG, {"message": "hello"})
        span.close()
        d = span.to_dict()
        restored = Span.from_dict(d)
        assert restored.name == span.name
        assert restored.metadata == {"key": "value"}
        assert len(restored.events) == 1

    def test_get_nonexistent_span(self) -> None:
        trace = Trace(name="test")
        trace.add_span("exists")
        assert trace.get_span("nonexistent") is None

    def test_get_existing_span(self) -> None:
        trace = Trace(name="test")
        span = trace.add_span("target")
        found = trace.get_span(span.span_id)
        assert found is not None
        assert found.name == "target"

    def test_span_close_idempotent(self) -> None:
        span = Span(name="test")
        span.close()
        end1 = span.end_time
        assert end1 is not None
        # Closing again updates the time
        span.close()
        assert span.end_time is not None


class TestRecorderEdgeCases:
    """Edge cases for the recorder."""

    def test_event_without_explicit_span(self) -> None:
        """Events recorded without an explicit span should auto-create a default span."""
        with Recorder("auto-span") as rec:
            rec.log("no span opened")
        assert len(rec.trace.spans) == 1
        assert rec.trace.spans[0].name == "default"

    def test_error_event(self) -> None:
        with Recorder("errors") as rec:
            with rec.span("failing"):
                rec.error("something broke", exception="ValueError")
        events = rec.trace.all_events()
        assert events[0].event_type == EventType.ERROR
        assert events[0].data["message"] == "something broke"
        assert events[0].data["exception"] == "ValueError"

    def test_state_change_event(self) -> None:
        with Recorder("states") as rec:
            with rec.span("transition"):
                rec.state_change("status", old="idle", new="running")
        events = rec.trace.all_events()
        assert events[0].event_type == EventType.STATE_CHANGE
        assert events[0].data["old"] == "idle"
        assert events[0].data["new"] == "running"

    def test_metadata_preserved(self) -> None:
        with Recorder("meta", metadata={"env": "test", "user": "ci"}) as rec:
            with rec.span("s"):
                rec.log("test")
        assert rec.trace.metadata == {"env": "test", "user": "ci"}

    def test_deeply_nested_spans(self) -> None:
        with Recorder("deep") as rec:
            with rec.span("level-1"):
                with rec.span("level-2"):
                    with rec.span("level-3"):
                        rec.log("deep event")
        assert len(rec.trace.spans) == 3
        # Check nesting
        assert rec.trace.spans[1].parent_id == rec.trace.spans[0].span_id
        assert rec.trace.spans[2].parent_id == rec.trace.spans[1].span_id

    def test_finish_returns_trace(self) -> None:
        rec = Recorder("manual")
        with rec.span("s"):
            rec.log("test")
        trace = rec.finish()
        assert trace.name == "manual"
        assert trace.end_time is not None


class TestReplayEdgeCases:
    """Edge cases for the replay engine."""

    def test_step_past_end(self) -> None:
        trace = Trace(name="short")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {"message": "only"})
        engine = ReplayEngine(trace)
        engine.step()
        assert engine.step() is None

    def test_step_back_at_start(self) -> None:
        trace = Trace(name="test")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {})
        engine = ReplayEngine(trace)
        assert engine.step_back() is None

    def test_jump_out_of_range(self) -> None:
        trace = Trace(name="test")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {})
        engine = ReplayEngine(trace)
        assert engine.jump(-1) is None
        assert engine.jump(999) is None

    def test_reset(self) -> None:
        with Recorder("test") as rec:
            with rec.span("s"):
                rec.log("a")
                rec.log("b")
        engine = ReplayEngine(rec.trace)
        engine.step()
        engine.step()
        assert engine.position == 2
        engine.reset()
        assert engine.position == 0

    def test_peek_does_not_advance(self) -> None:
        with Recorder("test") as rec:
            with rec.span("s"):
                rec.log("a")
        engine = ReplayEngine(rec.trace)
        result = engine.peek()
        assert result is not None
        assert engine.position == 0

    def test_current_span_events_at_end(self) -> None:
        trace = Trace(name="test")
        engine = ReplayEngine(trace)
        assert engine.current_span_events() == []

    def test_search_no_results(self) -> None:
        with Recorder("test") as rec:
            with rec.span("s"):
                rec.log("hello world")
        engine = ReplayEngine(rec.trace)
        results = engine.search("nonexistent_query_xyz")
        assert results == []

    def test_from_file(self, tmp_path: Path) -> None:
        with Recorder("file-test") as rec:
            with rec.span("s"):
                rec.log("test")
        path = tmp_path / "replay.jsonl"
        rec.trace.save(path)
        engine = ReplayEngine.from_file(path)
        assert engine.total_steps == 1


class TestDiffEdgeCases:
    """Edge cases for trace diffing."""

    def test_diff_one_empty(self) -> None:
        a = Trace(name="a")
        span = a.add_span("s")
        span.add_event(EventType.LOG, {})
        b = Trace(name="b")
        result = diff_traces(a, b)
        assert not result.identical
        assert len(result.divergences) == 1

    def test_diff_event_type_mismatch(self) -> None:
        with Recorder("a") as r:
            with r.span("s"):
                r.llm_request(model="gpt-4")
        with Recorder("b") as r2:
            with r2.span("s"):
                r2.tool_call("search", {})
        result = diff_traces(r.trace, r2.trace)
        assert result.critical_count >= 1

    def test_diff_decision_divergence(self) -> None:
        with Recorder("a") as r:
            with r.span("s"):
                r.decision("action", choice="search")
        with Recorder("b") as r2:
            with r2.span("s"):
                r2.decision("action", choice="browse")
        result = diff_traces(r.trace, r2.trace)
        assert result.critical_count >= 1
        assert "Decision divergence" in result.divergences[0].description

    def test_diff_llm_response_content_differs(self) -> None:
        with Recorder("a") as r:
            with r.span("s"):
                r.llm_response(content="hello")
        with Recorder("b") as r2:
            with r2.span("s"):
                r2.llm_response(content="goodbye")
        result = diff_traces(r.trace, r2.trace)
        assert not result.identical
        assert any("LLM response" in d.description for d in result.divergences)

    def test_diff_summary_identical(self) -> None:
        a = Trace(name="a")
        b = Trace(name="b")
        result = diff_traces(a, b)
        assert "identical" in result.summary.lower()

    def test_diff_to_dict(self) -> None:
        with Recorder("a") as r:
            with r.span("s"):
                r.tool_call("search", {})
        with Recorder("b") as r2:
            with r2.span("s"):
                r2.tool_call("browse", {})
        result = diff_traces(r.trace, r2.trace)
        d = result.to_dict()
        assert "divergences" in d
        assert "identical" in d
        assert isinstance(d["divergence_count"], int)


class TestTraceLoadEdgeCases:
    """Edge cases for loading traces."""

    def test_load_with_blank_lines(self, tmp_path: Path) -> None:
        """JSONL files with blank lines should be handled."""
        path = tmp_path / "blanks.jsonl"
        trace = Trace(name="blanks")
        trace.save(path)
        # Insert blank lines
        content = path.read_text()
        path.write_text(content + "\n\n\n")
        loaded = Trace.load(path)
        assert loaded.name == "blanks"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        trace = Trace(name="nested")
        path = tmp_path / "deep" / "nested" / "trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.save(path)
        assert path.exists()

    def test_trace_metadata_roundtrip(self, tmp_path: Path) -> None:
        trace = Trace(name="meta", metadata={"env": "prod", "version": 42})
        path = tmp_path / "meta.jsonl"
        trace.save(path)
        loaded = Trace.load(path)
        assert loaded.metadata == {"env": "prod", "version": 42}
