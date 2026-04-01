"""Tests for Trace.filter_events() and Trace.event_type_counts()."""

from agent_replay.recorder import Recorder
from agent_replay.trace import EventType, Trace


def _make_trace() -> Trace:
    """Create a trace with a mix of event types."""
    with Recorder("test-agent") as rec:
        with rec.span("planning"):
            rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
            rec.llm_response(content="Hi!", tokens=5)
            rec.decision("next step", choice="search")
        with rec.span("execution"):
            rec.tool_call("search", {"q": "python"})
            rec.tool_result("search", {"url": "https://python.org"})
            rec.llm_request(model="gpt-4")
            rec.llm_response(content="Done", tokens=3)
        with rec.span("cleanup"):
            rec.log("cleaning up")
            rec.error("minor issue", exception="ValueError")
    return rec.trace


class TestFilterEvents:
    def test_filter_single_type(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.LLM_REQUEST)
        assert len(results) == 2
        assert all(e.event_type == EventType.LLM_REQUEST for e in results)

    def test_filter_multiple_types(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.TOOL_CALL, EventType.TOOL_RESULT)
        assert len(results) == 2
        types = {e.event_type for e in results}
        assert types == {EventType.TOOL_CALL, EventType.TOOL_RESULT}

    def test_filter_no_match(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.STATE_CHANGE)
        assert results == []

    def test_filter_all_types(self) -> None:
        trace = _make_trace()
        all_types = list(EventType)
        results = trace.filter_events(*all_types)
        assert len(results) == trace.event_count

    def test_filter_sorted_by_timestamp(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.LLM_REQUEST, EventType.LLM_RESPONSE)
        timestamps = [e.timestamp for e in results]
        assert timestamps == sorted(timestamps)

    def test_filter_empty_trace(self) -> None:
        trace = Trace(name="empty")
        results = trace.filter_events(EventType.LLM_REQUEST)
        assert results == []

    def test_filter_errors_only(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.ERROR)
        assert len(results) == 1
        assert results[0].data["message"] == "minor issue"

    def test_filter_decisions(self) -> None:
        trace = _make_trace()
        results = trace.filter_events(EventType.DECISION)
        assert len(results) == 1
        assert results[0].data["choice"] == "search"


class TestEventTypeCounts:
    def test_counts_basic(self) -> None:
        trace = _make_trace()
        counts = trace.event_type_counts()
        assert counts["llm_request"] == 2
        assert counts["llm_response"] == 2
        assert counts["tool_call"] == 1
        assert counts["tool_result"] == 1
        assert counts["decision"] == 1
        assert counts["log"] == 1
        assert counts["error"] == 1

    def test_counts_empty_trace(self) -> None:
        trace = Trace(name="empty")
        counts = trace.event_type_counts()
        assert counts == {}

    def test_counts_sum_equals_event_count(self) -> None:
        trace = _make_trace()
        counts = trace.event_type_counts()
        assert sum(counts.values()) == trace.event_count

    def test_counts_single_type_trace(self) -> None:
        with Recorder("single-type") as rec:
            with rec.span("work"):
                rec.log("msg1")
                rec.log("msg2")
                rec.log("msg3")
        counts = rec.trace.event_type_counts()
        assert counts == {"log": 3}
