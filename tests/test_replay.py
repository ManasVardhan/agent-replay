"""Tests for the replay engine."""

from agent_replay.recorder import Recorder
from agent_replay.replay import ReplayEngine
from agent_replay.trace import EventType


def _make_trace():
    with Recorder("replay-test") as rec:
        with rec.span("step-1"):
            rec.llm_request(model="gpt-4")
            rec.llm_response(content="hello")
        with rec.span("step-2"):
            rec.tool_call("search", {"q": "test"})
            rec.tool_result("search", {"r": "found"})
    return rec.trace


def test_replay_step_through():
    engine = ReplayEngine(_make_trace())
    assert engine.total_steps == 4
    assert engine.position == 0

    span, event = engine.step()
    assert event.event_type == EventType.LLM_REQUEST
    assert engine.position == 1


def test_replay_step_back():
    engine = ReplayEngine(_make_trace())
    engine.step()
    engine.step()
    assert engine.position == 2
    engine.step_back()
    assert engine.position == 1


def test_replay_search():
    engine = ReplayEngine(_make_trace())
    results = engine.search("search")
    assert len(results) >= 1


def test_replay_jump():
    engine = ReplayEngine(_make_trace())
    result = engine.jump(2)
    assert result is not None
    assert engine.position == 2
