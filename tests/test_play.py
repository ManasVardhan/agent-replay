"""Tests for timed playback: ReplayEngine.playback_plan and the play CLI command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_replay import Event, EventType, PlaybackStep, ReplayEngine, Trace
from agent_replay.cli import cli


def _trace_with_gaps() -> Trace:
    """Trace with events at t=100.0, 100.5, 102.5, 110.5 (gaps 0.5, 2.0, 8.0)."""
    trace = Trace(name="play-test")
    span = trace.add_span("agent-loop")
    span.events = [
        Event(EventType.LLM_REQUEST, timestamp=100.0, data={"model": "gpt-4o"}),
        Event(EventType.LLM_RESPONSE, timestamp=100.5, data={"content": "hi", "tokens": 5}),
        Event(EventType.TOOL_CALL, timestamp=102.5, data={"tool": "search", "args": {"q": "x"}}),
        Event(EventType.TOOL_RESULT, timestamp=110.5, data={"tool": "search", "result": "ok"}),
    ]
    trace.close()
    return trace


def _save(trace: Trace, tmp_path: Path) -> Path:
    return trace.save(tmp_path / "trace.jsonl")


class TestPlaybackPlan:
    def test_empty_trace(self):
        assert ReplayEngine(Trace(name="empty")).playback_plan() == []

    def test_first_step_has_zero_delay(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan()
        assert plan[0].delay == 0.0
        assert plan[0].elapsed == 0.0

    def test_delays_match_timestamp_gaps(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan()
        assert [round(s.delay, 6) for s in plan] == [0.0, 0.5, 2.0, 8.0]

    def test_elapsed_is_cumulative_original_time(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan()
        assert [round(s.elapsed, 6) for s in plan] == [0.0, 0.5, 2.5, 10.5]

    def test_speed_divides_delays(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan(speed=2.0)
        assert [round(s.delay, 6) for s in plan] == [0.0, 0.25, 1.0, 4.0]

    def test_speed_does_not_change_elapsed(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan(speed=4.0)
        assert [round(s.elapsed, 6) for s in plan] == [0.0, 0.5, 2.5, 10.5]

    def test_max_delay_caps_pauses(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan(max_delay=1.5)
        assert [round(s.delay, 6) for s in plan] == [0.0, 0.5, 1.5, 1.5]

    def test_max_delay_applied_after_speed(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan(speed=2.0, max_delay=0.5)
        assert [round(s.delay, 6) for s in plan] == [0.0, 0.25, 0.5, 0.5]

    def test_zero_speed_raises(self):
        with pytest.raises(ValueError, match="speed must be > 0"):
            ReplayEngine(_trace_with_gaps()).playback_plan(speed=0)

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError, match="speed must be > 0"):
            ReplayEngine(_trace_with_gaps()).playback_plan(speed=-1.0)

    def test_negative_max_delay_raises(self):
        with pytest.raises(ValueError, match="max_delay must be >= 0"):
            ReplayEngine(_trace_with_gaps()).playback_plan(max_delay=-0.1)

    def test_steps_sorted_across_spans(self):
        trace = Trace(name="two-spans")
        s1 = trace.add_span("first")
        s2 = trace.add_span("second")
        s1.events = [Event(EventType.LOG, timestamp=10.0, data={"message": "a"})]
        s2.events = [Event(EventType.LOG, timestamp=5.0, data={"message": "b"})]
        plan = ReplayEngine(trace).playback_plan()
        assert [s.event.data["message"] for s in plan] == ["b", "a"]
        assert plan[0].span.name == "second"

    def test_equal_timestamps_zero_delay(self):
        trace = Trace(name="same-ts")
        span = trace.add_span("s")
        span.events = [
            Event(EventType.LOG, timestamp=1.0, data={"message": "a"}),
            Event(EventType.LOG, timestamp=1.0, data={"message": "b"}),
        ]
        plan = ReplayEngine(trace).playback_plan()
        assert plan[1].delay == 0.0

    def test_returns_playback_steps(self):
        plan = ReplayEngine(_trace_with_gaps()).playback_plan()
        assert all(isinstance(s, PlaybackStep) for s in plan)


class TestPlayCli:
    def test_play_no_delay_prints_all_events(self, tmp_path):
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--no-delay"])
        assert result.exit_code == 0
        assert "Playing: play-test" in result.output
        assert "[1/4]" in result.output
        assert "[4/4]" in result.output
        assert "llm_request" in result.output
        assert "tool_result" in result.output
        assert "Played 4 events" in result.output

    def test_play_shows_elapsed_timeline(self, tmp_path):
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--no-delay"])
        assert "+   0.000s" in result.output
        assert "+  10.500s" in result.output

    def test_play_shows_event_summaries(self, tmp_path):
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--no-delay"])
        assert "model=gpt-4o" in result.output
        assert "search" in result.output

    def test_play_sleeps_between_events(self, tmp_path, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("agent_replay.cli.time.sleep", sleeps.append)
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--speed", "2", "--max-delay", "10"])
        assert result.exit_code == 0
        assert [round(s, 6) for s in sleeps] == [0.25, 1.0, 4.0]

    def test_play_max_delay_default_caps_sleeps(self, tmp_path, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("agent_replay.cli.time.sleep", sleeps.append)
        path = _save(_trace_with_gaps(), tmp_path)
        CliRunner().invoke(cli, ["play", str(path)])
        assert max(sleeps) == 2.0

    def test_play_no_delay_never_sleeps(self, tmp_path, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("agent_replay.cli.time.sleep", sleeps.append)
        path = _save(_trace_with_gaps(), tmp_path)
        CliRunner().invoke(cli, ["play", str(path), "--no-delay"])
        assert sleeps == []

    def test_play_zero_speed_rejected(self, tmp_path):
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--speed", "0"])
        assert result.exit_code == 2
        assert "must be > 0" in result.output

    def test_play_negative_max_delay_rejected(self, tmp_path):
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--max-delay", "-1"])
        assert result.exit_code == 2
        assert "must be >= 0" in result.output

    def test_play_empty_trace(self, tmp_path):
        path = _save(Trace(name="empty"), tmp_path)
        result = CliRunner().invoke(cli, ["play", str(path), "--no-delay"])
        assert result.exit_code == 0
        assert "no events" in result.output

    def test_play_missing_file(self):
        result = CliRunner().invoke(cli, ["play", "/nonexistent/trace.jsonl"])
        assert result.exit_code == 2

    def test_play_keyboard_interrupt(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_sleep(_seconds: float) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr("agent_replay.cli.time.sleep", fake_sleep)
        path = _save(_trace_with_gaps(), tmp_path)
        result = CliRunner().invoke(
            cli, ["play", str(path)], standalone_mode=False, catch_exceptions=True
        )
        assert result.exit_code == 0
        assert "interrupted at 2/4" in result.output


class TestEventSummary:
    def _summary(self, event: Event) -> str:
        from agent_replay.viewer import TraceViewer

        return TraceViewer()._event_summary(event)

    def test_llm_request(self):
        e = Event(EventType.LLM_REQUEST, data={"model": "gpt-4o", "messages": [1, 2]})
        assert self._summary(e) == "model=gpt-4o messages=2"

    def test_llm_response_truncates(self):
        e = Event(EventType.LLM_RESPONSE, data={"content": "x" * 100})
        assert "..." in self._summary(e)

    def test_llm_response_tokens(self):
        e = Event(EventType.LLM_RESPONSE, data={"content": "hi", "tokens": 7})
        assert "(7 tokens)" in self._summary(e)

    def test_tool_call(self):
        e = Event(EventType.TOOL_CALL, data={"tool": "grep", "args": {"q": "a"}})
        assert self._summary(e) == "grep({'q': 'a'})"

    def test_tool_result_truncates(self):
        e = Event(EventType.TOOL_RESULT, data={"tool": "grep", "result": "y" * 80})
        summary = self._summary(e)
        assert summary.startswith("grep ->")
        assert "..." in summary

    def test_decision(self):
        e = Event(EventType.DECISION, data={"description": "route", "choice": "left"})
        assert self._summary(e) == "route -> left"

    def test_error(self):
        e = Event(EventType.ERROR, data={"message": "boom"})
        assert self._summary(e) == "boom"

    def test_log_fallback(self):
        e = Event(EventType.LOG, data={"message": "note"})
        assert self._summary(e) == "note"
