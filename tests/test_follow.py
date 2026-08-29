"""Tests for live trace following: TraceFollower and the follow CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from agent_replay import Event, EventType, FollowUpdate, Span, Trace, TraceFollower
from agent_replay.cli import cli
from agent_replay.follow import KIND_HEADER, KIND_MALFORMED, KIND_SPAN


def _header_line(name: str = "run", tags: list[str] | None = None) -> str:
    return json.dumps(
        {
            "type": "trace_header",
            "trace_id": "abc123",
            "name": name,
            "start_time": 100.0,
            "end_time": None,
            "metadata": {},
            "tags": tags or [],
        }
    )


def _span_line(name: str, events: list[Event] | None = None) -> str:
    span = Span(name=name, start_time=100.0, end_time=101.0)
    span.events = events or []
    return json.dumps({"type": "span", **span.to_dict()})


def _write_lines(path: Path, lines: list[str], append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode) as f:
        f.writelines(line + "\n" for line in lines)


class TestTraceFollowerParsing:
    def test_reads_existing_header_and_spans(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, [_header_line("demo"), _span_line("plan"), _span_line("act")])
        updates = TraceFollower(path).poll()
        assert [u.kind for u in updates] == [KIND_HEADER, KIND_SPAN, KIND_SPAN]
        assert updates[0].header["name"] == "demo"
        assert updates[1].span.name == "plan"
        assert updates[2].span.name == "act"

    def test_span_events_preserved(self, tmp_path):
        path = tmp_path / "t.jsonl"
        events = [
            Event(EventType.LLM_REQUEST, timestamp=100.0, data={"model": "gpt-4o"}),
            Event(EventType.TOOL_CALL, timestamp=100.5, data={"tool": "search"}),
        ]
        _write_lines(path, [_header_line(), _span_line("work", events)])
        updates = TraceFollower(path).poll()
        span = updates[1].span
        assert isinstance(span, Span)
        assert [e.event_type for e in span.events] == [
            EventType.LLM_REQUEST,
            EventType.TOOL_CALL,
        ]

    def test_incremental_appends(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, [_header_line()])
        follower = TraceFollower(path)
        first = follower.poll()
        assert [u.kind for u in first] == [KIND_HEADER]
        # nothing new yet
        assert follower.poll() == []
        _write_lines(path, [_span_line("later")], append=True)
        second = follower.poll()
        assert [u.kind for u in second] == [KIND_SPAN]
        assert second[0].span.name == "later"

    def test_partial_line_buffered_until_newline(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, [_header_line()])
        follower = TraceFollower(path)
        follower.poll()
        # write a span line without its trailing newline
        line = _span_line("half")
        with open(path, "a") as f:
            f.write(line[: len(line) // 2])
        assert follower.poll() == []  # incomplete, buffered
        with open(path, "a") as f:
            f.write(line[len(line) // 2 :] + "\n")
        updates = follower.poll()
        assert [u.kind for u in updates] == [KIND_SPAN]
        assert updates[0].span.name == "half"

    def test_from_end_skips_existing(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, [_header_line(), _span_line("old")])
        follower = TraceFollower(path, from_start=False)
        assert follower.poll() == []
        _write_lines(path, [_span_line("new")], append=True)
        updates = follower.poll()
        assert [u.kind for u in updates] == [KIND_SPAN]
        assert updates[0].span.name == "new"

    def test_malformed_line(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, ["{not json", _span_line("ok")])
        updates = TraceFollower(path).poll()
        assert updates[0].kind == KIND_MALFORMED
        assert updates[1].kind == KIND_SPAN

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "t.jsonl"
        _write_lines(path, [_header_line(), "", _span_line("a"), "  "])
        updates = TraceFollower(path).poll()
        assert [u.kind for u in updates] == [KIND_HEADER, KIND_SPAN]

    def test_missing_file_returns_empty(self, tmp_path):
        follower = TraceFollower(tmp_path / "nope.jsonl")
        assert follower.poll() == []

    def test_follow_update_dataclass(self):
        u = FollowUpdate(kind=KIND_MALFORMED, raw="x")
        assert u.kind == KIND_MALFORMED
        assert u.raw == "x"
        assert u.span is None


class TestFollowCLI:
    def _complete_trace(self, tmp_path: Path) -> Path:
        trace = Trace(name="cli-follow", tags=["prod"])
        span = trace.add_span("agent-loop")
        span.events = [
            Event(EventType.LLM_REQUEST, timestamp=100.0, data={"model": "gpt-4o"}),
            Event(EventType.LLM_RESPONSE, timestamp=100.5, data={"content": "hi", "tokens": 5}),
        ]
        trace.close()
        return trace.save(tmp_path / "trace.jsonl")

    def test_follows_then_times_out(self, tmp_path):
        path = self._complete_trace(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["follow", str(path), "--poll-interval", "0.01", "--timeout", "0.05"],
        )
        assert result.exit_code == 0
        assert "Following" in result.output
        assert "cli-follow" in result.output
        assert "agent-loop" in result.output
        assert "No new spans" in result.output

    def test_from_end_shows_no_existing_spans(self, tmp_path):
        path = self._complete_trace(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["follow", str(path), "--from-end", "--poll-interval", "0.01", "--timeout", "0.05"],
        )
        assert result.exit_code == 0
        assert "new spans only" in result.output
        assert "agent-loop" not in result.output

    def test_bad_poll_interval(self, tmp_path):
        path = self._complete_trace(tmp_path)
        result = CliRunner().invoke(cli, ["follow", str(path), "--poll-interval", "0"])
        assert result.exit_code != 0

    def test_missing_file(self):
        result = CliRunner().invoke(cli, ["follow", "/no/such/trace.jsonl"])
        assert result.exit_code != 0
