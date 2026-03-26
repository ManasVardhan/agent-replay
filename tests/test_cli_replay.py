"""Tests for the CLI interactive replay command and remaining CLI gaps."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.recorder import Recorder
from agent_replay.replay import ReplayEngine
from agent_replay.trace import EventType, Trace


def _create_trace_file(tmp_path: Path, name: str = "replay-test") -> Path:
    """Create a simple trace file with multiple events for replay testing."""
    with Recorder(name) as rec:
        with rec.span("planning"):
            rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            rec.llm_response(content="Hello there!", tokens=10)
        with rec.span("execution"):
            rec.tool_call("search", {"query": "python"})
            rec.tool_result("search", {"results": ["docs.python.org"]})
            rec.decision("next action", choice="respond")
    trace_path = tmp_path / "trace.jsonl"
    rec.trace.save(trace_path)
    return trace_path


class TestCLIReplayInteractive:
    """Test the interactive replay command with simulated stdin input."""

    def test_replay_quit(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="q\n")
        assert result.exit_code == 0
        assert "Replay:" in result.output
        assert "replay-test" in result.output

    def test_replay_exit_command(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="exit\n")
        assert result.exit_code == 0

    def test_replay_quit_full(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="quit\n")
        assert result.exit_code == 0

    def test_replay_next(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        # Step next twice then quit
        result = runner.invoke(cli, ["replay", str(trace_path)], input="n\nn\nq\n")
        assert result.exit_code == 0
        assert "planning" in result.output or "execution" in result.output

    def test_replay_next_full_word(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="next\nq\n")
        assert result.exit_code == 0

    def test_replay_prev(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        # Go forward then back
        result = runner.invoke(cli, ["replay", str(trace_path)], input="n\np\nq\n")
        assert result.exit_code == 0

    def test_replay_prev_full_word(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="n\nprev\nq\n")
        assert result.exit_code == 0

    def test_replay_back_command(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="n\nback\nq\n")
        assert result.exit_code == 0

    def test_replay_jump(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="j 3\nq\n")
        assert result.exit_code == 0

    def test_replay_jump_full_word(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="jump 2\nq\n")
        assert result.exit_code == 0

    def test_replay_jump_invalid(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="j abc\nq\n")
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_replay_unknown_command(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="xyzzy\nq\n")
        assert result.exit_code == 0
        assert "Unknown command" in result.output

    def test_replay_default_command_is_next(self, tmp_path: Path) -> None:
        """Pressing enter with no input should advance (default='n')."""
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="\nq\n")
        assert result.exit_code == 0

    def test_replay_eof_breaks_loop(self, tmp_path: Path) -> None:
        """EOF (empty input) should exit without an unhandled exception."""
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        # CliRunner with empty input triggers Abort via click.prompt
        result = runner.invoke(cli, ["replay", str(trace_path)], input="")
        # click.prompt aborts with exit_code 1 on EOF, but no traceback
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_replay_shows_event_count(self, tmp_path: Path) -> None:
        trace_path = _create_trace_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(trace_path)], input="q\n")
        assert "events" in result.output.lower()


class TestReplayCurrentSpanEvents:
    """Test the current_span_events method coverage."""

    def test_current_span_events_returns_matching(self) -> None:
        with Recorder("test") as rec:
            with rec.span("alpha"):
                rec.log("a1")
                rec.log("a2")
            with rec.span("beta"):
                rec.log("b1")
        engine = ReplayEngine(rec.trace)
        # At position 0, we're in alpha span
        events = engine.current_span_events()
        assert len(events) == 2
        assert all(s.name == "alpha" for s, e in events)

    def test_current_span_events_after_stepping(self) -> None:
        with Recorder("test") as rec:
            with rec.span("alpha"):
                rec.log("a1")
            with rec.span("beta"):
                rec.log("b1")
                rec.log("b2")
        engine = ReplayEngine(rec.trace)
        engine.step()  # Move past alpha's event
        events = engine.current_span_events()
        assert len(events) == 2
        assert all(s.name == "beta" for s, e in events)


class TestTraceFromDict:
    """Test Trace.from_dict coverage."""

    def test_trace_from_dict_basic(self) -> None:
        trace = Trace(name="roundtrip", metadata={"env": "test"})
        span = trace.add_span("s1")
        span.add_event(EventType.LOG, {"message": "hello"})
        span.close()
        trace.close()

        d = trace.to_dict()
        restored = Trace.from_dict(d)
        assert restored.name == "roundtrip"
        assert restored.trace_id == trace.trace_id
        assert restored.metadata == {"env": "test"}
        assert len(restored.spans) == 1
        assert restored.spans[0].name == "s1"
        assert len(restored.spans[0].events) == 1

    def test_trace_from_dict_unnamed(self) -> None:
        d = {
            "trace_id": "abc123",
            "start_time": 1000.0,
            "end_time": 1001.0,
            "spans": [],
            "metadata": {},
        }
        restored = Trace.from_dict(d)
        assert restored.name == "unnamed"


class TestMalformedTraceFiles:
    """Test handling of edge-case trace files."""

    def test_load_file_no_header(self, tmp_path: Path) -> None:
        """A JSONL file with span records but no header should still load."""
        path = tmp_path / "no_header.jsonl"
        span_data = {
            "type": "span",
            "name": "orphan",
            "span_id": "abc123",
            "parent_id": None,
            "start_time": 1000.0,
            "end_time": 1001.0,
            "events": [],
            "metadata": {},
        }
        import json
        path.write_text(json.dumps(span_data) + "\n")
        loaded = Trace.load(path)
        assert len(loaded.spans) == 1
        assert loaded.spans[0].name == "orphan"
        # Name defaults to "unnamed" since no header
        assert loaded.name == "unnamed"

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """An empty file should load as an empty trace."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loaded = Trace.load(path)
        assert len(loaded.spans) == 0

    def test_load_only_whitespace(self, tmp_path: Path) -> None:
        """A file with only whitespace should load as empty trace."""
        path = tmp_path / "whitespace.jsonl"
        path.write_text("   \n\n  \n")
        loaded = Trace.load(path)
        assert len(loaded.spans) == 0

    def test_load_malformed_json_skipped(self, tmp_path: Path, capsys: object) -> None:
        """Malformed JSON lines should be skipped, not crash."""
        import json as _json
        path = tmp_path / "bad.jsonl"
        header = {
            "type": "trace_header",
            "trace_id": "test123",
            "name": "bad-test",
            "start_time": 1000.0,
            "end_time": None,
            "metadata": {},
        }
        span = {
            "type": "span",
            "name": "good",
            "span_id": "s1",
            "parent_id": None,
            "start_time": 1000.0,
            "end_time": 1001.0,
            "events": [],
            "metadata": {},
        }
        lines = [
            _json.dumps(header),
            "this is not valid json {{{",
            _json.dumps(span),
            "another broken line!!!",
        ]
        path.write_text("\n".join(lines) + "\n")
        loaded = Trace.load(path)
        assert loaded.name == "bad-test"
        assert len(loaded.spans) == 1
        assert loaded.spans[0].name == "good"
