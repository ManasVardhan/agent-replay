"""Tests for the CLI stats command."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.recorder import Recorder


def _make_trace_file() -> Path:
    """Create a trace file with varied events."""
    path = Path(tempfile.mktemp(suffix=".jsonl"))
    with Recorder("stats-test", output_path=path) as rec:
        with rec.span("planning"):
            rec.llm_request(model="gpt-4")
            rec.llm_response(content="Hi", tokens=5)
        with rec.span("execution"):
            rec.tool_call("search", {"q": "test"})
            rec.tool_result("search", {"result": "found"})
            rec.decision("next", choice="finish")
        with rec.span("logging"):
            rec.log("done")
            rec.error("oops")
    return path


class TestStatsCommand:
    def test_stats_basic(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "Stats:" in result.output
            assert "stats-test" in result.output
            assert "Spans:" in result.output or "spans" in result.output.lower()
            assert "Total events:" in result.output or "events" in result.output.lower()
        finally:
            path.unlink(missing_ok=True)

    def test_stats_shows_event_breakdown(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "Event breakdown" in result.output
            assert "llm_request" in result.output
            assert "tool_call" in result.output
        finally:
            path.unlink(missing_ok=True)

    def test_stats_shows_models_and_tools(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "gpt-4" in result.output
            assert "search" in result.output
        finally:
            path.unlink(missing_ok=True)

    def test_stats_shows_span_durations(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "Span durations" in result.output
            assert "planning" in result.output
            assert "execution" in result.output
        finally:
            path.unlink(missing_ok=True)

    def test_stats_json_output(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path), "--json-output"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["name"] == "stats-test"
            assert isinstance(data["event_type_counts"], dict)
            assert isinstance(data["span_durations"], dict)
            assert data["events"] == 7
            assert data["spans"] == 3
        finally:
            path.unlink(missing_ok=True)

    def test_stats_json_has_all_fields(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path), "--json-output"])
            data = json.loads(result.output)
            required_keys = {
                "name", "trace_id", "spans", "events", "duration",
                "event_type_counts", "span_durations", "total_tokens",
                "models_used", "tools_used",
            }
            assert required_keys.issubset(data.keys())
        finally:
            path.unlink(missing_ok=True)

    def test_stats_json_models_and_tools(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path), "--json-output"])
            data = json.loads(result.output)
            assert "gpt-4" in data["models_used"]
            assert "search" in data["tools_used"]
            assert data["total_tokens"] == 5
        finally:
            path.unlink(missing_ok=True)

    def test_stats_empty_trace(self) -> None:
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        with Recorder("empty-trace", output_path=path):
            pass  # no events
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "empty-trace" in result.output
            assert "Total events: 0" in result.output
        finally:
            path.unlink(missing_ok=True)

    def test_stats_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", "/tmp/nonexistent.jsonl"])
        assert result.exit_code != 0

    def test_stats_shows_token_count(self) -> None:
        path = _make_trace_file()
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", str(path)])
            assert result.exit_code == 0
            assert "tokens" in result.output.lower()
        finally:
            path.unlink(missing_ok=True)
