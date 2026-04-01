"""Tests for the stats and search CLI commands."""

from pathlib import Path

from click.testing import CliRunner

from agent_replay import Recorder
from agent_replay.cli import cli


def _make_trace(path: Path) -> Path:
    """Create a sample trace file with diverse events."""
    with Recorder("test-agent", output_path=path) as rec:
        with rec.span("planning"):
            rec.llm_request(model="gpt-4o", messages=[{"role": "user", "content": "Plan a trip"}])
            rec.llm_response(content="Here is a plan.", tokens=25)
            rec.decision("next step", choice="search")
        with rec.span("execution"):
            rec.tool_call("web_search", {"query": "flights to paris"})
            rec.tool_result("web_search", {"results": ["flight1", "flight2"]})
            rec.llm_request(model="claude-3-5-sonnet", messages=[{"role": "user", "content": "Pick best"}])
            rec.llm_response(content="Flight 1 is cheapest.", tokens=15)
            rec.tool_call("book_flight", {"flight": "flight1"})
            rec.tool_result("book_flight", {"confirmation": "ABC123"})
        with rec.span("error-handling"):
            rec.error("Payment failed", exception="TimeoutError")
            rec.log("Retrying payment", level="warning")
    return path


class TestStatsCommand:
    def test_stats_basic(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert result.exit_code == 0
        assert "Stats: test-agent" in result.output

    def test_stats_event_breakdown(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "llm_request" in result.output
        assert "llm_response" in result.output
        assert "tool_call" in result.output
        assert "tool_result" in result.output

    def test_stats_shows_token_count(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "Total LLM tokens: 40" in result.output

    def test_stats_shows_models(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "gpt-4o" in result.output
        assert "claude-3-5-sonnet" in result.output

    def test_stats_shows_tools(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "web_search" in result.output
        assert "book_flight" in result.output

    def test_stats_span_count(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "Spans:        3" in result.output

    def test_stats_total_events(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert "Total events: 11" in result.output

    def test_stats_empty_trace(self, tmp_path: Path) -> None:
        trace_file = tmp_path / "empty.jsonl"
        with Recorder("empty-agent", output_path=trace_file):
            pass
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert result.exit_code == 0
        assert "Total events: 0" in result.output

    def test_stats_no_tokens(self, tmp_path: Path) -> None:
        """When no LLM responses have token counts, token line should not appear."""
        trace_file = tmp_path / "no_tokens.jsonl"
        with Recorder("no-token-agent", output_path=trace_file) as rec:
            with rec.span("step"):
                rec.tool_call("test", {"a": 1})
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(trace_file)])
        assert result.exit_code == 0
        assert "Total LLM tokens" not in result.output


class TestSearchCommand:
    def test_search_finds_match(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "paris"])
        assert result.exit_code == 0
        assert "1 match" in result.output

    def test_search_finds_tool_name(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "web_search"])
        assert result.exit_code == 0
        assert "match" in result.output

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "PARIS"])
        assert result.exit_code == 0
        assert "match" in result.output

    def test_search_no_results(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "nonexistent_xyz"])
        assert result.exit_code == 0
        assert "No events matching" in result.output

    def test_search_multiple_matches(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        # "flight" appears in tool_call args and tool_result
        result = runner.invoke(cli, ["search", str(trace_file), "flight"])
        assert result.exit_code == 0
        # Should find multiple matches
        assert "match" in result.output

    def test_search_by_span_name(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "planning"])
        assert result.exit_code == 0
        assert "match" in result.output

    def test_search_error_events(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "Payment failed"])
        assert result.exit_code == 0
        assert "1 match" in result.output

    def test_search_model_name(self, tmp_path: Path) -> None:
        trace_file = _make_trace(tmp_path / "trace.jsonl")
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_file), "gpt-4o"])
        assert result.exit_code == 0
        assert "match" in result.output
