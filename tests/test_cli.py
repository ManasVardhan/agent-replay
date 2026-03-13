"""Tests for the CLI interface."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.recorder import Recorder


def _create_trace(path: Path, name: str = "cli-test") -> Path:
    """Helper to create a trace file for CLI testing."""
    with Recorder(name) as rec:
        with rec.span("step-1"):
            rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
            rec.llm_response(content="Hello!", tokens=5)
        with rec.span("step-2"):
            rec.tool_call("search", {"query": "python"})
            rec.tool_result("search", {"results": ["docs.python.org"]})
            rec.decision("next action", choice="respond")
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / "trace.jsonl"
    rec.trace.save(trace_path)
    return trace_path


class TestCLIVersion:
    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.1" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Record, replay, and debug" in result.output


class TestCLIShow:
    def test_show_trace(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["show", str(trace_path)])
        assert result.exit_code == 0
        assert "cli-test" in result.output
        assert "LLM REQUEST" in result.output

    def test_show_tree(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["show", str(trace_path), "--tree"])
        assert result.exit_code == 0
        assert "cli-test" in result.output
        assert "step-1" in result.output

    def test_show_nonexistent(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["show", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0


class TestCLIInfo:
    def test_info(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(trace_path)])
        assert result.exit_code == 0
        assert "cli-test" in result.output
        assert "Spans:" in result.output
        assert "Events:" in result.output

    def test_info_shows_duration(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(trace_path)])
        assert result.exit_code == 0
        assert "Duration:" in result.output


class TestCLIDiff:
    def test_diff_identical(self, tmp_path: Path) -> None:
        path_a = _create_trace(tmp_path / "a", "trace-a")
        path_b = _create_trace(tmp_path / "b", "trace-b")
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(path_a), str(path_b)])
        assert result.exit_code == 0
        assert "Trace Diff" in result.output

    def test_diff_different(self, tmp_path: Path) -> None:
        # Trace A: uses search
        with Recorder("a") as r:
            with r.span("s"):
                r.tool_call("search", {})
        path_a = tmp_path / "a.jsonl"
        r.trace.save(path_a)

        # Trace B: uses browse
        with Recorder("b") as r:
            with r.span("s"):
                r.tool_call("browse", {})
        path_b = tmp_path / "b.jsonl"
        r.trace.save(path_b)

        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(path_a), str(path_b)])
        assert result.exit_code == 0
        assert "Divergence" in result.output or "divergence" in result.output


class TestCLIExport:
    def test_export_json(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        output = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["export", str(trace_path), "--format", "json", "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["name"] == "cli-test"

    def test_export_html(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        output = tmp_path / "out.html"
        runner = CliRunner()
        result = runner.invoke(cli, ["export", str(trace_path), "--format", "html", "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        html = output.read_text()
        assert "cli-test" in html

    def test_export_default_output(self, tmp_path: Path) -> None:
        trace_path = _create_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["export", str(trace_path), "--format", "json"])
        assert result.exit_code == 0
        default_output = trace_path.with_suffix(".json")
        assert default_output.exists()
