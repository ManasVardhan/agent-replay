"""Tests for cross-run trace search (directory search, CLI, and server routes)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.recorder import Recorder
from agent_replay.server import (
    TraceServer,
    render_search_html,
    search_directory,
    search_trace,
)
from agent_replay.trace import Trace


def _make_trace(name: str, path: Path, tool: str = "search", tool_arg: str = "weather") -> Path:
    recorder = Recorder(name)
    with recorder.span("agent-loop"):
        recorder.llm_request(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        recorder.llm_response(
            content="hello there",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        recorder.tool_call(tool, {"query": tool_arg})
        recorder.tool_result(tool, "ok")
    trace = recorder.finish()
    trace.save(path)
    return path


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    directory.mkdir()
    _make_trace("run-alpha", directory / "alpha.jsonl", tool="fetch_weather")
    _make_trace("run-beta", directory / "beta.jsonl", tool="send_email", tool_arg="bob")
    return directory


class TestSearchTrace:
    def test_finds_matches_with_file_label(self, trace_dir: Path) -> None:
        trace = Trace.load(trace_dir / "alpha.jsonl")
        matches = search_trace(trace, "fetch_weather", file_name="alpha.jsonl")
        assert len(matches) == 2  # tool_call + tool_result
        assert all(m.file_name == "alpha.jsonl" for m in matches)
        assert matches[0].trace_name == "run-alpha"
        assert matches[0].span_name == "agent-loop"

    def test_no_matches(self, trace_dir: Path) -> None:
        trace = Trace.load(trace_dir / "alpha.jsonl")
        assert search_trace(trace, "no-such-thing") == []

    def test_preview_truncated(self, tmp_path: Path) -> None:
        recorder = Recorder("long")
        with recorder.span("s"):
            recorder.tool_call("big", {"blob": "x" * 500})
        trace = recorder.finish()
        matches = search_trace(trace, "blob")
        assert matches[0].preview.endswith("...")
        assert len(matches[0].preview) <= 130

    def test_to_dict(self, trace_dir: Path) -> None:
        trace = Trace.load(trace_dir / "alpha.jsonl")
        d = search_trace(trace, "fetch_weather", file_name="alpha.jsonl")[0].to_dict()
        assert d["file"] == "alpha.jsonl"
        assert d["trace"] == "run-alpha"
        assert d["span"] == "agent-loop"
        assert d["event_type"] == "tool_call"
        assert "position" in d and "preview" in d


class TestSearchDirectory:
    def test_matches_across_files(self, trace_dir: Path) -> None:
        scanned, matches = search_directory(trace_dir, "tool_call")
        assert scanned == 2
        assert {m.file_name for m in matches} == {"alpha.jsonl", "beta.jsonl"}

    def test_query_specific_to_one_file(self, trace_dir: Path) -> None:
        scanned, matches = search_directory(trace_dir, "send_email")
        assert scanned == 2
        assert {m.file_name for m in matches} == {"beta.jsonl"}

    def test_skips_non_trace_files(self, trace_dir: Path) -> None:
        (trace_dir / "junk.jsonl").write_text('{"foo": "bar"}\n')
        (trace_dir / "broken.jsonl").write_text("{{{{\n")
        scanned, matches = search_directory(trace_dir, "tool_call")
        assert scanned == 2
        assert len(matches) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        scanned, matches = search_directory(tmp_path, "anything")
        assert scanned == 0
        assert matches == []

    def test_matches_ordered_by_file(self, trace_dir: Path) -> None:
        _, matches = search_directory(trace_dir, "tool")
        files = [m.file_name for m in matches]
        assert files == sorted(files)


class TestRenderSearchHtml:
    def test_renders_matches(self, trace_dir: Path) -> None:
        scanned, matches = search_directory(trace_dir, "fetch_weather")
        html = render_search_html("fetch_weather", matches, scanned, trace_dir)
        assert "Search: fetch_weather" in html
        assert "alpha.jsonl" in html
        assert "2 trace(s) scanned" in html

    def test_escapes_query(self, trace_dir: Path) -> None:
        html = render_search_html("<script>", [], 2, trace_dir)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_no_matches_message(self, trace_dir: Path) -> None:
        html = render_search_html("nope", [], 2, trace_dir)
        assert "No events matched" in html


class TestCliSearch:
    def test_single_file_still_works(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_dir / "alpha.jsonl"), "fetch_weather"])
        assert result.exit_code == 0, result.output
        assert "Found 2 match(es)" in result.output
        assert "agent-loop" in result.output

    def test_single_file_json_output(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["search", str(trace_dir / "alpha.jsonl"), "fetch_weather", "--json-output"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["query"] == "fetch_weather"
        assert len(payload["matches"]) == 2

    def test_directory_mode(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_dir), "tool_call"])
        assert result.exit_code == 0, result.output
        assert "across 2 trace(s)" in result.output
        assert "alpha.jsonl" in result.output
        assert "beta.jsonl" in result.output

    def test_directory_mode_json_output(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_dir), "send_email", "--json-output"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["traces_scanned"] == 2
        assert {m["file"] for m in payload["matches"]} == {"beta.jsonl"}

    def test_directory_mode_no_matches(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["search", str(trace_dir), "no-such-thing"])
        assert result.exit_code == 0, result.output
        assert "No events matching" in result.output


@pytest.fixture()
def server(trace_dir: Path):
    srv = TraceServer(trace_dir, port=0)
    srv.serve_in_background()
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(server: TraceServer, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(server.url.rstrip("/") + path) as resp:
        return resp.status, resp.read().decode("utf-8")


def _get_error(server: TraceServer, path: str) -> tuple[int, str]:
    try:
        return _get(server, path)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class TestServerSearch:
    def test_index_has_search_form(self, server: TraceServer) -> None:
        status, body = _get(server, "/")
        assert status == 200
        assert 'action="/search"' in body

    def test_search_page(self, server: TraceServer) -> None:
        status, body = _get(server, "/search?q=fetch_weather")
        assert status == 200
        assert "alpha.jsonl" in body
        assert "beta.jsonl" not in body.split("</h1>")[1]  # beta not in results table

    def test_search_page_no_matches(self, server: TraceServer) -> None:
        status, body = _get(server, "/search?q=zzz-nothing")
        assert status == 200
        assert "No events matched" in body

    def test_search_page_missing_query(self, server: TraceServer) -> None:
        status, body = _get_error(server, "/search")
        assert status == 400
        assert "?q=" in body

    def test_api_search(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/search?q=send_email")
        assert status == 200
        payload = json.loads(body)
        assert payload["query"] == "send_email"
        assert payload["traces_scanned"] == 2
        assert {m["file"] for m in payload["matches"]} == {"beta.jsonl"}

    def test_api_search_missing_query(self, server: TraceServer) -> None:
        status, body = _get_error(server, "/api/search")
        assert status == 400
        assert "error" in json.loads(body)
