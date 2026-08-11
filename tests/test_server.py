"""Tests for the trace server (agent-replay serve)."""

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
    discover_traces,
    render_cost_html,
    render_index_html,
)
from agent_replay.cost import analyze_trace
from agent_replay.trace import Trace


def _make_trace(name: str, path: Path, model: str = "gpt-4o-mini") -> Path:
    recorder = Recorder(name)
    with recorder.span("agent-loop"):
        recorder.llm_request(model=model, messages=[{"role": "user", "content": "hi"}])
        recorder.llm_response(
            content="<b>hello</b> & goodbye",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        recorder.tool_call("search", {"query": "weather"})
        recorder.tool_result("search", "sunny")
    trace = recorder.finish()
    trace.save(path)
    return path


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    directory.mkdir()
    _make_trace("run-alpha", directory / "alpha.jsonl")
    _make_trace("run-beta", directory / "beta.jsonl", model="claude-sonnet-4-5")
    return directory


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


class TestDiscoverTraces:
    def test_finds_trace_files_sorted(self, trace_dir: Path) -> None:
        infos = discover_traces(trace_dir)
        assert [i.file_name for i in infos] == ["alpha.jsonl", "beta.jsonl"]
        assert infos[0].name == "run-alpha"
        assert infos[0].spans == 1
        assert infos[0].events == 4
        assert infos[0].duration is not None

    def test_skips_non_trace_files(self, trace_dir: Path) -> None:
        (trace_dir / "notes.txt").write_text("not a trace")
        (trace_dir / "junk.jsonl").write_text('{"foo": "bar"}\n')
        (trace_dir / "broken.jsonl").write_text("{{{{\n")
        infos = discover_traces(trace_dir)
        assert [i.file_name for i in infos] == ["alpha.jsonl", "beta.jsonl"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_traces(tmp_path) == []

    def test_info_to_dict(self, trace_dir: Path) -> None:
        info = discover_traces(trace_dir)[0]
        d = info.to_dict()
        assert d["file"] == "alpha.jsonl"
        assert d["name"] == "run-alpha"
        assert d["spans"] == 1
        assert d["events"] == 4


class TestRenderers:
    def test_index_lists_traces_and_links(self, trace_dir: Path) -> None:
        html = render_index_html(discover_traces(trace_dir), trace_dir)
        assert "run-alpha" in html
        assert "run-beta" in html
        assert '/trace/alpha.jsonl' in html
        assert '/trace/alpha.jsonl/cost' in html
        assert "/diff?a=alpha.jsonl" in html

    def test_index_empty(self, tmp_path: Path) -> None:
        html = render_index_html([], tmp_path)
        assert "No trace files found" in html

    def test_index_escapes_names(self, tmp_path: Path) -> None:
        directory = tmp_path / "traces"
        directory.mkdir()
        _make_trace("<script>alert(1)</script>", directory / "evil.jsonl")
        html = render_index_html(discover_traces(directory), directory)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_cost_page_contains_models(self, trace_dir: Path) -> None:
        trace = Trace.load(trace_dir / "alpha.jsonl")
        report = analyze_trace(trace)
        html = render_cost_html(trace, report, "alpha.jsonl")
        assert "gpt-4o-mini" in html
        assert "By model" in html
        assert "By span" in html

    def test_cost_page_no_calls(self, tmp_path: Path) -> None:
        recorder = Recorder("empty-run")
        with recorder.span("s"):
            recorder.log("nothing here")
        trace = recorder.finish()
        html = render_cost_html(trace, analyze_trace(trace), "e.jsonl")
        assert "No LLM calls" in html


class TestServerRoutes:
    def test_index(self, server: TraceServer) -> None:
        status, body = _get(server, "/")
        assert status == 200
        assert "run-alpha" in body
        assert "run-beta" in body

    def test_api_traces(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/traces")
        assert status == 200
        data = json.loads(body)
        assert [t["file"] for t in data] == ["alpha.jsonl", "beta.jsonl"]
        assert data[0]["events"] == 4

    def test_timeline_view(self, server: TraceServer) -> None:
        status, body = _get(server, "/trace/alpha.jsonl")
        assert status == 200
        assert "run-alpha" in body
        assert "LLM REQUEST" in body

    def test_timeline_escapes_event_data(self, server: TraceServer) -> None:
        _, body = _get(server, "/trace/alpha.jsonl")
        assert "<b>hello</b>" not in body
        assert "&lt;b&gt;hello&lt;/b&gt;" in body

    def test_cost_view(self, server: TraceServer) -> None:
        status, body = _get(server, "/trace/alpha.jsonl/cost")
        assert status == 200
        assert "gpt-4o-mini" in body
        assert "$" in body

    def test_diff_view(self, server: TraceServer) -> None:
        status, body = _get(server, "/diff?a=alpha.jsonl&b=beta.jsonl")
        assert status == 200
        assert "alpha.jsonl vs beta.jsonl" in body

    def test_unknown_trace_404(self, server: TraceServer) -> None:
        status, body = _get_error(server, "/trace/nope.jsonl")
        assert status == 404
        assert "nope.jsonl" in body

    def test_unknown_route_404(self, server: TraceServer) -> None:
        status, _ = _get_error(server, "/whatever")
        assert status == 404

    def test_path_traversal_rejected(self, server: TraceServer, tmp_path: Path) -> None:
        secret = tmp_path / "secret.jsonl"
        _make_trace("secret-run", secret)
        status, body = _get_error(server, "/trace/..%2Fsecret.jsonl")
        assert status == 404
        assert "secret-run" not in body

    def test_diff_missing_params(self, server: TraceServer) -> None:
        status, _ = _get_error(server, "/diff?a=alpha.jsonl")
        assert status == 400

    def test_diff_unknown_trace(self, server: TraceServer) -> None:
        status, _ = _get_error(server, "/diff?a=alpha.jsonl&b=missing.jsonl")
        assert status == 404

    def test_new_trace_appears_without_restart(self, server: TraceServer) -> None:
        _make_trace("run-gamma", server.directory / "gamma.jsonl")
        _, body = _get(server, "/")
        assert "run-gamma" in body

    def test_pricing_override(self, trace_dir: Path) -> None:
        srv = TraceServer(trace_dir, port=0, pricing={"gpt-4o-mini": (100.0, 200.0)})
        srv.serve_in_background()
        try:
            _, body = _get(srv, "/trace/alpha.jsonl/cost")
            # 10 prompt tokens at $100/1M + 5 completion tokens at $200/1M
            assert "$0.002000" in body
        finally:
            srv.shutdown()
            srv.server_close()


class TestServerConstruction:
    def test_rejects_non_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.jsonl"
        file_path.write_text("")
        with pytest.raises(NotADirectoryError):
            TraceServer(file_path, port=0)

    def test_url_property(self, server: TraceServer) -> None:
        assert server.url.startswith("http://127.0.0.1:")


class TestServeCli:
    def test_serve_rejects_bad_price(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", str(trace_dir), "--price", "bad-spec"])
        assert result.exit_code != 0
        assert "MODEL=INPUT:OUTPUT" in result.output

    def test_serve_rejects_missing_directory(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", str(tmp_path / "missing")])
        assert result.exit_code != 0

    def test_serve_requires_directory_not_file(self, trace_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", str(trace_dir / "alpha.jsonl")])
        assert result.exit_code != 0

    def test_serve_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.output
