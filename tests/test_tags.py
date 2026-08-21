"""Tests for trace tagging and tag-based filtering."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.recorder import Recorder, record_trace
from agent_replay.server import (
    TraceServer,
    discover_traces,
    list_tags,
    render_index_html,
    search_directory,
)
from agent_replay.trace import Trace, normalize_tags


def _make_trace(name: str, path: Path, tags: list[str] | None = None, tool: str = "search") -> Path:
    recorder = Recorder(name, tags=tags)
    with recorder.span("agent-loop"):
        recorder.llm_request(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        recorder.llm_response(
            content="hello there",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        recorder.tool_call(tool, {"query": "weather"})
        recorder.tool_result(tool, "ok")
    trace = recorder.finish()
    trace.save(path)
    return path


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    directory.mkdir()
    _make_trace("run-prod", directory / "prod.jsonl", tags=["prod", "checkout"])
    _make_trace("run-stage", directory / "stage.jsonl", tags=["staging"], tool="send_email")
    _make_trace("run-plain", directory / "plain.jsonl")
    return directory


class TestNormalizeTags:
    def test_strips_dedupes_and_drops_empties(self) -> None:
        assert normalize_tags([" prod ", "prod", "", "  ", "checkout"]) == [
            "prod",
            "checkout",
        ]

    def test_none_and_empty(self) -> None:
        assert normalize_tags(None) == []
        assert normalize_tags([]) == []

    def test_preserves_order(self) -> None:
        assert normalize_tags(["b", "a", "b"]) == ["b", "a"]


class TestTraceTags:
    def test_trace_normalizes_tags_on_init(self) -> None:
        trace = Trace(name="t", tags=[" prod ", "prod", ""])
        assert trace.tags == ["prod"]

    def test_tags_roundtrip_through_save_and_load(self, tmp_path: Path) -> None:
        path = _make_trace("tagged", tmp_path / "t.jsonl", tags=["prod", "checkout"])
        loaded = Trace.load(path)
        assert loaded.tags == ["prod", "checkout"]

    def test_tags_in_to_dict_and_from_dict(self) -> None:
        trace = Trace(name="t", tags=["prod"])
        d = trace.to_dict()
        assert d["tags"] == ["prod"]
        assert Trace.from_dict(d).tags == ["prod"]

    def test_untagged_traces_load_with_empty_tags(self, tmp_path: Path) -> None:
        # Traces saved before tags existed have no tags key in the header.
        path = _make_trace("legacy", tmp_path / "legacy.jsonl")
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        header.pop("tags", None)
        lines[0] = json.dumps(header)
        path.write_text("\n".join(lines) + "\n")
        assert Trace.load(path).tags == []

    def test_from_dict_without_tags_key(self) -> None:
        trace = Trace(name="t")
        d = trace.to_dict()
        d.pop("tags")
        assert Trace.from_dict(d).tags == []


class TestRecorderTags:
    def test_recorder_passes_tags_to_trace(self) -> None:
        recorder = Recorder("run", tags=["prod", "checkout"])
        assert recorder.trace.tags == ["prod", "checkout"]

    def test_recorder_defaults_to_no_tags(self) -> None:
        assert Recorder("run").trace.tags == []

    def test_record_trace_decorator_tags(self, tmp_path: Path) -> None:
        out = tmp_path / "decorated.jsonl"

        @record_trace("decorated", output_path=out, tags=["batch"])
        def run(recorder: Recorder | None = None) -> None:
            assert recorder is not None
            recorder.log("hello")

        run()
        assert Trace.load(out).tags == ["batch"]


class TestDiscoverTracesTagFilter:
    def test_infos_include_tags(self, trace_dir: Path) -> None:
        infos = discover_traces(trace_dir)
        by_file = {i.file_name: i.tags for i in infos}
        assert by_file == {
            "prod.jsonl": ["prod", "checkout"],
            "stage.jsonl": ["staging"],
            "plain.jsonl": [],
        }

    def test_tag_filter(self, trace_dir: Path) -> None:
        infos = discover_traces(trace_dir, tag="prod")
        assert [i.file_name for i in infos] == ["prod.jsonl"]

    def test_tag_filter_no_matches(self, trace_dir: Path) -> None:
        assert discover_traces(trace_dir, tag="nope") == []

    def test_to_dict_includes_tags(self, trace_dir: Path) -> None:
        info = discover_traces(trace_dir, tag="staging")[0]
        assert info.to_dict()["tags"] == ["staging"]

    def test_list_tags(self, trace_dir: Path) -> None:
        assert list_tags(trace_dir) == ["checkout", "prod", "staging"]

    def test_list_tags_empty_directory(self, tmp_path: Path) -> None:
        assert list_tags(tmp_path) == []


class TestSearchDirectoryTagFilter:
    def test_tag_scopes_the_scan(self, trace_dir: Path) -> None:
        scanned, matches = search_directory(trace_dir, "send_email", tag="staging")
        assert scanned == 1
        assert {m.file_name for m in matches} == {"stage.jsonl"}

    def test_tag_excludes_matches_in_other_traces(self, trace_dir: Path) -> None:
        # Every trace has an llm_request, but only prod.jsonl carries the tag.
        scanned, matches = search_directory(trace_dir, "gpt-4o-mini", tag="prod")
        assert scanned == 1
        assert {m.file_name for m in matches} == {"prod.jsonl"}

    def test_without_tag_scans_everything(self, trace_dir: Path) -> None:
        scanned, _ = search_directory(trace_dir, "gpt-4o-mini")
        assert scanned == 3


class TestSearchCliTagFilter:
    def test_directory_search_with_tag(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(
            cli, ["search", str(trace_dir), "gpt-4o-mini", "--tag", "prod", "--json-output"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["tag"] == "prod"
        assert payload["traces_scanned"] == 1
        assert {m["file"] for m in payload["matches"]} == {"prod.jsonl"}

    def test_directory_search_tag_without_matches(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(
            cli, ["search", str(trace_dir), "gpt-4o-mini", "--tag", "nope"]
        )
        assert result.exit_code == 0
        assert "0 trace(s)" in result.output

    def test_file_search_with_matching_tag(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(
            cli,
            ["search", str(trace_dir / "prod.jsonl"), "gpt-4o-mini", "--tag", "prod"],
        )
        assert result.exit_code == 0
        assert "match" in result.output

    def test_file_search_with_non_matching_tag(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(
            cli,
            [
                "search",
                str(trace_dir / "prod.jsonl"),
                "gpt-4o-mini",
                "--tag",
                "staging",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["matches"] == []

    def test_info_shows_tags(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(cli, ["info", str(trace_dir / "prod.jsonl")])
        assert result.exit_code == 0
        assert "prod, checkout" in result.output

    def test_info_shows_none_for_untagged(self, trace_dir: Path) -> None:
        result = CliRunner().invoke(cli, ["info", str(trace_dir / "plain.jsonl")])
        assert result.exit_code == 0
        assert "(none)" in result.output


class TestIndexRendering:
    def test_index_lists_tags(self, trace_dir: Path) -> None:
        html = render_index_html(discover_traces(trace_dir), trace_dir)
        assert "?tag=prod" in html
        assert "?tag=staging" in html

    def test_index_filter_note(self, trace_dir: Path) -> None:
        infos = discover_traces(trace_dir, tag="prod")
        html = render_index_html(infos, trace_dir, tag="prod", all_tags=["prod"])
        assert "Filtered to tag" in html
        assert "clear filter" in html

    def test_index_empty_filter_message(self, trace_dir: Path) -> None:
        html = render_index_html([], trace_dir, tag="nope")
        assert "No traces carry the tag" in html

    def test_tag_html_is_escaped(self, tmp_path: Path) -> None:
        directory = tmp_path / "traces"
        directory.mkdir()
        _make_trace("x", directory / "x.jsonl", tags=["<script>"])
        html = render_index_html(discover_traces(directory), directory)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


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


class TestServerTagRoutes:
    def test_index_tag_filter(self, server: TraceServer) -> None:
        status, body = _get(server, "/?tag=prod")
        assert status == 200
        assert "prod.jsonl" in body
        assert "stage.jsonl" not in body
        assert "Filtered to tag" in body

    def test_api_traces_includes_tags(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/traces")
        assert status == 200
        payload = json.loads(body)
        tags = {entry["file"]: entry["tags"] for entry in payload}
        assert tags["prod.jsonl"] == ["prod", "checkout"]
        assert tags["plain.jsonl"] == []

    def test_api_traces_tag_filter(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/traces?tag=staging")
        assert status == 200
        payload = json.loads(body)
        assert [entry["file"] for entry in payload] == ["stage.jsonl"]

    def test_search_page_tag_filter(self, server: TraceServer) -> None:
        status, body = _get(server, "/search?q=gpt-4o-mini&tag=prod")
        assert status == 200
        assert "1 trace(s) scanned" in body
        assert "prod.jsonl" in body
        assert "stage.jsonl" not in body

    def test_api_search_tag_filter(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/search?q=gpt-4o-mini&tag=staging")
        assert status == 200
        payload = json.loads(body)
        assert payload["tag"] == "staging"
        assert payload["traces_scanned"] == 1
        assert {m["file"] for m in payload["matches"]} == {"stage.jsonl"}

    def test_api_search_without_tag_scans_all(self, server: TraceServer) -> None:
        status, body = _get(server, "/api/search?q=gpt-4o-mini")
        assert status == 200
        payload = json.loads(body)
        assert payload["traces_scanned"] == 3
        assert "tag" not in payload
