"""Tests for live trace streaming over server-sent events."""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest

from agent_replay.recorder import Recorder
from agent_replay.server import TraceServer, render_live_html
from agent_replay.trace import Span, Trace


def _make_trace(name: str, path: Path, spans: int = 1) -> Trace:
    recorder = Recorder(name)
    for i in range(spans):
        with recorder.span(f"step-{i}"):
            recorder.tool_call("search", {"query": f"q{i}"})
            recorder.tool_result("search", "ok")
    trace = recorder.finish()
    trace.save(path)
    return trace


def _append_span(path: Path, name: str) -> None:
    span = Span(name=name, start_time=1.0, end_time=2.0)
    with open(path, "a") as f:
        f.write(json.dumps({"type": "span", **span.to_dict()}) + "\n")


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    directory.mkdir()
    _make_trace("run-live", directory / "live.jsonl", spans=2)
    return directory


@pytest.fixture()
def server(trace_dir: Path):
    srv = TraceServer(trace_dir, port=0)
    srv.serve_in_background()
    yield srv
    srv.shutdown()
    srv.server_close()


def _read_sse(server: TraceServer, path: str, timeout: float = 5.0) -> tuple[int, str]:
    """Open an SSE endpoint and read the stream until the server closes it."""
    parsed = urlparse(server.url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


def _events(body: str) -> list[tuple[str, dict]]:
    """Parse SSE text into (event, data) pairs, skipping comments."""
    parsed = []
    event = None
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event is not None:
            parsed.append((event, json.loads(line.split(":", 1)[1].strip())))
            event = None
    return parsed


class TestSSEEndpoint:
    def test_streams_existing_spans_then_ends_on_idle_timeout(self, server):
        status, body = _read_sse(
            server, "/trace/live.jsonl/events?poll=0.05&timeout=0.2"
        )
        assert status == 200
        events = _events(body)
        kinds = [e for e, _ in events]
        assert kinds.count("header") == 1
        assert kinds.count("span") == 2
        assert kinds[-1] == "end"
        span_names = [d["name"] for e, d in events if e == "span"]
        assert span_names == ["step-0", "step-1"]

    def test_from_end_skips_existing_spans(self, server):
        status, body = _read_sse(
            server, "/trace/live.jsonl/events?from_end=1&poll=0.05&timeout=0.2"
        )
        assert status == 200
        events = _events(body)
        assert [e for e, _ in events] == ["end"]

    def test_streams_appended_spans(self, server, trace_dir):
        path = trace_dir / "live.jsonl"

        def append_later() -> None:
            time.sleep(0.15)
            _append_span(path, "late-span")

        writer = threading.Thread(target=append_later)
        writer.start()
        status, body = _read_sse(
            server, "/trace/live.jsonl/events?from_end=1&poll=0.05&timeout=0.5"
        )
        writer.join()
        assert status == 200
        events = _events(body)
        span_names = [d["name"] for e, d in events if e == "span"]
        assert span_names == ["late-span"]

    def test_malformed_lines_reported(self, server, trace_dir):
        path = trace_dir / "live.jsonl"

        def append_later() -> None:
            time.sleep(0.15)
            with open(path, "a") as f:
                f.write("{not json}\n")

        writer = threading.Thread(target=append_later)
        writer.start()
        status, body = _read_sse(
            server, "/trace/live.jsonl/events?from_end=1&poll=0.05&timeout=0.5"
        )
        writer.join()
        events = _events(body)
        assert ("malformed", {"raw": "{not json}"}) in events

    def test_unknown_trace_404(self, server):
        status, _ = _read_sse(server, "/trace/nope.jsonl/events?timeout=0.1")
        assert status == 404

    def test_traversal_blocked(self, server):
        status, _ = _read_sse(server, "/trace/..%2Fsecret.jsonl/events?timeout=0.1")
        assert status == 404

    def test_bad_params_400(self, server):
        for query in ["poll=abc", "timeout=-1", "poll=-0.5"]:
            status, _ = _read_sse(server, f"/trace/live.jsonl/events?{query}")
            assert status == 400, query

    def test_content_type_is_event_stream(self, server):
        parsed = urlparse(server.url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            conn.request("GET", "/trace/live.jsonl/events?poll=0.05&timeout=0.1")
            resp = conn.getresponse()
            assert resp.getheader("Content-Type", "").startswith("text/event-stream")
            resp.read()
        finally:
            conn.close()


class TestLivePage:
    def _get(self, server: TraceServer, path: str) -> tuple[int, str]:
        with urllib.request.urlopen(server.url.rstrip("/") + path) as resp:
            return resp.status, resp.read().decode("utf-8")

    def test_live_page_renders_existing_spans(self, server):
        status, body = self._get(server, "/trace/live.jsonl")
        assert status == 200
        status, body = self._get(server, "/trace/live.jsonl/live")
        assert status == 200
        assert "Live: run-live" in body
        assert "step-0" in body
        assert "step-1" in body
        assert "EventSource" in body
        assert "/trace/live.jsonl/events?from_end=1" in body

    def test_live_page_unknown_trace_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            self._get(server, "/trace/nope.jsonl/live")
        assert excinfo.value.code == 404

    def test_index_links_live_view(self, server):
        status, body = self._get(server, "/")
        assert status == 200
        assert "/trace/live.jsonl/live" in body
        assert ">live</a>" in body

    def test_render_live_html_escapes_names(self, tmp_path):
        trace = _make_trace("<b>sneaky</b>", tmp_path / "x.jsonl")
        html = render_live_html(trace, "x.jsonl")
        assert "<b>sneaky</b>" not in html
        assert "&lt;b&gt;sneaky&lt;/b&gt;" in html
