"""Tests for exporters."""

import json
from pathlib import Path

from agent_replay.exporters import export_html, export_json
from agent_replay.recorder import Recorder


def _make_trace():
    with Recorder("export-test") as rec:
        with rec.span("step"):
            rec.llm_request(model="gpt-4")
            rec.llm_response(content="done", tokens=10)
            rec.tool_call("search", {"q": "test"})
    return rec.trace


def test_export_json(tmp_path: Path):
    trace = _make_trace()
    out = export_json(trace, tmp_path / "trace.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["name"] == "export-test"
    assert len(data["spans"]) == 1


def test_export_html(tmp_path: Path):
    trace = _make_trace()
    out = export_html(trace, tmp_path / "trace.html")
    assert out.exists()
    html = out.read_text()
    assert "export-test" in html
    assert "LLM REQUEST" in html
