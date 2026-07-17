"""Tests for the side-by-side HTML comparison report."""

import json

from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.diff import diff_traces
from agent_replay.diff_html import export_diff_html, render_diff_html
from agent_replay.recorder import Recorder
from agent_replay.trace import Trace


def _make_trace(name, steps):
    """Build a trace from a list of (method, args, kwargs) recorder calls."""
    with Recorder(name) as r:
        with r.span("main"):
            for method, args, kwargs in steps:
                getattr(r, method)(*args, **kwargs)
    return r.trace


def _identical_pair():
    steps = [
        ("llm_request", (), {"model": "gpt-4"}),
        ("llm_response", (), {"content": "hello", "tokens": 5}),
        ("tool_call", ("search",), {"args": {"q": "python"}}),
    ]
    return _make_trace("run-a", steps), _make_trace("run-b", steps)


class TestRenderDiffHtml:
    def test_identical_traces(self):
        a, b = _identical_pair()
        out = render_diff_html(a, b)
        assert "<!DOCTYPE html>" in out
        assert "identical" in out
        assert 'class="summary same"' in out
        assert "badge" not in out.split("summary same")[1].split("<table")[0]

    def test_contains_trace_names_and_ids(self):
        a, b = _identical_pair()
        out = render_diff_html(a, b)
        assert "run-a" in out
        assert "run-b" in out
        assert a.trace_id in out
        assert b.trace_id in out

    def test_divergent_tool_call_highlighted(self):
        a = _make_trace("a", [("tool_call", ("search",), {})])
        b = _make_trace("b", [("tool_call", ("browse",), {})])
        out = render_diff_html(a, b)
        assert 'class="summary diff"' in out
        assert 'class="critical"' in out
        assert "Different tool called" in out
        assert "badge critical" in out

    def test_extra_event_marked_warning(self):
        a = _make_trace("a", [("llm_request", (), {"model": "gpt-4"})])
        b = _make_trace(
            "b",
            [
                ("llm_request", (), {"model": "gpt-4"}),
                ("llm_response", (), {"content": "extra"}),
            ],
        )
        out = render_diff_html(a, b)
        assert 'class="warning"' in out
        assert "(no event)" in out
        assert "badge warning" in out

    def test_llm_response_difference_marked_info(self):
        a = _make_trace("a", [("llm_response", (), {"content": "yes"})])
        b = _make_trace("b", [("llm_response", (), {"content": "no"})])
        out = render_diff_html(a, b)
        assert 'class="info"' in out
        assert "badge info" in out

    def test_event_summaries_present(self):
        a, b = _identical_pair()
        out = render_diff_html(a, b)
        assert "model=gpt-4" in out
        assert "hello" in out
        assert "(5 tokens)" in out
        assert "search" in out

    def test_span_names_shown(self):
        a, b = _identical_pair()
        out = render_diff_html(a, b)
        assert "[main]" in out

    def test_html_escaping_of_content(self):
        payload = "<script>alert('x')</script>"
        a = _make_trace("a", [("llm_response", (), {"content": payload})])
        b = _make_trace("b", [("llm_response", (), {"content": payload})])
        out = render_diff_html(a, b)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_html_escaping_of_trace_name(self):
        a = _make_trace("<img src=x onerror=alert(1)>", [("log", ("hi",), {})])
        b = _make_trace("b", [("log", ("hi",), {})])
        out = render_diff_html(a, b)
        assert "<img" not in out
        assert "&lt;img" in out

    def test_custom_title(self):
        a, b = _identical_pair()
        out = render_diff_html(a, b, title="My Regression Check")
        assert "<title>My Regression Check</title>" in out
        assert "<h1>My Regression Check</h1>" in out

    def test_empty_traces(self):
        out = render_diff_html(Trace(name="a"), Trace(name="b"))
        assert "Both traces are empty." in out
        assert 'class="summary same"' in out

    def test_long_summary_truncated(self):
        a = _make_trace("a", [("llm_response", (), {"content": "x" * 500})])
        b = _make_trace("b", [("llm_response", (), {"content": "x" * 500})])
        out = render_diff_html(a, b)
        assert "x" * 99 + "..." in out
        assert "x" * 200 not in out

    def test_accepts_precomputed_result(self):
        a, b = _identical_pair()
        result = diff_traces(a, b)
        out = render_diff_html(a, b, result=result)
        assert result.summary in out

    def test_no_javascript_or_external_assets(self):
        a = _make_trace("a", [("tool_call", ("search",), {})])
        b = _make_trace("b", [("tool_call", ("browse",), {})])
        out = render_diff_html(a, b)
        assert "<script" not in out
        assert "src=" not in out
        assert "href=" not in out


class TestExportDiffHtml:
    def test_writes_file(self, tmp_path):
        a, b = _identical_pair()
        target = tmp_path / "report.html"
        written = export_diff_html(a, b, target)
        assert written == target
        assert target.exists()
        assert "<!DOCTYPE html>" in target.read_text()

    def test_accepts_str_path(self, tmp_path):
        a, b = _identical_pair()
        target = tmp_path / "report.html"
        written = export_diff_html(a, b, str(target))
        assert written == target
        assert target.exists()


class TestDiffCliHtml:
    def _save_pair(self, tmp_path):
        a = _make_trace("a", [("tool_call", ("search",), {})])
        b = _make_trace("b", [("tool_call", ("browse",), {})])
        pa = tmp_path / "a.jsonl"
        pb = tmp_path / "b.jsonl"
        a.save(pa)
        b.save(pb)
        return pa, pb

    def test_diff_html_option_writes_report(self, tmp_path):
        pa, pb = self._save_pair(tmp_path)
        report = tmp_path / "cmp.html"
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(pa), str(pb), "--html", str(report)])
        assert result.exit_code == 0
        assert report.exists()
        content = report.read_text()
        assert "Different tool called" in content
        assert "HTML comparison written to" in result.output

    def test_diff_html_title_option(self, tmp_path):
        pa, pb = self._save_pair(tmp_path)
        report = tmp_path / "cmp.html"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["diff", str(pa), str(pb), "--html", str(report), "--title", "Nightly Check"],
        )
        assert result.exit_code == 0
        assert "<h1>Nightly Check</h1>" in report.read_text()

    def test_diff_json_output(self, tmp_path):
        pa, pb = self._save_pair(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(pa), str(pb), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["identical"] is False
        assert data["divergence_count"] >= 1
        assert data["divergences"][0]["severity"] == "critical"

    def test_diff_json_output_identical(self, tmp_path):
        a, b = _identical_pair()
        pa = tmp_path / "a.jsonl"
        pb = tmp_path / "b.jsonl"
        a.save(pa)
        b.save(pb)
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(pa), str(pb), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["identical"] is True
        assert data["divergences"] == []

    def test_diff_json_output_long_lines_stay_valid_json(self, tmp_path):
        # Regression test: rich console soft-wrapped long lines, injecting
        # newlines inside JSON strings and corrupting piped output.
        long_text = "Lisbon has about 548,703 residents per the 2021 census. " * 5
        a = _make_trace("a", [("llm_response", (), {"content": long_text})])
        b = _make_trace("b", [("llm_response", (), {"content": long_text + " more"})])
        pa = tmp_path / "a.jsonl"
        pb = tmp_path / "b.jsonl"
        a.save(pa)
        b.save(pb)
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(pa), str(pb), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["divergence_count"] == 1

    def test_diff_without_html_unchanged(self, tmp_path):
        pa, pb = self._save_pair(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", str(pa), str(pb)])
        assert result.exit_code == 0
        assert "Trace Diff" in result.output
        assert "HTML comparison" not in result.output

    def test_public_api_exports(self):
        import agent_replay

        assert hasattr(agent_replay, "render_diff_html")
        assert hasattr(agent_replay, "export_diff_html")
