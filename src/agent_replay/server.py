"""Local web server for browsing a directory of saved traces.

Serves an index of every trace in a directory plus the HTML timeline,
side-by-side diff, and cost views, all rendered on the fly with no
files written to disk. Built on the standard library only.
"""

from __future__ import annotations

import html as html_mod
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .cost import CostReport, analyze_trace
from .diff_html import render_diff_html
from .exporters import render_trace_html
from .trace import Trace

TRACE_SUFFIXES = (".jsonl",)

_PAGE_STYLE = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117;
           color: #c9d1d9; padding: 2rem; }
    h1 { color: #58a6ff; margin-bottom: 0.5rem; }
    .meta { color: #8b949e; margin-bottom: 1.5rem; font-size: 0.9rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #21262d;
             font-size: 0.9rem; }
    th { color: #8b949e; font-weight: normal; }
    td.num { text-align: right; }
    a { color: #58a6ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .actions a { margin-right: 0.75rem; }
    .empty { color: #8b949e; }
    .error { color: #f85149; }
    .back { display: inline-block; margin-bottom: 1rem; color: #8b949e; }
    .est { color: #e3b341; }
"""


@dataclass
class TraceInfo:
    """Summary of one trace file found in the served directory."""

    file_name: str
    path: Path
    name: str
    trace_id: str
    spans: int
    events: int
    duration: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_name,
            "name": self.name,
            "trace_id": self.trace_id,
            "spans": self.spans,
            "events": self.events,
            "duration": self.duration,
        }


def discover_traces(directory: str | Path) -> list[TraceInfo]:
    """Find loadable trace files directly inside a directory.

    Files that cannot be parsed as traces are skipped. Results are sorted
    by file name for a stable listing.
    """
    directory = Path(directory)
    infos: list[TraceInfo] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TRACE_SUFFIXES:
            continue
        try:
            trace = Trace.load(path)
        except Exception:
            continue
        if not trace.spans and trace.start_time == 0:
            # No spans and no trace header: not an agent-replay trace file.
            continue
        infos.append(
            TraceInfo(
                file_name=path.name,
                path=path,
                name=trace.name,
                trace_id=trace.trace_id,
                spans=len(trace.spans),
                events=trace.event_count,
                duration=trace.duration,
            )
        )
    return infos


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html_mod.escape(title)}</title>\n<style>{_PAGE_STYLE}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def render_index_html(infos: list[TraceInfo], directory: Path) -> str:
    """Render the trace listing page."""
    if not infos:
        body = (
            "<h1>agent-replay</h1>"
            f"<div class=\"meta\">Serving {html_mod.escape(str(directory))}</div>"
            "<p class=\"empty\">No trace files found in this directory.</p>"
        )
        return _page("agent-replay traces", body)

    rows = []
    for info in infos:
        link = quote(info.file_name)
        duration = f"{info.duration:.3f}s" if info.duration is not None else "running"
        diff_links = ""
        others = [i for i in infos if i.file_name != info.file_name]
        if others:
            first = quote(others[0].file_name)
            diff_links = f"<a href=\"/diff?a={link}&amp;b={first}\">diff</a>"
        rows.append(
            "<tr>"
            f"<td><a href=\"/trace/{link}\">{html_mod.escape(info.name)}</a></td>"
            f"<td>{html_mod.escape(info.file_name)}</td>"
            f"<td class=\"num\">{info.spans}</td>"
            f"<td class=\"num\">{info.events}</td>"
            f"<td class=\"num\">{duration}</td>"
            "<td class=\"actions\">"
            f"<a href=\"/trace/{link}\">timeline</a>"
            f"<a href=\"/trace/{link}/cost\">cost</a>"
            f"{diff_links}"
            "</td>"
            "</tr>"
        )

    body = (
        "<h1>agent-replay</h1>"
        f"<div class=\"meta\">Serving {html_mod.escape(str(directory))} | "
        f"{len(infos)} trace(s)</div>"
        "<table><thead><tr><th>Trace</th><th>File</th><th>Spans</th>"
        "<th>Events</th><th>Duration</th><th>Views</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<div class=\"meta\">Diff any two traces via "
        "/diff?a=&lt;file&gt;&amp;b=&lt;file&gt;</div>"
    )
    return _page("agent-replay traces", body)


def render_cost_html(trace: Trace, report: CostReport, file_name: str) -> str:
    """Render the cost report page for one trace."""
    name = html_mod.escape(trace.name)
    body_parts = [
        "<a class=\"back\" href=\"/\">&larr; all traces</a>",
        f"<h1>Cost: {name}</h1>",
        f"<div class=\"meta\">{html_mod.escape(file_name)} | "
        f"{len(report.calls)} LLM call(s) | {report.total_tokens:,} tokens | "
        f"${report.total_cost_usd:.6f}</div>",
    ]

    if not report.calls:
        body_parts.append(
            "<p class=\"empty\">No LLM calls with token usage found in this trace.</p>"
        )
    else:
        model_rows = []
        for row in report.by_model():
            cost = f"${row['cost_usd']:.6f}" if row["cost_usd"] is not None else "n/a"
            if row["estimated"]:
                cost = f"<span class=\"est\">~{cost}</span>"
            model_rows.append(
                "<tr>"
                f"<td>{html_mod.escape(str(row['model']))}</td>"
                f"<td class=\"num\">{row['calls']}</td>"
                f"<td class=\"num\">{row['prompt_tokens']:,}</td>"
                f"<td class=\"num\">{row['completion_tokens']:,}</td>"
                f"<td class=\"num\">{row['total_tokens']:,}</td>"
                f"<td class=\"num\">{cost}</td>"
                "</tr>"
            )
        body_parts.append(
            "<h2>By model</h2><table><thead><tr><th>Model</th><th>Calls</th>"
            "<th>Prompt</th><th>Completion</th><th>Tokens</th><th>Cost</th>"
            f"</tr></thead><tbody>{''.join(model_rows)}</tbody></table>"
        )

        span_rows = []
        for row in report.by_span():
            cost = f"${row['cost_usd']:.6f}" if row["cost_usd"] is not None else "n/a"
            span_rows.append(
                "<tr>"
                f"<td>{html_mod.escape(str(row['span_name']))}</td>"
                f"<td class=\"num\">{row['calls']}</td>"
                f"<td class=\"num\">{row['total_tokens']:,}</td>"
                f"<td class=\"num\">{cost}</td>"
                "</tr>"
            )
        body_parts.append(
            "<h2>By span</h2><table><thead><tr><th>Span</th><th>Calls</th>"
            "<th>Tokens</th><th>Cost</th></tr></thead>"
            f"<tbody>{''.join(span_rows)}</tbody></table>"
        )

    notes = []
    if report.has_estimates:
        notes.append(
            "~ = estimated: only a token total was recorded, priced at the "
            "average of input and output rates."
        )
    if report.unpriced_models:
        notes.append("No pricing for: " + ", ".join(report.unpriced_models) + ".")
    if report.calls_without_usage:
        notes.append(
            f"{report.calls_without_usage} llm_response event(s) had no "
            f"token usage data."
        )
    for note in notes:
        body_parts.append(f"<div class=\"meta\">{html_mod.escape(note)}</div>")

    return _page(f"Cost: {trace.name}", "".join(body_parts))


class TraceRequestHandler(BaseHTTPRequestHandler):
    """Routes: /, /api/traces, /trace/<file>, /trace/<file>/cost, /diff?a=&b=."""

    server: TraceServer

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        segments = [s for s in parsed.path.split("/") if s]
        try:
            if not segments:
                self._send_html(render_index_html(self._infos(), self.server.directory))
            elif segments == ["api", "traces"]:
                payload = [info.to_dict() for info in self._infos()]
                self._send_json(payload)
            elif segments[0] == "trace" and len(segments) == 2:
                self._serve_trace_view(segments[1], view="timeline")
            elif segments[0] == "trace" and len(segments) == 3 and segments[2] == "cost":
                self._serve_trace_view(segments[1], view="cost")
            elif segments == ["diff"]:
                self._serve_diff(parse_qs(parsed.query))
            else:
                self._send_error_page(404, "Page not found.")
        except BrokenPipeError:
            pass

    def _infos(self) -> list[TraceInfo]:
        return discover_traces(self.server.directory)

    def _lookup(self, file_name: str) -> TraceInfo | None:
        """Resolve a file name against the current directory listing.

        Only exact matches for discovered files are served, so path
        traversal segments like .. can never reach outside the directory.
        """
        for info in self._infos():
            if info.file_name == file_name:
                return info
        return None

    def _serve_trace_view(self, file_name: str, view: str) -> None:
        info = self._lookup(file_name)
        if info is None:
            self._send_error_page(404, f"No trace named {file_name!r} in this directory.")
            return
        trace = Trace.load(info.path)
        if view == "cost":
            report = analyze_trace(trace, pricing=self.server.pricing or None)
            self._send_html(render_cost_html(trace, report, info.file_name))
        else:
            self._send_html(render_trace_html(trace))

    def _serve_diff(self, query: dict[str, list[str]]) -> None:
        name_a = (query.get("a") or [""])[0]
        name_b = (query.get("b") or [""])[0]
        if not name_a or not name_b:
            self._send_error_page(400, "The diff view needs ?a=<file>&b=<file>.")
            return
        info_a = self._lookup(name_a)
        info_b = self._lookup(name_b)
        if info_a is None or info_b is None:
            missing = name_a if info_a is None else name_b
            self._send_error_page(404, f"No trace named {missing!r} in this directory.")
            return
        trace_a = Trace.load(info_a.path)
        trace_b = Trace.load(info_b.path)
        title = f"{info_a.file_name} vs {info_b.file_name}"
        self._send_html(render_diff_html(trace_a, trace_b, title=title))

    def _send_html(self, content: str, status: int = 200) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_page(self, status: int, message: str) -> None:
        body = (
            "<a class=\"back\" href=\"/\">&larr; all traces</a>"
            f"<h1>{status}</h1><p class=\"error\">{html_mod.escape(message)}</p>"
        )
        self._send_html(_page(f"{status} - agent-replay", body), status=status)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            super().log_message(format, *args)


class TraceServer(ThreadingHTTPServer):
    """HTTP server that browses a directory of agent-replay traces.

    The directory is re-scanned on every request, so traces saved while
    the server is running show up on refresh.
    """

    daemon_threads = True

    def __init__(
        self,
        directory: str | Path,
        host: str = "127.0.0.1",
        port: int = 8600,
        pricing: dict[str, tuple[float, float]] | None = None,
        verbose: bool = False,
    ) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        self.directory = directory.resolve()
        self.pricing = pricing or {}
        self.verbose = verbose
        super().__init__((host, port), TraceRequestHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        if isinstance(host, bytes):
            host = host.decode("ascii")
        return f"http://{host}:{port}/"

    def serve_in_background(self) -> threading.Thread:
        """Start serve_forever in a daemon thread (used by tests and embedding)."""
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread
