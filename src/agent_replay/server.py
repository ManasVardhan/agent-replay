"""Local web server for browsing a directory of saved traces.

Serves an index of every trace in a directory plus the HTML timeline,
side-by-side diff, and cost views, all rendered on the fly with no
files written to disk. Built on the standard library only.
"""

from __future__ import annotations

import html as html_mod
import json
import threading
import time
from dataclasses import dataclass, field as dataclass_field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .cost import CostReport, analyze_trace
from .diff_html import render_diff_html
from .exporters import render_trace_html
from .follow import KIND_HEADER, KIND_MALFORMED, KIND_SPAN, TraceFollower
from .replay import ReplayEngine
from .trace import Trace

SSE_DEFAULT_POLL = 0.5
SSE_MIN_POLL = 0.05
SSE_MAX_POLL = 5.0

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
    form.search { margin-bottom: 1.5rem; }
    form.search input { background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
                        padding: 0.4rem 0.6rem; border-radius: 4px; width: 20rem;
                        font-family: inherit; font-size: 0.9rem; }
    form.search button { background: #21262d; border: 1px solid #30363d; color: #58a6ff;
                         padding: 0.4rem 0.8rem; border-radius: 4px; cursor: pointer;
                         font-family: inherit; font-size: 0.9rem; }
    td.preview { color: #8b949e; font-size: 0.85rem; }
    .tag { display: inline-block; background: #21262d; border: 1px solid #30363d;
           border-radius: 10px; padding: 0.05rem 0.5rem; font-size: 0.8rem;
           color: #8b949e; }
    a.tag:hover { text-decoration: none; border-color: #58a6ff; color: #58a6ff; }
    .tag.active { color: #58a6ff; border-color: #58a6ff; }
    .pill { display: inline-block; border-radius: 10px; padding: 0.05rem 0.6rem;
            font-size: 0.8rem; border: 1px solid #30363d; }
    .pill.live { color: #3fb950; border-color: #3fb950; }
    .pill.off { color: #e3b341; border-color: #e3b341; }
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
    tags: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_name,
            "name": self.name,
            "trace_id": self.trace_id,
            "spans": self.spans,
            "events": self.events,
            "duration": self.duration,
            "tags": self.tags,
        }


def discover_traces(directory: str | Path, *, tag: str | None = None) -> list[TraceInfo]:
    """Find loadable trace files directly inside a directory.

    Files that cannot be parsed as traces are skipped. Results are sorted
    by file name for a stable listing. Pass *tag* to keep only traces
    carrying that tag.
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
        if tag is not None and tag not in trace.tags:
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
                tags=trace.tags,
            )
        )
    return infos


def list_tags(directory: str | Path) -> list[str]:
    """Return every tag used by traces in a directory, sorted alphabetically."""
    tags: set[str] = set()
    for info in discover_traces(directory):
        tags.update(info.tags)
    return sorted(tags)


_PREVIEW_WIDTH = 120


@dataclass(slots=True)
class SearchMatch:
    """One event that matched a search query, with enough context to locate it."""

    file_name: str
    trace_name: str
    position: int
    span_name: str
    event_type: str
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_name,
            "trace": self.trace_name,
            "position": self.position,
            "span": self.span_name,
            "event_type": self.event_type,
            "preview": self.preview,
        }


def _match_preview(data: object) -> str:
    text = str(data)
    if len(text) > _PREVIEW_WIDTH:
        return text[:_PREVIEW_WIDTH] + "..."
    return text


def search_trace(trace: Trace, query: str, *, file_name: str = "") -> list[SearchMatch]:
    """Search one loaded trace, returning a SearchMatch per hit.

    Matching is the same case-insensitive substring search as
    ``ReplayEngine.search``: span name, event type, and event data are all
    searchable. *file_name* labels the matches when searching many files.
    """
    engine = ReplayEngine(trace)
    matches: list[SearchMatch] = []
    for pos in engine.search(query):
        pair = engine.jump(pos)
        if pair is None:
            continue
        span, event = pair
        matches.append(
            SearchMatch(
                file_name=file_name,
                trace_name=trace.name,
                position=pos,
                span_name=span.name,
                event_type=event.event_type.value,
                preview=_match_preview(event.data),
            )
        )
    return matches


def search_directory(
    directory: str | Path, query: str, *, tag: str | None = None
) -> tuple[int, list[SearchMatch]]:
    """Search every trace in a directory for *query*.

    Returns ``(traces_scanned, matches)``. Files that are not loadable
    traces are skipped, matching ``discover_traces``. Pass *tag* to scan
    only traces carrying that tag. Matches are ordered by file name, then
    event position.
    """
    infos = discover_traces(directory, tag=tag)
    matches: list[SearchMatch] = []
    for info in infos:
        try:
            trace = Trace.load(info.path)
        except Exception:
            continue
        matches.extend(search_trace(trace, query, file_name=info.file_name))
    return len(infos), matches


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html_mod.escape(title)}</title>\n<style>{_PAGE_STYLE}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def _render_tags(tags: list[str], active: str | None = None) -> str:
    """Render a trace's tags as filter links, highlighting the active one."""
    parts = []
    for t in tags:
        cls = "tag active" if t == active else "tag"
        parts.append(
            f"<a class=\"{cls}\" href=\"/?tag={quote(t)}\">{html_mod.escape(t)}</a>"
        )
    return " ".join(parts)


def render_index_html(
    infos: list[TraceInfo],
    directory: Path,
    tag: str | None = None,
    all_tags: list[str] | None = None,
) -> str:
    """Render the trace listing page, optionally filtered to one tag."""
    filter_note = ""
    if tag is not None:
        filter_note = (
            f"<div class=\"meta\">Filtered to tag "
            f"<span class=\"tag active\">{html_mod.escape(tag)}</span> | "
            "<a href=\"/\">clear filter</a></div>"
        )
    tag_bar = ""
    if all_tags:
        tag_bar = (
            "<div class=\"meta\">Tags: "
            + " ".join(
                f"<a class=\"tag{' active' if t == tag else ''}\" "
                f"href=\"/?tag={quote(t)}\">{html_mod.escape(t)}</a>"
                for t in all_tags
            )
            + "</div>"
        )

    if not infos:
        message = (
            f"No traces carry the tag {html_mod.escape(repr(tag))}."
            if tag is not None
            else "No trace files found in this directory."
        )
        body = (
            "<h1>agent-replay</h1>"
            f"<div class=\"meta\">Serving {html_mod.escape(str(directory))}</div>"
            f"{filter_note}{tag_bar}"
            f"<p class=\"empty\">{message}</p>"
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
            f"<td>{_render_tags(info.tags, active=tag)}</td>"
            f"<td class=\"num\">{info.spans}</td>"
            f"<td class=\"num\">{info.events}</td>"
            f"<td class=\"num\">{duration}</td>"
            "<td class=\"actions\">"
            f"<a href=\"/trace/{link}\">timeline</a>"
            f"<a href=\"/trace/{link}/live\">live</a>"
            f"<a href=\"/trace/{link}/cost\">cost</a>"
            f"{diff_links}"
            "</td>"
            "</tr>"
        )

    body = (
        "<h1>agent-replay</h1>"
        f"<div class=\"meta\">Serving {html_mod.escape(str(directory))} | "
        f"{len(infos)} trace(s)</div>"
        f"{filter_note}{tag_bar}"
        f"{_search_form(tag=tag)}"
        "<table><thead><tr><th>Trace</th><th>File</th><th>Tags</th><th>Spans</th>"
        "<th>Events</th><th>Duration</th><th>Views</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<div class=\"meta\">Diff any two traces via "
        "/diff?a=&lt;file&gt;&amp;b=&lt;file&gt;</div>"
    )
    return _page("agent-replay traces", body)


def _search_form(query: str = "", tag: str | None = None) -> str:
    hidden = ""
    if tag is not None:
        hidden = (
            f"<input type=\"hidden\" name=\"tag\" "
            f"value=\"{html_mod.escape(tag, quote=True)}\">"
        )
    return (
        "<form class=\"search\" action=\"/search\" method=\"get\">"
        f"<input type=\"text\" name=\"q\" placeholder=\"Search all traces\" "
        f"value=\"{html_mod.escape(query, quote=True)}\">"
        f"{hidden}"
        "<button type=\"submit\">Search</button></form>"
    )


def render_search_html(
    query: str,
    matches: list[SearchMatch],
    scanned: int,
    directory: Path,
    tag: str | None = None,
) -> str:
    """Render the cross-trace search results page."""
    tag_note = ""
    if tag is not None:
        tag_note = (
            f" | tag <span class=\"tag active\">{html_mod.escape(tag)}</span>"
        )
    body_parts = [
        "<a class=\"back\" href=\"/\">&larr; all traces</a>",
        f"<h1>Search: {html_mod.escape(query)}</h1>",
        f"<div class=\"meta\">Serving {html_mod.escape(str(directory))} | "
        f"{scanned} trace(s) scanned | {len(matches)} match(es){tag_note}</div>",
        _search_form(query, tag=tag),
    ]
    if not matches:
        body_parts.append("<p class=\"empty\">No events matched this query.</p>")
    else:
        rows = []
        for m in matches:
            link = quote(m.file_name)
            rows.append(
                "<tr>"
                f"<td><a href=\"/trace/{link}\">{html_mod.escape(m.file_name)}</a></td>"
                f"<td>{html_mod.escape(m.trace_name)}</td>"
                f"<td class=\"num\">{m.position + 1}</td>"
                f"<td>{html_mod.escape(m.span_name)}</td>"
                f"<td>{html_mod.escape(m.event_type)}</td>"
                f"<td class=\"preview\">{html_mod.escape(m.preview)}</td>"
                "</tr>"
            )
        body_parts.append(
            "<table><thead><tr><th>File</th><th>Trace</th><th>#</th><th>Span</th>"
            "<th>Event</th><th>Preview</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return _page(f"Search: {query}", "".join(body_parts))


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


_LIVE_SCRIPT = """
(function () {
  var tbody = document.getElementById("spans");
  var status = document.getElementById("status");
  var count = document.getElementById("count");
  var source = new EventSource(EVENTS_URL);
  function fmtDuration(s) {
    if (s.end_time == null) { return "running"; }
    return (s.end_time - s.start_time).toFixed(3) + "s";
  }
  source.addEventListener("span", function (e) {
    var s = JSON.parse(e.data);
    var tr = document.createElement("tr");
    var cells = [
      s.name,
      s.span_id,
      String((s.events || []).length),
      fmtDuration(s),
    ];
    for (var i = 0; i < cells.length; i++) {
      var td = document.createElement("td");
      if (i >= 2) { td.className = "num"; }
      td.textContent = cells[i];
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    count.textContent = String(parseInt(count.textContent, 10) + 1);
  });
  source.addEventListener("end", function () {
    status.textContent = "stream ended";
    status.className = "pill off";
    source.close();
  });
  source.onopen = function () {
    status.textContent = "live";
    status.className = "pill live";
  };
  source.onerror = function () {
    status.textContent = "reconnecting";
    status.className = "pill off";
  };
})();
"""


def render_live_html(trace: Trace, file_name: str) -> str:
    """Render the live-following page for one trace.

    Spans already in the file are rendered server-side; the page then
    subscribes to the trace's server-sent events stream and appends each
    new span as the writer saves it.
    """
    link = quote(file_name)
    rows = []
    for span in trace.spans:
        duration = f"{span.duration:.3f}s" if span.duration is not None else "running"
        rows.append(
            "<tr>"
            f"<td>{html_mod.escape(span.name)}</td>"
            f"<td>{html_mod.escape(span.span_id)}</td>"
            f"<td class=\"num\">{len(span.events)}</td>"
            f"<td class=\"num\">{duration}</td>"
            "</tr>"
        )
    script = (
        f"var EVENTS_URL = \"/trace/{link}/events?from_end=1\";" + _LIVE_SCRIPT
    )
    body = (
        "<a class=\"back\" href=\"/\">&larr; all traces</a>"
        f"<h1>Live: {html_mod.escape(trace.name)} "
        "<span id=\"status\" class=\"pill off\">connecting</span></h1>"
        f"<div class=\"meta\">{html_mod.escape(file_name)} | "
        f"<span id=\"count\">{len(trace.spans)}</span> span(s) | "
        f"new spans stream in as the agent writes them | "
        f"<a href=\"/trace/{link}\">static timeline</a></div>"
        "<table><thead><tr><th>Span</th><th>Id</th><th>Events</th>"
        "<th>Duration</th></tr></thead>"
        f"<tbody id=\"spans\">{''.join(rows)}</tbody></table>"
        f"<script>{script}</script>"
    )
    return _page(f"Live: {trace.name}", body)


class TraceRequestHandler(BaseHTTPRequestHandler):
    """Routes: /?tag=, /api/traces?tag=, /api/search?q=&tag=, /trace/<file>,
    /trace/<file>/cost, /trace/<file>/live, /trace/<file>/events (SSE),
    /diff?a=&b=, /search?q=&tag=."""

    server: TraceServer

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        segments = [s for s in parsed.path.split("/") if s]
        try:
            if not segments:
                self._serve_index(parse_qs(parsed.query))
            elif segments == ["api", "traces"]:
                tag = self._tag_param(parse_qs(parsed.query))
                payload = [
                    info.to_dict()
                    for info in discover_traces(self.server.directory, tag=tag)
                ]
                self._send_json(payload)
            elif segments == ["api", "search"]:
                self._serve_search_api(parse_qs(parsed.query))
            elif segments == ["search"]:
                self._serve_search(parse_qs(parsed.query))
            elif segments[0] == "trace" and len(segments) == 2:
                self._serve_trace_view(segments[1], view="timeline")
            elif segments[0] == "trace" and len(segments) == 3 and segments[2] == "cost":
                self._serve_trace_view(segments[1], view="cost")
            elif segments[0] == "trace" and len(segments) == 3 and segments[2] == "live":
                self._serve_trace_view(segments[1], view="live")
            elif segments[0] == "trace" and len(segments) == 3 and segments[2] == "events":
                self._serve_trace_events(segments[1], parse_qs(parsed.query))
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
        elif view == "live":
            self._send_html(render_live_html(trace, info.file_name))
        else:
            self._send_html(render_trace_html(trace))

    @staticmethod
    def _float_param(
        query: dict[str, list[str]], name: str, default: float
    ) -> float | None:
        """Parse a non-negative float query parameter, None when invalid."""
        raw = (query.get(name) or [""])[0].strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return None
        if value < 0:
            return None
        return value

    def _serve_trace_events(self, file_name: str, query: dict[str, list[str]]) -> None:
        """Stream trace updates as server-sent events.

        Emits ``header``, ``span``, and ``malformed`` events as the trace
        file grows, using the same tailing logic as the follow command.
        ``?from_end=1`` skips spans already in the file, ``?poll=`` sets
        the file check interval in seconds, and ``?timeout=`` closes the
        stream after that many idle seconds (0, the default, streams until
        the client disconnects).
        """
        info = self._lookup(file_name)
        if info is None:
            self._send_error_page(404, f"No trace named {file_name!r} in this directory.")
            return
        from_end = (query.get("from_end") or [""])[0].strip() in ("1", "true", "yes")
        poll = self._float_param(query, "poll", SSE_DEFAULT_POLL)
        timeout = self._float_param(query, "timeout", 0.0)
        if poll is None or timeout is None:
            self._send_error_page(400, "poll and timeout must be non-negative numbers.")
            return
        poll = min(max(poll, SSE_MIN_POLL), SSE_MAX_POLL)

        follower = TraceFollower(info.path, from_start=not from_end)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        idle = 0.0
        try:
            self.wfile.write(b": stream open\n\n")
            self.wfile.flush()
            while True:
                updates = follower.poll()
                if updates:
                    idle = 0.0
                    for update in updates:
                        payload: Any
                        if update.kind == KIND_HEADER:
                            payload = update.header
                        elif update.kind == KIND_SPAN and update.span is not None:
                            payload = update.span.to_dict()
                        elif update.kind == KIND_MALFORMED:
                            payload = {"raw": update.raw}
                        else:
                            continue
                        message = (
                            f"event: {update.kind}\n"
                            f"data: {json.dumps(payload)}\n\n"
                        )
                        self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
                    continue
                if timeout and idle >= timeout:
                    self.wfile.write(b"event: end\ndata: {\"reason\": \"idle timeout\"}\n\n")
                    self.wfile.flush()
                    return
                # Keepalive comment so idle streams are not dropped by proxies.
                if idle and idle % 15 < poll:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(poll)
                idle += poll
        except (BrokenPipeError, ConnectionResetError):
            return

    @staticmethod
    def _tag_param(query: dict[str, list[str]]) -> str | None:
        tag = (query.get("tag") or [""])[0].strip()
        return tag or None

    def _serve_index(self, query: dict[str, list[str]]) -> None:
        tag = self._tag_param(query)
        infos = discover_traces(self.server.directory, tag=tag)
        all_tags = list_tags(self.server.directory)
        self._send_html(
            render_index_html(infos, self.server.directory, tag=tag, all_tags=all_tags)
        )

    def _serve_search(self, query: dict[str, list[str]]) -> None:
        q = (query.get("q") or [""])[0].strip()
        if not q:
            self._send_error_page(400, "The search view needs ?q=<query>.")
            return
        tag = self._tag_param(query)
        scanned, matches = search_directory(self.server.directory, q, tag=tag)
        self._send_html(
            render_search_html(q, matches, scanned, self.server.directory, tag=tag)
        )

    def _serve_search_api(self, query: dict[str, list[str]]) -> None:
        q = (query.get("q") or [""])[0].strip()
        if not q:
            self._send_json({"error": "The search API needs ?q=<query>."}, status=400)
            return
        tag = self._tag_param(query)
        scanned, matches = search_directory(self.server.directory, q, tag=tag)
        payload: dict[str, Any] = {
            "query": q,
            "traces_scanned": scanned,
            "matches": [m.to_dict() for m in matches],
        }
        if tag is not None:
            payload["tag"] = tag
        self._send_json(payload)

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

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
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
