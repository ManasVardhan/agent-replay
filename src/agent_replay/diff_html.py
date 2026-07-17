"""Side-by-side HTML comparison report for two traces.

Renders a self-contained HTML document (inline CSS, no JavaScript,
no external assets) that aligns the events of two traces column by
column and highlights every divergence found by :func:`diff_traces`.
"""

from __future__ import annotations

import html
from pathlib import Path

from .diff import DiffResult, diff_traces
from .trace import Event, EventType, Trace

_MAX_SUMMARY = 100

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial,
       sans-serif; margin: 0; background: #f6f8fa; color: #1f2328; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #59636e; font-size: 13px; margin-bottom: 16px; }
.summary { padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;
           font-size: 14px; border: 1px solid; }
.summary.same { background: #dafbe1; border-color: #1a7f37; }
.summary.diff { background: #ffebe9; border-color: #cf222e; }
.badges { margin-bottom: 20px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 12px; font-weight: 600; margin-right: 8px; color: #fff; }
.badge.critical { background: #cf222e; }
.badge.warning { background: #9a6700; }
.badge.info { background: #0969da; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #d1d9e0; border-radius: 8px; font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top;
         border-bottom: 1px solid #d1d9e0; }
th { background: #f6f8fa; font-size: 12px; text-transform: uppercase;
     letter-spacing: 0.03em; color: #59636e; }
td.pos { color: #59636e; width: 36px; }
td.cell { width: 46%; }
.etype { font-weight: 600; }
.span { color: #59636e; font-size: 11px; }
.data { color: #59636e; word-break: break-word; }
.missing { color: #9a6700; font-style: italic; }
tr.critical { background: #ffebe9; }
tr.warning { background: #fff8c5; }
tr.info { background: #ddf4ff; }
tr.critical td.cell { border-left: 3px solid #cf222e; }
.note { font-size: 12px; color: #59636e; margin-top: 4px; }
footer { margin-top: 24px; font-size: 12px; color: #59636e; }
"""


def _event_summary(event: Event) -> str:
    """Return a compact plain-text summary of an event's data."""
    data = event.data
    if event.event_type == EventType.LLM_REQUEST:
        return f"model={data.get('model', '')} messages={len(data.get('messages', []))}"
    if event.event_type == EventType.LLM_RESPONSE:
        content = str(data.get("content", ""))
        tokens = data.get("tokens")
        tok = f" ({tokens} tokens)" if tokens else ""
        return f'"{content}"{tok}'
    if event.event_type == EventType.TOOL_CALL:
        return f"{data.get('tool', '')}({data.get('args', {})})"
    if event.event_type == EventType.TOOL_RESULT:
        return f"{data.get('tool', '')} -> {data.get('result', '')}"
    if event.event_type == EventType.DECISION:
        return f"{data.get('description', '')} -> {data.get('choice', '')}"
    if event.event_type == EventType.ERROR:
        return str(data.get("message", ""))
    return str(data.get("message", data))


def _truncate(text: str) -> str:
    if len(text) > _MAX_SUMMARY:
        return text[:_MAX_SUMMARY] + "..."
    return text


def _span_names(trace: Trace) -> dict[str, str]:
    """Map event_id to the name of the span that owns it."""
    names: dict[str, str] = {}
    for span in trace.spans:
        for event in span.events:
            names[event.event_id] = span.name
    return names


def _event_cell(event: Event | None, span_name: str) -> str:
    if event is None:
        return '<td class="cell"><span class="missing">(no event)</span></td>'
    etype = html.escape(event.event_type.value)
    summary = html.escape(_truncate(_event_summary(event)))
    span_html = html.escape(span_name)
    return (
        f'<td class="cell"><span class="etype">{etype}</span> '
        f'<span class="span">[{span_html}]</span>'
        f'<div class="data">{summary}</div></td>'
    )


def render_diff_html(
    trace_a: Trace,
    trace_b: Trace,
    result: DiffResult | None = None,
    title: str = "Trace Comparison",
) -> str:
    """Render a side-by-side HTML comparison of two traces.

    Parameters
    ----------
    trace_a, trace_b : the traces to compare
    result : optional precomputed DiffResult; computed via diff_traces if omitted
    title : document title

    Returns
    -------
    A complete self-contained HTML document as a string.
    """
    if result is None:
        result = diff_traces(trace_a, trace_b)

    events_a = trace_a.all_events()
    events_b = trace_b.all_events()
    spans_a = _span_names(trace_a)
    spans_b = _span_names(trace_b)
    divergences = {d.position: d for d in result.divergences}
    max_len = max(len(events_a), len(events_b))

    n = len(result.divergences)
    n_critical = result.critical_count
    n_warning = sum(1 for d in result.divergences if d.severity == "warning")
    n_info = n - n_critical - n_warning

    summary_class = "same" if result.identical else "diff"
    badges = ""
    if not result.identical:
        parts = []
        if n_critical:
            parts.append(f'<span class="badge critical">{n_critical} critical</span>')
        if n_warning:
            parts.append(f'<span class="badge warning">{n_warning} warning</span>')
        if n_info:
            parts.append(f'<span class="badge info">{n_info} info</span>')
        badges = f'<div class="badges">{"".join(parts)}</div>'

    rows: list[str] = []
    for i in range(max_len):
        ea = events_a[i] if i < len(events_a) else None
        eb = events_b[i] if i < len(events_b) else None
        div = divergences.get(i)
        row_class = f' class="{html.escape(div.severity)}"' if div else ""
        note = ""
        if div:
            note = f'<div class="note">{html.escape(div.description)}</div>'
        cell_a = _event_cell(ea, spans_a.get(ea.event_id, "") if ea else "")
        cell_b = _event_cell(eb, spans_b.get(eb.event_id, "") if eb else "")
        rows.append(
            f"<tr{row_class}><td class=\"pos\">{i + 1}</td>"
            f"{cell_a}{cell_b}</tr>"
            + (f'<tr{row_class}><td></td><td colspan="2">{note}</td></tr>' if note else "")
        )

    if not rows:
        rows.append('<tr><td colspan="3"><span class="missing">Both traces are empty.</span></td></tr>')

    name_a = html.escape(trace_a.name)
    name_b = html.escape(trace_b.name)
    id_a = html.escape(trace_a.trace_id)
    id_b = html.escape(trace_b.trace_id)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="sub">A: {name_a} ({id_a}) &middot; B: {name_b} ({id_b})</div>
<div class="summary {summary_class}">{html.escape(result.summary)}</div>
{badges}
<table>
<thead><tr><th>#</th><th>Trace A: {name_a}</th><th>Trace B: {name_b}</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<footer>Generated by agent-replay.</footer>
</div>
</body>
</html>
"""


def export_diff_html(
    trace_a: Trace,
    trace_b: Trace,
    path: str | Path,
    result: DiffResult | None = None,
    title: str = "Trace Comparison",
) -> Path:
    """Write a side-by-side HTML comparison report to ``path``."""
    path = Path(path)
    path.write_text(render_diff_html(trace_a, trace_b, result=result, title=title))
    return path
