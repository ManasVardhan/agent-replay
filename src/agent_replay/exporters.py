"""Export traces to JSON and HTML timeline formats."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .trace import EventType, Trace

EVENT_COLORS_HTML: dict[EventType, str] = {
    EventType.LLM_REQUEST: "#06b6d4",
    EventType.LLM_RESPONSE: "#22c55e",
    EventType.TOOL_CALL: "#eab308",
    EventType.TOOL_RESULT: "#3b82f6",
    EventType.DECISION: "#a855f7",
    EventType.STATE_CHANGE: "#6b7280",
    EventType.ERROR: "#ef4444",
    EventType.LOG: "#9ca3af",
}


def export_json(trace: Trace, path: str | Path) -> Path:
    """Export trace as a single JSON file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2)
    return path


def export_html(trace: Trace, path: str | Path) -> Path:
    """Export trace as a self-contained HTML timeline."""
    path = Path(path)

    events_html = []
    for span in trace.spans:
        for event in span.events:
            color = EVENT_COLORS_HTML.get(event.event_type, "#6b7280")
            ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            label = event.event_type.value.replace("_", " ").upper()
            data_preview = json.dumps(event.data, indent=2, default=str)
            events_html.append(f"""
        <div class="event" style="border-left: 4px solid {color};">
            <div class="event-header">
                <span class="event-type" style="color: {color};">{label}</span>
                <span class="event-span">{span.name}</span>
                <span class="event-time">{ts}</span>
            </div>
            <pre class="event-data">{data_preview}</pre>
        </div>""")

    duration = f"{trace.duration:.3f}s" if trace.duration else "running"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent Trace: {trace.name}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
    h1 {{ color: #58a6ff; margin-bottom: 0.5rem; }}
    .meta {{ color: #8b949e; margin-bottom: 2rem; font-size: 0.9rem; }}
    .event {{ background: #161b22; border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }}
    .event-header {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 0.5rem; }}
    .event-type {{ font-weight: bold; font-size: 0.85rem; }}
    .event-span {{ color: #e3b341; font-size: 0.8rem; }}
    .event-time {{ color: #8b949e; font-size: 0.8rem; margin-left: auto; }}
    .event-data {{ color: #8b949e; font-size: 0.8rem; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
</style>
</head>
<body>
    <h1>🔍 {trace.name}</h1>
    <div class="meta">
        ID: {trace.trace_id} | Spans: {len(trace.spans)} | Events: {trace.event_count} | Duration: {duration}
    </div>
    {"".join(events_html)}
</body>
</html>"""

    path.write_text(html)
    return path
