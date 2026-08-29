"""agent-replay: Record, replay, and debug AI agent execution traces."""

__version__ = "0.4.0"

from .cost import CostReport, LLMCall, analyze_trace, resolve_pricing
from .recorder import Recorder, record_trace
from .trace import Event, EventType, Span, Trace, normalize_tags
from .replay import PlaybackStep, ReplayEngine
from .diff import diff_traces, DiffResult, Divergence
from .diff_html import export_diff_html, render_diff_html
from .exporters import export_html, export_json, render_trace_html
from .follow import FollowUpdate, TraceFollower
from .otel import export_otlp, to_otlp
from .redact import redact_trace, BUILTIN_PATTERNS
from .server import (
    SearchMatch,
    TraceInfo,
    TraceServer,
    discover_traces,
    list_tags,
    search_directory,
    search_trace,
)

__all__ = [
    "Recorder",
    "record_trace",
    "Event",
    "EventType",
    "Span",
    "Trace",
    "PlaybackStep",
    "ReplayEngine",
    "diff_traces",
    "DiffResult",
    "Divergence",
    "export_diff_html",
    "render_diff_html",
    "export_html",
    "export_json",
    "render_trace_html",
    "FollowUpdate",
    "TraceFollower",
    "SearchMatch",
    "TraceInfo",
    "TraceServer",
    "discover_traces",
    "list_tags",
    "normalize_tags",
    "search_directory",
    "search_trace",
    "export_otlp",
    "to_otlp",
    "redact_trace",
    "BUILTIN_PATTERNS",
    "CostReport",
    "LLMCall",
    "analyze_trace",
    "resolve_pricing",
]
