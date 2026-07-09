"""agent-replay: Record, replay, and debug AI agent execution traces."""

__version__ = "0.1.1"

from .recorder import Recorder, record_trace
from .trace import Event, EventType, Span, Trace
from .replay import PlaybackStep, ReplayEngine
from .diff import diff_traces, DiffResult, Divergence
from .exporters import export_html, export_json
from .redact import redact_trace, BUILTIN_PATTERNS

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
    "export_html",
    "export_json",
    "redact_trace",
    "BUILTIN_PATTERNS",
]
