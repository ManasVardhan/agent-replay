"""agent-replay: Record, replay, and debug AI agent execution traces."""

__version__ = "0.1.0"

from .recorder import Recorder, record_trace
from .trace import Event, EventType, Span, Trace
from .replay import ReplayEngine
from .diff import diff_traces, DiffResult, Divergence
from .exporters import export_html, export_json

__all__ = [
    "Recorder",
    "record_trace",
    "Event",
    "EventType",
    "Span",
    "Trace",
    "ReplayEngine",
    "diff_traces",
    "DiffResult",
    "Divergence",
    "export_html",
    "export_json",
]
