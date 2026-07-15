"""Export traces in OpenTelemetry OTLP/JSON format.

Produces a dict (or file) matching the OTLP ``ExportTraceServiceRequest``
JSON encoding, so traces can be sent to any OTEL-compatible backend
(Jaeger, Grafana Tempo, Honeycomb, etc.) or imported by OTEL tooling.
No OpenTelemetry SDK dependency is required.

Mapping notes:
- agent-replay trace ids (16 hex chars) are zero-padded to the 32 hex
  chars OTLP requires; span ids (12 hex chars) are padded to 16.
  Non-hex ids are deterministically hashed so links stay stable.
- Timestamps are converted from epoch seconds (float) to nanosecond
  strings, per the protobuf JSON mapping for fixed64.
- Span events become OTLP span events; event data dicts become typed
  attributes (nested structures are JSON-encoded strings).
- A span containing at least one ERROR event gets STATUS_CODE_ERROR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import __version__
from .trace import Event, EventType, Span, Trace

_HEX_DIGITS = set("0123456789abcdef")


def _normalize_id(raw: str, length: int) -> str:
    """Return a lowercase hex id of exactly ``length`` chars.

    Valid hex input is zero-padded (or truncated) to fit. Anything else
    is hashed with SHA-256 so the result is deterministic.
    """
    candidate = raw.lower()
    if candidate and set(candidate) <= _HEX_DIGITS:
        return candidate[-length:].zfill(length)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _nanos(timestamp: float) -> str:
    """Convert epoch seconds to a nanosecond string (OTLP fixed64)."""
    return str(int(round(timestamp * 1_000_000_000)))


def _any_value(value: Any) -> dict[str, Any]:
    """Encode a Python value as an OTLP AnyValue."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, default=str)}


def _attributes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode a dict as an OTLP attribute list."""
    return [{"key": str(k), "value": _any_value(v)} for k, v in data.items()]


def _event_to_otlp(event: Event) -> dict[str, Any]:
    attrs = _attributes(event.data)
    attrs.append({"key": "agent_replay.event_id", "value": {"stringValue": event.event_id}})
    return {
        "timeUnixNano": _nanos(event.timestamp),
        "name": event.event_type.value,
        "attributes": attrs,
    }


def _span_to_otlp(span: Span, trace_id: str) -> dict[str, Any]:
    end_time = span.end_time if span.end_time is not None else span.start_time
    otlp_span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": _normalize_id(span.span_id, 16),
        "name": span.name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": _nanos(span.start_time),
        "endTimeUnixNano": _nanos(end_time),
        "attributes": _attributes(span.metadata),
        "events": [_event_to_otlp(e) for e in span.events],
    }
    if span.parent_id:
        otlp_span["parentSpanId"] = _normalize_id(span.parent_id, 16)
    if any(e.event_type is EventType.ERROR for e in span.events):
        otlp_span["status"] = {"code": "STATUS_CODE_ERROR"}
    else:
        otlp_span["status"] = {"code": "STATUS_CODE_UNSET"}
    return otlp_span


def to_otlp(trace: Trace) -> dict[str, Any]:
    """Convert a Trace to an OTLP/JSON ExportTraceServiceRequest dict."""
    trace_id = _normalize_id(trace.trace_id, 32)
    resource_attrs = _attributes({"service.name": trace.name, **trace.metadata})
    resource_attrs.append(
        {"key": "agent_replay.trace_id", "value": {"stringValue": trace.trace_id}}
    )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {"name": "agent-replay", "version": __version__},
                        "spans": [_span_to_otlp(s, trace_id) for s in trace.spans],
                    }
                ],
            }
        ]
    }


def export_otlp(trace: Trace, path: str | Path) -> Path:
    """Export a trace as an OTLP/JSON file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(to_otlp(trace), f, indent=2)
    return path
