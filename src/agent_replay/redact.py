"""Redact sensitive data from traces before sharing them.

Traces often contain API keys, tokens, and email addresses inside prompts,
tool arguments, and metadata. This module scrubs them so a trace file can be
attached to a bug report or shared publicly without leaking secrets.

Example:
    >>> from agent_replay.redact import redact_trace
    >>> clean, counts = redact_trace(trace)
    >>> clean.save("trace.redacted.jsonl")
"""

from __future__ import annotations

import re
from typing import Any

from .trace import Event, Span, Trace

# Ordered dict: more specific patterns first so labels are accurate.
BUILTIN_PATTERNS: dict[str, str] = {
    "anthropic_key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "openai_key": r"sk-[A-Za-z0-9_-]{20,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{36,}",
    "bearer_token": r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}",
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+\w",
}


def _compile_patterns(
    extra_patterns: dict[str, str] | None = None,
) -> dict[str, re.Pattern[str]]:
    """Compile builtin plus any extra patterns into regex objects.

    Extra patterns are checked before builtins so custom rules win.
    Raises re.error if a custom pattern is invalid.
    """
    merged: dict[str, str] = {}
    if extra_patterns:
        merged.update(extra_patterns)
    for name, pat in BUILTIN_PATTERNS.items():
        merged.setdefault(name, pat)
    return {name: re.compile(pat) for name, pat in merged.items()}


def redact_text(
    text: str,
    patterns: dict[str, re.Pattern[str]],
    placeholder: str = "[REDACTED:{label}]",
) -> tuple[str, dict[str, int]]:
    """Redact all pattern matches in a string.

    Returns the scrubbed string and a dict of match counts per label.
    """
    counts: dict[str, int] = {}
    for label, pattern in patterns.items():
        text, n = pattern.subn(placeholder.format(label=label), text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def _redact_value(
    value: Any,
    patterns: dict[str, re.Pattern[str]],
    placeholder: str,
    counts: dict[str, int],
) -> Any:
    """Recursively redact strings inside nested dicts, lists, and tuples."""
    if isinstance(value, str):
        new_value, found = redact_text(value, patterns, placeholder)
        for label, n in found.items():
            counts[label] = counts.get(label, 0) + n
        return new_value
    if isinstance(value, dict):
        return {
            k: _redact_value(v, patterns, placeholder, counts)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        redacted = [_redact_value(v, patterns, placeholder, counts) for v in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted
    return value


def redact_trace(
    trace: Trace,
    extra_patterns: dict[str, str] | None = None,
    placeholder: str = "[REDACTED:{label}]",
) -> tuple[Trace, dict[str, int]]:
    """Return a redacted copy of a trace plus per-label match counts.

    The original trace is not modified. String values in trace metadata,
    span metadata, and event data (including nested structures) are scrubbed
    using the builtin patterns merged with any extra_patterns.

    Parameters
    ----------
    trace : the Trace to scrub
    extra_patterns : optional mapping of label to regex source string
    placeholder : replacement template, may contain {label}

    Returns
    -------
    (redacted_trace, counts) where counts maps pattern label to hits.
    """
    patterns = _compile_patterns(extra_patterns)
    counts: dict[str, int] = {}

    new_spans: list[Span] = []
    for span in trace.spans:
        new_events = [
            Event(
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=_redact_value(event.data, patterns, placeholder, counts),
                event_id=event.event_id,
            )
            for event in span.events
        ]
        new_spans.append(
            Span(
                name=span.name,
                span_id=span.span_id,
                parent_id=span.parent_id,
                start_time=span.start_time,
                end_time=span.end_time,
                events=new_events,
                metadata=_redact_value(span.metadata, patterns, placeholder, counts),
            )
        )

    redacted = Trace(
        trace_id=trace.trace_id,
        name=trace.name,
        start_time=trace.start_time,
        end_time=trace.end_time,
        spans=new_spans,
        metadata=_redact_value(trace.metadata, patterns, placeholder, counts),
    )
    return redacted, counts
