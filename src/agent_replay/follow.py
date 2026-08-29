"""Live trace following: tail a JSONL trace file as an agent writes it.

Traces are saved as JSONL, one line for the header and one line per span
(see :meth:`agent_replay.trace.Trace.save`). A writer that appends spans as
they close produces a growing file, and :class:`TraceFollower` streams those
appended lines as they land, so a long-running agent can be watched without
waiting for the whole run to finish.

The follower tracks a byte offset into the file and buffers any partial
trailing line, so a span that is still mid-write is not parsed until its
newline arrives. Each call to :meth:`TraceFollower.poll` returns the updates
that appeared since the previous call, as parsed :class:`FollowUpdate`
records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trace import Span

KIND_HEADER = "header"
KIND_SPAN = "span"
KIND_MALFORMED = "malformed"


@dataclass(slots=True)
class FollowUpdate:
    """One line that appeared in a followed trace file."""

    kind: str
    header: dict[str, Any] | None = None
    span: Span | None = None
    raw: str = ""


class TraceFollower:
    """Incrementally read appended lines from a JSONL trace file.

    Parameters
    ----------
    path : the trace file to follow.
    from_start : when True (default) the first ``poll`` returns every line
        already in the file and then follows new ones; when False the
        existing content is skipped and only lines appended after
        construction are returned.
    """

    def __init__(self, path: str | Path, *, from_start: bool = True) -> None:
        self.path = Path(path)
        self._offset = 0
        self._buffer = b""
        if not from_start:
            try:
                self._offset = self.path.stat().st_size
            except OSError:
                self._offset = 0

    def poll(self) -> list[FollowUpdate]:
        """Return updates that appeared since the previous poll.

        Reads any bytes appended since the last call, splits them into
        complete newline-terminated lines, and parses each. A partial
        trailing line (no newline yet) is buffered and parsed on a later
        poll once the rest arrives. Returns an empty list when nothing new
        is available or the file does not exist yet.
        """
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return []

        if not chunk:
            return []

        self._buffer += chunk
        updates: list[FollowUpdate] = []
        while b"\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition(b"\n")
            update = self._parse(line)
            if update is not None:
                updates.append(update)
        return updates

    @staticmethod
    def _parse(line: bytes) -> FollowUpdate | None:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            return FollowUpdate(kind=KIND_MALFORMED, raw=text)
        if not isinstance(record, dict):
            return FollowUpdate(kind=KIND_MALFORMED, raw=text)
        if record.get("type") == "trace_header":
            return FollowUpdate(kind=KIND_HEADER, header=record)
        if record.get("type") == "span":
            payload = {k: v for k, v in record.items() if k != "type"}
            try:
                span = Span.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                return FollowUpdate(kind=KIND_MALFORMED, raw=text)
            return FollowUpdate(kind=KIND_SPAN, span=span)
        return FollowUpdate(kind=KIND_MALFORMED, raw=text)
