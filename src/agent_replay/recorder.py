"""Recorder: decorator and context manager to capture agent execution traces."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

from .trace import EventType, Span, Trace


class Recorder:
    """Records agent execution into a Trace.

    Use as a context manager or call methods directly.

    Example::

        with Recorder("my-agent-run") as rec:
            rec.llm_request(model="gpt-4", messages=[...])
            rec.llm_response(content="Hello!", tokens=42)
            rec.tool_call("search", {"query": "python"})
            rec.tool_result("search", {"results": [...]})

        trace = rec.trace
        trace.save("trace.jsonl")
    """

    def __init__(
        self,
        name: str = "agent-run",
        metadata: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.trace = Trace(name=name, metadata=metadata or {})
        self.output_path = Path(output_path) if output_path else None
        self._span_stack: list[Span] = []
        self._current_span: Span | None = None

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()

    def finish(self) -> Trace:
        """Finalize the trace, closing open spans and optionally saving."""
        self.trace.close()
        if self.output_path:
            self.trace.save(self.output_path)
        return self.trace

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Generator[Span, None, None]:
        """Open a named span (context manager). Spans can nest."""
        parent_id = self._current_span.span_id if self._current_span else None
        s = self.trace.add_span(name, parent_id=parent_id, metadata=metadata or {})
        self._span_stack.append(s)
        prev = self._current_span
        self._current_span = s
        try:
            yield s
        finally:
            s.close()
            self._span_stack.pop()
            self._current_span = prev

    def _ensure_span(self) -> Span:
        if self._current_span is None:
            s = self.trace.add_span("default")
            self._span_stack.append(s)
            self._current_span = s
        return self._current_span

    def event(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Record a raw event in the current span."""
        self._ensure_span().add_event(event_type, data)

    def llm_request(self, model: str = "", messages: list[Any] | None = None, **kwargs: Any) -> None:
        self.event(EventType.LLM_REQUEST, {"model": model, "messages": messages or [], **kwargs})

    def llm_response(self, content: str = "", tokens: int | None = None, **kwargs: Any) -> None:
        self.event(EventType.LLM_RESPONSE, {"content": content, "tokens": tokens, **kwargs})

    def tool_call(self, tool: str, args: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.event(EventType.TOOL_CALL, {"tool": tool, "args": args or {}, **kwargs})

    def tool_result(self, tool: str, result: Any = None, **kwargs: Any) -> None:
        self.event(EventType.TOOL_RESULT, {"tool": tool, "result": result, **kwargs})

    def decision(self, description: str, choice: str = "", **kwargs: Any) -> None:
        self.event(EventType.DECISION, {"description": description, "choice": choice, **kwargs})

    def state_change(self, key: str, old: Any = None, new: Any = None, **kwargs: Any) -> None:
        self.event(EventType.STATE_CHANGE, {"key": key, "old": old, "new": new, **kwargs})

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        self.event(EventType.LOG, {"message": message, "level": level, **kwargs})

    def error(self, message: str, exception: str | None = None, **kwargs: Any) -> None:
        self.event(EventType.ERROR, {"message": message, "exception": exception, **kwargs})


def record_trace(
    name: str = "agent-run",
    output_path: str | Path | None = None,
) -> Callable:
    """Decorator that wraps a function with a Recorder.

    The decorated function receives a ``recorder`` keyword argument.

    Example::

        @record_trace("my-agent", output_path="trace.jsonl")
        def run_agent(task: str, recorder: Recorder | None = None):
            recorder.llm_request(model="gpt-4", messages=[{"role": "user", "content": task}])
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Recorder(name=name, output_path=output_path) as rec:
                kwargs["recorder"] = rec
                return fn(*args, **kwargs)
        return wrapper
    return decorator
