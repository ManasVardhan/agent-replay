"""LlamaIndex integration: capture query and agent traces automatically.

Attach :class:`AgentReplayLlamaIndexHandler` to LlamaIndex's callback
manager and every query, retrieval, synthesis, sub-question, and agent
step becomes a span, while LLM calls, tool calls, embeddings, and errors
become events on the span they belong to.

Requires the optional ``llama-index-core`` package::

    pip install llama-index-core

Example
-------
>>> from llama_index.core import Settings
>>> from llama_index.core.callbacks import CallbackManager
>>> from agent_replay.integrations.llamaindex import AgentReplayLlamaIndexHandler
>>> handler = AgentReplayLlamaIndexHandler("my-rag-run")
>>> Settings.callback_manager = CallbackManager([handler])
>>> # ... run queries / agents ...
>>> handler.finish("trace.jsonl")  # close and save the trace
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent_replay.trace import EventType, Span, Trace

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
except ImportError as e:  # pragma: no cover - exercised via test reload
    raise ImportError(
        "AgentReplayLlamaIndexHandler requires the optional llama-index-core "
        "package. Install it with: pip install llama-index-core"
    ) from e

_TRUNCATE_AT = 500

# LlamaIndex event types that represent a nested unit of work. These become
# spans; everything else becomes a point-in-time event on its parent span.
_SPAN_EVENTS = {"query", "retrieve", "synthesize", "sub_question", "agent_step", "tree"}

# Event types summarized as log events when they complete.
_LOG_EVENTS = {"chunking", "node_parsing", "embedding", "templating", "reranking"}


def _safe(value: Any, limit: int = _TRUNCATE_AT) -> str:
    """Stringify any LlamaIndex payload defensively, truncating long values."""
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _event_name(event_type: Any) -> str:
    """Normalize a CBEventType (str enum) or plain string to its value."""
    value = getattr(event_type, "value", event_type)
    return str(value)


def _model_name(payload: dict[str, Any]) -> str:
    """Best-effort model name from an LLM event payload."""
    serialized = payload.get("serialized")
    if isinstance(serialized, dict):
        for key in ("model", "model_name"):
            if serialized.get(key):
                return str(serialized[key])
    if payload.get("model_name"):
        return str(payload["model_name"])
    return "llm"


def _messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract chat messages or the formatted prompt from an LLM payload."""
    messages = payload.get("messages")
    if isinstance(messages, list):
        return [
            {
                "role": str(getattr(m, "role", type(m).__name__)),
                "content": _safe(getattr(m, "content", m)),
            }
            for m in messages
        ]
    prompt = payload.get("formatted_prompt")
    if prompt is not None:
        return [{"role": "user", "content": _safe(prompt)}]
    return []


def _response_text(payload: dict[str, Any]) -> str:
    """Extract response text from an LLM end payload, tolerating odd shapes."""
    response = payload.get("response")
    if response is not None:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return _safe(content)
        return _safe(response)
    completion = payload.get("completion")
    if completion is not None:
        text = getattr(completion, "text", None)
        return _safe(text if text is not None else completion)
    return ""


def _token_usage(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage from an LLM end payload across response shapes."""
    for key in ("response", "completion"):
        raw = getattr(payload.get(key), "raw", None)
        usage: Any = None
        if isinstance(raw, dict):
            usage = raw.get("usage")
        elif raw is not None:
            usage = getattr(raw, "usage", None)
        if isinstance(usage, dict):
            return dict(usage)
        if usage is not None:
            fields = ("prompt_tokens", "completion_tokens", "total_tokens")
            found = {
                f: getattr(usage, f)
                for f in fields
                if isinstance(getattr(usage, f, None), int)
            }
            if found:
                return found
    return {}


def _tool_name(payload: dict[str, Any]) -> str:
    """Best-effort tool name from a function_call event payload."""
    tool = payload.get("tool")
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, str) and tool:
        return tool
    return "tool"


class AgentReplayLlamaIndexHandler(BaseCallbackHandler):
    """A LlamaIndex callback handler that records runs into a Trace.

    Query, retrieval, synthesis, sub-question, tree, and agent-step events
    become spans (nested via ``parent_id``); LLM requests/responses, tool
    calls/results, embeddings, and exceptions become events on the span
    they belong to. Events that arrive with no known parent land on a root
    session span, so nothing is ever dropped.

    Parameters
    ----------
    name : trace name
    metadata : optional metadata stored on the trace
    """

    def __init__(
        self, name: str = "llamaindex-run", metadata: dict[str, Any] | None = None
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.trace = Trace(name=name, metadata=metadata or {})
        self._spans: dict[str, Span] = {}
        self._tool_names: dict[str, str] = {}
        self._root: Span | None = None

    # -- span bookkeeping ----------------------------------------------------

    def _root_span(self) -> Span:
        if self._root is None:
            self._root = self.trace.add_span("session")
        return self._root

    def _parent_span(self, parent_id: str) -> Span:
        """Span that owns an event, given LlamaIndex's parent event id."""
        if parent_id and parent_id in self._spans:
            return self._spans[parent_id]
        return self._root_span()

    # -- trace lifecycle -----------------------------------------------------

    def start_trace(self, trace_id: str | None = None) -> None:
        """Called by LlamaIndex when a trace starts. The root span is lazy."""

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Called by LlamaIndex when a trace ends. Spans close individually."""

    # -- event lifecycle -----------------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())
        payload = payload or {}
        name = _event_name(event_type)

        if name in _SPAN_EVENTS:
            parent = self._spans.get(parent_id)
            span_meta: dict[str, Any] = {}
            if payload.get("query_str") is not None:
                span_meta["query"] = _safe(payload["query_str"])
            if payload.get("sub_question") is not None:
                span_meta["sub_question"] = _safe(payload["sub_question"])
            span = self.trace.add_span(
                name,
                parent_id=parent.span_id if parent else None,
                metadata=span_meta,
            )
            self._spans[event_id] = span
            return event_id

        span = self._parent_span(parent_id)
        self._spans[event_id] = span

        if name == "llm":
            span.add_event(
                EventType.LLM_REQUEST,
                {"model": _model_name(payload), "messages": _messages(payload)},
            )
        elif name == "function_call":
            tool = _tool_name(payload)
            self._tool_names[event_id] = tool
            span.add_event(
                EventType.TOOL_CALL,
                {"tool": tool, "args": {"input": _safe(payload.get("function_call"))}},
            )
        elif name == "exception":
            self._record_exception(span, payload)
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        payload = payload or {}
        name = _event_name(event_type)
        span = self._spans.get(event_id) or self._root_span()

        if "exception" in payload:
            self._record_exception(span, payload, source=name)

        if name in _SPAN_EVENTS:
            if payload.get("response") is not None:
                span.metadata["response"] = _response_text(payload)
            nodes = payload.get("nodes")
            if isinstance(nodes, list):
                span.metadata["nodes_retrieved"] = len(nodes)
            if span.end_time is None:
                span.close()
            return

        if name == "llm":
            data: dict[str, Any] = {"content": _response_text(payload)}
            usage = _token_usage(payload)
            if usage:
                data["token_usage"] = usage
                total = usage.get("total_tokens")
                if isinstance(total, int):
                    data["tokens"] = total
            span.add_event(EventType.LLM_RESPONSE, data)
        elif name == "function_call":
            tool = self._tool_names.pop(event_id, "tool")
            span.add_event(
                EventType.TOOL_RESULT,
                {"tool": tool, "result": _safe(payload.get("function_call_response"))},
            )
        elif name in _LOG_EVENTS:
            details: list[str] = []
            for key in ("chunks", "documents", "embeddings", "nodes"):
                value = payload.get(key)
                if isinstance(value, list):
                    details.append(f"{len(value)} {key}")
            message = name if not details else f"{name}: {', '.join(details)}"
            span.add_event(EventType.LOG, {"message": message, "level": "info"})

    # -- helpers -------------------------------------------------------------

    def _record_exception(
        self, span: Span, payload: dict[str, Any], source: str = "exception"
    ) -> None:
        error = payload.get("exception")
        span.add_event(
            EventType.ERROR,
            {
                "message": _safe(error),
                "exception": type(error).__name__
                if isinstance(error, BaseException)
                else "Exception",
                "source": source,
            },
        )

    # -- finalization --------------------------------------------------------

    def finish(self, path: str | Path | None = None) -> Trace:
        """Close all open spans and optionally save the trace as JSONL."""
        self.trace.close()
        if path is not None:
            self.trace.save(path)
        return self.trace
