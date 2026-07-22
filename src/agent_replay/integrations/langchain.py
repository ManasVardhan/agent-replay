"""LangChain integration: capture agent traces without manual instrumentation.

Attach :class:`AgentReplayCallbackHandler` to any LangChain runnable, chain,
agent, or LLM call and every chain run, LLM request/response, tool call, and
agent decision is recorded into an ``agent_replay.Trace`` automatically.

Requires the optional ``langchain-core`` package::

    pip install langchain-core

Example
-------
>>> from agent_replay.integrations.langchain import AgentReplayCallbackHandler
>>> handler = AgentReplayCallbackHandler("my-agent-run")
>>> chain.invoke({"question": "..."}, config={"callbacks": [handler]})
>>> handler.finish("trace.jsonl")  # close and save the trace
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_replay.trace import EventType, Span, Trace

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:  # pragma: no cover - exercised via test reload
    raise ImportError(
        "AgentReplayCallbackHandler requires the optional langchain-core package. "
        "Install it with: pip install langchain-core"
    ) from e

_TRUNCATE_AT = 500


def _safe(value: Any, limit: int = _TRUNCATE_AT) -> str:
    """Stringify any LangChain payload defensively, truncating long values."""
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _run_name(
    serialized: dict[str, Any] | None, kwargs: dict[str, Any], fallback: str
) -> str:
    """Best-effort human name for a run from LangChain's serialized payload."""
    if kwargs.get("name"):
        return str(kwargs["name"])
    if isinstance(serialized, dict):
        if serialized.get("name"):
            return str(serialized["name"])
        ident = serialized.get("id")
        if isinstance(ident, list) and ident:
            return str(ident[-1])
    return fallback


def _token_usage(response: Any) -> dict[str, Any]:
    """Extract token usage from an LLMResult across LangChain versions."""
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict):
            return dict(usage)
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            return dict(usage)
    except (AttributeError, IndexError, TypeError):
        pass
    return {}


def _generation_texts(response: Any) -> list[str]:
    """Extract generated texts from an LLMResult, tolerating odd shapes."""
    texts: list[str] = []
    for batch in getattr(response, "generations", []) or []:
        for gen in batch:
            text = getattr(gen, "text", None)
            if not text:
                message = getattr(gen, "message", None)
                text = getattr(message, "content", None)
            texts.append(_safe(text if text is not None else gen))
    return texts


class AgentReplayCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that records runs into a Trace.

    Chain and agent runs become spans (nested via ``parent_run_id``); LLM
    requests/responses, tool calls/results, agent decisions, and errors
    become events on the span they belong to. Runs that arrive with no
    known parent land on a root session span, so nothing is ever dropped.

    Parameters
    ----------
    name : trace name
    metadata : optional metadata stored on the trace
    """

    def __init__(
        self, name: str = "langchain-run", metadata: dict[str, Any] | None = None
    ) -> None:
        self.trace = Trace(name=name, metadata=metadata or {})
        self._spans: dict[str, Span] = {}
        self._tool_names: dict[str, str] = {}
        self._root: Span | None = None

    # -- span bookkeeping ----------------------------------------------------

    def _root_span(self) -> Span:
        if self._root is None:
            self._root = self.trace.add_span("session")
        return self._root

    def _resolve_span(self, run_id: Any, parent_run_id: Any) -> Span:
        """Find the span for an event and remember the run_id -> span link."""
        key = str(run_id) if run_id else None
        if key and key in self._spans:
            return self._spans[key]
        parent_key = str(parent_run_id) if parent_run_id else None
        span = self._spans.get(parent_key) if parent_key else None
        if span is None:
            span = self._root_span()
        if key:
            self._spans[key] = span
        return span

    # -- chain lifecycle -----------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        parent_key = str(parent_run_id) if parent_run_id else None
        parent = self._spans.get(parent_key) if parent_key else None
        span = self.trace.add_span(
            _run_name(serialized, kwargs, "chain"),
            parent_id=parent.span_id if parent else None,
            metadata={"inputs": _safe(inputs)},
        )
        if run_id:
            self._spans[str(run_id)] = span

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.metadata["outputs"] = _safe(outputs)
        if span.end_time is None:
            span.close()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(
            EventType.ERROR,
            {
                "message": _safe(error),
                "exception": type(error).__name__,
                "source": "chain",
            },
        )
        if span.end_time is None:
            span.close()

    # -- LLM lifecycle -------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(
            EventType.LLM_REQUEST,
            {
                "model": _run_name(serialized, kwargs, "llm"),
                "messages": [_safe(p) for p in prompts or []],
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        flat = [
            {
                "role": type(m).__name__,
                "content": _safe(getattr(m, "content", m)),
            }
            for batch in messages or []
            for m in batch
        ]
        span.add_event(
            EventType.LLM_REQUEST,
            {"model": _run_name(serialized, kwargs, "chat_model"), "messages": flat},
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        data: dict[str, Any] = {"content": "\n".join(_generation_texts(response))}
        usage = _token_usage(response)
        if usage:
            data["token_usage"] = usage
            total = usage.get("total_tokens")
            if isinstance(total, int):
                data["tokens"] = total
        span.add_event(EventType.LLM_RESPONSE, data)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(
            EventType.ERROR,
            {
                "message": _safe(error),
                "exception": type(error).__name__,
                "source": "llm",
            },
        )

    # -- tool lifecycle ------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        name = _run_name(serialized, kwargs, "tool")
        span.add_event(
            EventType.TOOL_CALL, {"tool": name, "args": {"input": _safe(input_str)}}
        )
        if run_id:
            self._tool_names[str(run_id)] = name

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        name = self._tool_names.pop(str(run_id), "tool") if run_id else "tool"
        span.add_event(EventType.TOOL_RESULT, {"tool": name, "result": _safe(output)})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        name = self._tool_names.pop(str(run_id), "tool") if run_id else "tool"
        span.add_event(
            EventType.ERROR,
            {
                "message": _safe(error),
                "exception": type(error).__name__,
                "source": f"tool:{name}",
            },
        )

    # -- agent decisions -----------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(
            EventType.DECISION,
            {
                "description": _safe(getattr(action, "log", action)),
                "choice": _safe(getattr(action, "tool", "")),
                "tool_input": _safe(getattr(action, "tool_input", "")),
            },
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(
            EventType.DECISION,
            {
                "description": _safe(getattr(finish, "log", finish)),
                "choice": "finish",
                "return_values": _safe(getattr(finish, "return_values", "")),
            },
        )

    # -- text / logs ---------------------------------------------------------

    def on_text(
        self, text: str, *, run_id: Any = None, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        span = self._resolve_span(run_id, parent_run_id)
        span.add_event(EventType.LOG, {"message": _safe(text), "level": "info"})

    # -- finalization --------------------------------------------------------

    def finish(self, path: str | Path | None = None) -> Trace:
        """Close all open spans and optionally save the trace as JSONL."""
        self.trace.close()
        if path is not None:
            self.trace.save(path)
        return self.trace
