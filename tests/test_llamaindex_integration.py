"""Tests for the LlamaIndex callback handler integration.

llama-index-core is an optional dependency; when it is not installed, a
minimal stub providing BaseCallbackHandler is injected so the handler's
recording logic is still fully tested.
"""

from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_replay import EventType, Trace

try:
    import llama_index.core.callbacks.base_handler  # noqa: F401

    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    pkg = types.ModuleType("llama_index")
    core_mod = types.ModuleType("llama_index.core")
    callbacks_mod = types.ModuleType("llama_index.core.callbacks")
    base_handler_mod = types.ModuleType("llama_index.core.callbacks.base_handler")

    class BaseCallbackHandler:
        def __init__(
            self,
            event_starts_to_ignore: Any = (),
            event_ends_to_ignore: Any = (),
        ) -> None:
            self.event_starts_to_ignore = list(event_starts_to_ignore)
            self.event_ends_to_ignore = list(event_ends_to_ignore)

    base_handler_mod.BaseCallbackHandler = BaseCallbackHandler
    callbacks_mod.base_handler = base_handler_mod
    core_mod.callbacks = callbacks_mod
    pkg.core = core_mod
    sys.modules["llama_index"] = pkg
    sys.modules["llama_index.core"] = core_mod
    sys.modules["llama_index.core.callbacks"] = callbacks_mod
    sys.modules["llama_index.core.callbacks.base_handler"] = base_handler_mod

from agent_replay.integrations.llamaindex import (  # noqa: E402
    AgentReplayLlamaIndexHandler,
    _event_name,
    _messages,
    _model_name,
    _response_text,
    _safe,
    _token_usage,
    _tool_name,
)


# -- lightweight fakes mirroring LlamaIndex payload shapes --------------------


class FakeEventType(str, Enum):
    QUERY = "query"
    RETRIEVE = "retrieve"
    LLM = "llm"
    FUNCTION_CALL = "function_call"
    EMBEDDING = "embedding"
    EXCEPTION = "exception"


@dataclass
class FakeChatMessage:
    role: str = "user"
    content: str = ""


@dataclass
class FakeChatResponse:
    message: FakeChatMessage = field(default_factory=FakeChatMessage)
    raw: Any = None


@dataclass
class FakeCompletionResponse:
    text: str = ""
    raw: Any = None


@dataclass
class FakeToolMetadata:
    name: str = ""


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FakeRawResponse:
    usage: Any = None


def eid() -> str:
    return str(uuid.uuid4())


# -- helper tests -------------------------------------------------------------


class TestSafe:
    def test_passes_short_strings_through(self):
        assert _safe("hello") == "hello"

    def test_truncates_long_values(self):
        out = _safe("x" * 900)
        assert len(out) == 503
        assert out.endswith("...")

    def test_reprs_non_strings(self):
        assert _safe({"a": 1}) == "{'a': 1}"


class TestEventName:
    def test_str_enum_uses_value(self):
        assert _event_name(FakeEventType.QUERY) == "query"

    def test_plain_string_passthrough(self):
        assert _event_name("retrieve") == "retrieve"


class TestModelName:
    def test_from_serialized_model(self):
        assert _model_name({"serialized": {"model": "gpt-4o"}}) == "gpt-4o"

    def test_from_serialized_model_name(self):
        assert _model_name({"serialized": {"model_name": "claude-3"}}) == "claude-3"

    def test_from_top_level_model_name(self):
        assert _model_name({"model_name": "local-llm"}) == "local-llm"

    def test_fallback(self):
        assert _model_name({}) == "llm"


class TestMessages:
    def test_chat_messages(self):
        out = _messages({"messages": [FakeChatMessage("system", "be brief")]})
        assert out == [{"role": "system", "content": "be brief"}]

    def test_formatted_prompt(self):
        out = _messages({"formatted_prompt": "Answer: {q}"})
        assert out == [{"role": "user", "content": "Answer: {q}"}]

    def test_empty(self):
        assert _messages({}) == []


class TestResponseText:
    def test_chat_response(self):
        payload = {"response": FakeChatResponse(FakeChatMessage(content="hi"))}
        assert _response_text(payload) == "hi"

    def test_completion_response(self):
        assert _response_text({"completion": FakeCompletionResponse(text="done")}) == "done"

    def test_plain_string_response(self):
        assert _response_text({"response": "raw text"}) == "raw text"

    def test_empty(self):
        assert _response_text({}) == ""


class TestTokenUsage:
    def test_dict_raw_usage(self):
        payload = {
            "response": FakeChatResponse(raw={"usage": {"total_tokens": 30}})
        }
        assert _token_usage(payload) == {"total_tokens": 30}

    def test_object_usage_attributes(self):
        raw = FakeRawResponse(usage=FakeUsage(10, 20, 30))
        payload = {"completion": FakeCompletionResponse(text="x", raw=raw)}
        assert _token_usage(payload) == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_no_usage(self):
        assert _token_usage({"response": FakeChatResponse()}) == {}
        assert _token_usage({}) == {}


class TestToolName:
    def test_from_metadata_object(self):
        assert _tool_name({"tool": FakeToolMetadata(name="search")}) == "search"

    def test_from_string(self):
        assert _tool_name({"tool": "calculator"}) == "calculator"

    def test_fallback(self):
        assert _tool_name({}) == "tool"


# -- handler tests ------------------------------------------------------------


class TestSpanEvents:
    def test_query_becomes_span_with_query_metadata(self):
        handler = AgentReplayLlamaIndexHandler("t")
        qid = handler.on_event_start(
            FakeEventType.QUERY, {"query_str": "what is up"}, event_id=eid()
        )
        handler.on_event_end(FakeEventType.QUERY, {"response": "not much"}, event_id=qid)

        spans = handler.trace.spans
        assert len(spans) == 1
        assert spans[0].name == "query"
        assert spans[0].metadata["query"] == "what is up"
        assert spans[0].metadata["response"] == "not much"
        assert spans[0].end_time is not None

    def test_retrieve_nests_under_query(self):
        handler = AgentReplayLlamaIndexHandler("t")
        qid = handler.on_event_start(FakeEventType.QUERY, {}, event_id=eid())
        rid = handler.on_event_start(
            FakeEventType.RETRIEVE, {}, event_id=eid(), parent_id=qid
        )
        handler.on_event_end(
            FakeEventType.RETRIEVE, {"nodes": ["n1", "n2", "n3"]}, event_id=rid
        )

        query_span = handler.trace.spans[0]
        retrieve_span = handler.trace.spans[1]
        assert retrieve_span.parent_id == query_span.span_id
        assert retrieve_span.metadata["nodes_retrieved"] == 3

    def test_generates_event_id_when_missing(self):
        handler = AgentReplayLlamaIndexHandler("t")
        returned = handler.on_event_start(FakeEventType.QUERY, {})
        assert returned
        assert handler._spans[returned] is handler.trace.spans[0]


class TestLLMEvents:
    def test_llm_request_and_response_on_parent_span(self):
        handler = AgentReplayLlamaIndexHandler("t")
        qid = handler.on_event_start(FakeEventType.QUERY, {}, event_id=eid())
        lid = handler.on_event_start(
            FakeEventType.LLM,
            {
                "messages": [FakeChatMessage("user", "hello")],
                "serialized": {"model": "gpt-4o-mini"},
            },
            event_id=eid(),
            parent_id=qid,
        )
        handler.on_event_end(
            FakeEventType.LLM,
            {"response": FakeChatResponse(FakeChatMessage(content="hi there"), raw={"usage": {"total_tokens": 42}})},
            event_id=lid,
        )

        span = handler.trace.spans[0]
        types_ = [e.event_type for e in span.events]
        assert types_ == [EventType.LLM_REQUEST, EventType.LLM_RESPONSE]
        req, resp = span.events
        assert req.data["model"] == "gpt-4o-mini"
        assert req.data["messages"][0]["content"] == "hello"
        assert resp.data["content"] == "hi there"
        assert resp.data["tokens"] == 42

    def test_llm_without_parent_lands_on_root_session_span(self):
        handler = AgentReplayLlamaIndexHandler("t")
        lid = handler.on_event_start(FakeEventType.LLM, {}, event_id=eid())
        handler.on_event_end(FakeEventType.LLM, {}, event_id=lid)

        assert handler.trace.spans[0].name == "session"
        assert len(handler.trace.spans[0].events) == 2


class TestToolEvents:
    def test_function_call_records_tool_call_and_result(self):
        handler = AgentReplayLlamaIndexHandler("t")
        fid = handler.on_event_start(
            FakeEventType.FUNCTION_CALL,
            {"tool": FakeToolMetadata(name="search"), "function_call": '{"q": "x"}'},
            event_id=eid(),
        )
        handler.on_event_end(
            FakeEventType.FUNCTION_CALL,
            {"function_call_response": "3 results"},
            event_id=fid,
        )

        events = handler.trace.spans[0].events
        assert events[0].event_type == EventType.TOOL_CALL
        assert events[0].data["tool"] == "search"
        assert events[1].event_type == EventType.TOOL_RESULT
        assert events[1].data["tool"] == "search"
        assert events[1].data["result"] == "3 results"


class TestErrors:
    def test_exception_payload_on_event_end(self):
        handler = AgentReplayLlamaIndexHandler("t")
        qid = handler.on_event_start(FakeEventType.QUERY, {}, event_id=eid())
        handler.on_event_end(
            FakeEventType.QUERY,
            {"exception": ValueError("bad input")},
            event_id=qid,
        )

        span = handler.trace.spans[0]
        errors = [e for e in span.events if e.event_type == EventType.ERROR]
        assert len(errors) == 1
        assert errors[0].data["exception"] == "ValueError"
        assert "bad input" in errors[0].data["message"]

    def test_exception_event_type_on_start(self):
        handler = AgentReplayLlamaIndexHandler("t")
        handler.on_event_start(
            FakeEventType.EXCEPTION,
            {"exception": RuntimeError("boom")},
            event_id=eid(),
        )
        errors = [
            e
            for e in handler.trace.spans[0].events
            if e.event_type == EventType.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].data["exception"] == "RuntimeError"


class TestLogEvents:
    def test_embedding_end_logs_counts(self):
        handler = AgentReplayLlamaIndexHandler("t")
        bid = handler.on_event_start(FakeEventType.EMBEDDING, {}, event_id=eid())
        handler.on_event_end(
            FakeEventType.EMBEDDING,
            {"chunks": ["a", "b"], "embeddings": [[0.1], [0.2]]},
            event_id=bid,
        )

        logs = [
            e
            for e in handler.trace.spans[0].events
            if e.event_type == EventType.LOG
        ]
        assert len(logs) == 1
        assert "2 chunks" in logs[0].data["message"]
        assert "2 embeddings" in logs[0].data["message"]


class TestLifecycle:
    def test_start_and_end_trace_are_safe_noops(self):
        handler = AgentReplayLlamaIndexHandler("t")
        handler.start_trace("trace-1")
        handler.end_trace("trace-1", {"root": []})
        assert handler.trace.spans == []

    def test_finish_closes_spans_and_saves(self, tmp_path):
        handler = AgentReplayLlamaIndexHandler("rag-run", metadata={"env": "test"})
        qid = handler.on_event_start(
            FakeEventType.QUERY, {"query_str": "q"}, event_id=eid()
        )
        lid = handler.on_event_start(
            FakeEventType.LLM, {"formatted_prompt": "q"}, event_id=eid(), parent_id=qid
        )
        handler.on_event_end(
            FakeEventType.LLM,
            {"completion": FakeCompletionResponse(text="a")},
            event_id=lid,
        )

        out = tmp_path / "trace.jsonl"
        trace = handler.finish(out)

        assert trace.end_time is not None
        assert all(s.end_time is not None for s in trace.spans)

        loaded = Trace.load(out)
        assert loaded.name == "rag-run"
        assert loaded.metadata == {"env": "test"}
        assert loaded.spans[0].name == "query"
        assert loaded.event_count == 2
