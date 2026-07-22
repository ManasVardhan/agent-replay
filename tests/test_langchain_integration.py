"""Tests for the LangChain callback handler integration.

langchain-core is an optional dependency; when it is not installed, a
minimal stub providing BaseCallbackHandler is injected so the handler's
recording logic is still fully tested.
"""

from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

try:
    import langchain_core.callbacks  # noqa: F401

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    pkg = types.ModuleType("langchain_core")
    callbacks_mod = types.ModuleType("langchain_core.callbacks")

    class BaseCallbackHandler:
        pass

    callbacks_mod.BaseCallbackHandler = BaseCallbackHandler
    pkg.callbacks = callbacks_mod
    sys.modules["langchain_core"] = pkg
    sys.modules["langchain_core.callbacks"] = callbacks_mod

from agent_replay import EventType, Trace
from agent_replay.integrations.langchain import (
    AgentReplayCallbackHandler,
    _generation_texts,
    _run_name,
    _safe,
    _token_usage,
)


# -- lightweight fakes mirroring LangChain result shapes ----------------------


@dataclass
class FakeGeneration:
    text: str = ""
    message: Any = None


@dataclass
class FakeMessage:
    content: str = ""
    usage_metadata: Any = None


@dataclass
class FakeLLMResult:
    generations: list[list[FakeGeneration]] = field(default_factory=list)
    llm_output: Any = None


@dataclass
class FakeAgentAction:
    tool: str
    tool_input: str
    log: str


@dataclass
class FakeAgentFinish:
    return_values: dict
    log: str


def rid() -> uuid.UUID:
    return uuid.uuid4()


class TestHelpers:
    def test_safe_truncates(self):
        assert _safe("x" * 600) == "x" * 500 + "..."

    def test_safe_repr_for_non_string(self):
        assert _safe({"a": 1}) == "{'a': 1}"

    def test_safe_unprintable(self):
        class Bad:
            def __repr__(self):
                raise RuntimeError("nope")

        assert "unprintable" in _safe(Bad())

    def test_run_name_prefers_kwargs(self):
        assert _run_name({"name": "ser"}, {"name": "kw"}, "fb") == "kw"

    def test_run_name_from_serialized_id(self):
        assert (
            _run_name({"id": ["langchain", "llms", "FakeLLM"]}, {}, "fb") == "FakeLLM"
        )

    def test_run_name_fallback(self):
        assert _run_name(None, {}, "fb") == "fb"

    def test_token_usage_from_llm_output(self):
        result = FakeLLMResult(llm_output={"token_usage": {"total_tokens": 7}})
        assert _token_usage(result) == {"total_tokens": 7}

    def test_token_usage_from_usage_metadata(self):
        msg = FakeMessage(content="hi", usage_metadata={"total_tokens": 3})
        result = FakeLLMResult(generations=[[FakeGeneration(message=msg)]])
        assert _token_usage(result) == {"total_tokens": 3}

    def test_token_usage_missing(self):
        assert _token_usage(FakeLLMResult()) == {}

    def test_generation_texts_from_text_and_message(self):
        result = FakeLLMResult(
            generations=[
                [FakeGeneration(text="a")],
                [FakeGeneration(message=FakeMessage("b"))],
            ]
        )
        assert _generation_texts(result) == ["a", "b"]


class TestChainSpans:
    def test_chain_start_creates_span(self):
        handler = AgentReplayCallbackHandler("t")
        handler.on_chain_start({"name": "my_chain"}, {"q": "hi"}, run_id=rid())
        assert len(handler.trace.spans) == 1
        span = handler.trace.spans[0]
        assert span.name == "my_chain"
        assert "q" in span.metadata["inputs"]

    def test_chain_end_closes_span_and_stores_outputs(self):
        handler = AgentReplayCallbackHandler("t")
        run = rid()
        handler.on_chain_start({"name": "c"}, {}, run_id=run)
        handler.on_chain_end({"answer": 42}, run_id=run)
        span = handler.trace.spans[0]
        assert span.end_time is not None
        assert "42" in span.metadata["outputs"]

    def test_nested_chains_link_parent(self):
        handler = AgentReplayCallbackHandler("t")
        outer, inner = rid(), rid()
        handler.on_chain_start({"name": "outer"}, {}, run_id=outer)
        handler.on_chain_start({"name": "inner"}, {}, run_id=inner, parent_run_id=outer)
        outer_span, inner_span = handler.trace.spans
        assert inner_span.parent_id == outer_span.span_id

    def test_chain_error_records_event_and_closes(self):
        handler = AgentReplayCallbackHandler("t")
        run = rid()
        handler.on_chain_start({"name": "c"}, {}, run_id=run)
        handler.on_chain_error(ValueError("bad input"), run_id=run)
        span = handler.trace.spans[0]
        assert span.end_time is not None
        [event] = span.events
        assert event.event_type == EventType.ERROR
        assert event.data["exception"] == "ValueError"
        assert event.data["source"] == "chain"


class TestLLMEvents:
    def test_llm_start_records_request_on_parent_chain(self):
        handler = AgentReplayCallbackHandler("t")
        chain, llm = rid(), rid()
        handler.on_chain_start({"name": "c"}, {}, run_id=chain)
        handler.on_llm_start(
            {"id": ["x", "FakeLLM"]}, ["hello"], run_id=llm, parent_run_id=chain
        )
        [event] = handler.trace.spans[0].events
        assert event.event_type == EventType.LLM_REQUEST
        assert event.data["model"] == "FakeLLM"
        assert event.data["messages"] == ["hello"]

    def test_llm_end_matches_llm_run_span(self):
        handler = AgentReplayCallbackHandler("t")
        chain, llm = rid(), rid()
        handler.on_chain_start({"name": "c"}, {}, run_id=chain)
        handler.on_llm_start(None, ["p"], run_id=llm, parent_run_id=chain)
        result = FakeLLMResult(
            generations=[[FakeGeneration(text="out")]],
            llm_output={"token_usage": {"total_tokens": 11}},
        )
        handler.on_llm_end(result, run_id=llm)
        events = handler.trace.spans[0].events
        assert [e.event_type for e in events] == [
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
        ]
        assert events[1].data["content"] == "out"
        assert events[1].data["tokens"] == 11

    def test_chat_model_start_flattens_messages(self):
        handler = AgentReplayCallbackHandler("t")
        handler.on_chat_model_start(
            {"name": "FakeChat"},
            [[FakeMessage(content="sys"), FakeMessage(content="hi")]],
            run_id=rid(),
        )
        [event] = handler.trace.spans[0].events
        assert event.event_type == EventType.LLM_REQUEST
        assert [m["content"] for m in event.data["messages"]] == ["sys", "hi"]
        assert event.data["messages"][0]["role"] == "FakeMessage"

    def test_llm_error_records_error(self):
        handler = AgentReplayCallbackHandler("t")
        handler.on_llm_error(TimeoutError("slow"), run_id=rid())
        [event] = handler.trace.spans[0].events
        assert event.event_type == EventType.ERROR
        assert event.data["source"] == "llm"

    def test_orphan_llm_event_lands_on_session_span(self):
        handler = AgentReplayCallbackHandler("t")
        handler.on_llm_start(None, ["p"], run_id=rid())
        assert handler.trace.spans[0].name == "session"


class TestToolEvents:
    def test_tool_call_and_result_share_name(self):
        handler = AgentReplayCallbackHandler("t")
        chain, tool = rid(), rid()
        handler.on_chain_start({"name": "agent"}, {}, run_id=chain)
        handler.on_tool_start(
            {"name": "search"}, "python", run_id=tool, parent_run_id=chain
        )
        handler.on_tool_end("3 results", run_id=tool)
        events = handler.trace.spans[0].events
        assert [e.event_type for e in events] == [
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
        ]
        assert events[0].data["tool"] == "search"
        assert events[0].data["args"]["input"] == "python"
        assert events[1].data == {"tool": "search", "result": "3 results"}

    def test_tool_error(self):
        handler = AgentReplayCallbackHandler("t")
        tool = rid()
        handler.on_tool_start({"name": "web"}, "q", run_id=tool)
        handler.on_tool_error(ConnectionError("down"), run_id=tool)
        error = handler.trace.spans[0].events[-1]
        assert error.event_type == EventType.ERROR
        assert error.data["source"] == "tool:web"


class TestAgentDecisions:
    def test_agent_action_recorded_as_decision(self):
        handler = AgentReplayCallbackHandler("t")
        action = FakeAgentAction(tool="calculator", tool_input="2+2", log="using calc")
        handler.on_agent_action(action, run_id=rid())
        [event] = handler.trace.spans[0].events
        assert event.event_type == EventType.DECISION
        assert event.data["choice"] == "calculator"
        assert event.data["tool_input"] == "2+2"

    def test_agent_finish(self):
        handler = AgentReplayCallbackHandler("t")
        finish = FakeAgentFinish(return_values={"output": "done"}, log="finishing")
        handler.on_agent_finish(finish, run_id=rid())
        [event] = handler.trace.spans[0].events
        assert event.data["choice"] == "finish"
        assert "done" in event.data["return_values"]

    def test_on_text_logs(self):
        handler = AgentReplayCallbackHandler("t")
        handler.on_text("thinking...", run_id=rid())
        [event] = handler.trace.spans[0].events
        assert event.event_type == EventType.LOG
        assert event.data["message"] == "thinking..."


class TestFinish:
    def test_finish_closes_and_saves_roundtrip(self, tmp_path):
        handler = AgentReplayCallbackHandler("t", metadata={"env": "test"})
        run = rid()
        handler.on_chain_start({"name": "c"}, {"q": 1}, run_id=run)
        handler.on_llm_start(None, ["p"], run_id=rid(), parent_run_id=run)
        path = tmp_path / "trace.jsonl"
        trace = handler.finish(path)
        assert trace.end_time is not None
        assert all(s.end_time is not None for s in trace.spans)
        loaded = Trace.load(path)
        assert loaded.name == "t"
        assert loaded.metadata == {"env": "test"}
        assert loaded.event_count == 1

    def test_finish_without_path_does_not_save(self):
        handler = AgentReplayCallbackHandler("t")
        trace = handler.finish()
        assert trace.end_time is not None

    def test_full_agent_lifecycle_event_order(self):
        handler = AgentReplayCallbackHandler("run")
        agent, llm, tool = rid(), rid(), rid()
        handler.on_chain_start({"name": "agent"}, {"task": "add"}, run_id=agent)
        handler.on_llm_start(None, ["what is 2+2"], run_id=llm, parent_run_id=agent)
        handler.on_llm_end(
            FakeLLMResult(generations=[[FakeGeneration(text="use calc")]]), run_id=llm
        )
        handler.on_agent_action(
            FakeAgentAction("calc", "2+2", "calling calc"), run_id=agent
        )
        handler.on_tool_start({"name": "calc"}, "2+2", run_id=tool, parent_run_id=agent)
        handler.on_tool_end("4", run_id=tool)
        handler.on_agent_finish(FakeAgentFinish({"output": "4"}, "done"), run_id=agent)
        handler.on_chain_end({"output": "4"}, run_id=agent)
        trace = handler.finish()
        types_seen = [e.event_type for e in trace.all_events()]
        assert types_seen == [
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
            EventType.DECISION,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.DECISION,
        ]


class TestImportError:
    def test_missing_langchain_gives_helpful_error(self, monkeypatch):
        import importlib

        import agent_replay.integrations.langchain as module

        monkeypatch.setitem(sys.modules, "langchain_core", None)
        monkeypatch.setitem(sys.modules, "langchain_core.callbacks", None)
        try:
            with pytest.raises(ImportError, match="pip install langchain-core"):
                importlib.reload(module)
        finally:
            monkeypatch.undo()
            importlib.reload(module)
