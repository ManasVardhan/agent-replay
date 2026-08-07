"""Tests for trace cost analytics: extraction, pricing, aggregation, and CLI."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.cost import (
    UNKNOWN_MODEL,
    analyze_trace,
    resolve_pricing,
)
from agent_replay.trace import EventType, Trace


def make_trace() -> Trace:
    """Trace with two spans and a mix of usage shapes."""
    trace = Trace(name="cost-test")
    span1 = trace.add_span("plan")
    span1.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o", "prompt": "plan it"})
    span1.add_event(
        EventType.LLM_RESPONSE,
        {
            "content": "plan",
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        },
    )
    span2 = trace.add_span("execute")
    span2.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o-mini", "prompt": "do it"})
    span2.add_event(EventType.LLM_RESPONSE, {"content": "done", "tokens": 2000})
    trace.close()
    return trace


class TestResolvePricing:
    def test_exact_match(self):
        assert resolve_pricing("gpt-4o") == (2.50, 10.00)

    def test_prefix_match_versioned(self):
        assert resolve_pricing("gpt-4o-2024-08-06") == (2.50, 10.00)

    def test_longest_prefix_wins(self):
        assert resolve_pricing("gpt-4o-mini-2024-07-18") == (0.15, 0.60)

    def test_case_insensitive(self):
        assert resolve_pricing("GPT-4o") == (2.50, 10.00)

    def test_claude_prefix(self):
        assert resolve_pricing("claude-sonnet-4-20250514") == (3.00, 15.00)

    def test_unknown_returns_none(self):
        assert resolve_pricing("some-local-model") is None

    def test_overrides_win(self):
        assert resolve_pricing("gpt-4o", {"gpt-4o": (1.0, 2.0)}) == (1.0, 2.0)

    def test_overrides_add_new_model(self):
        assert resolve_pricing("my-llama", {"my-llama": (0.1, 0.2)}) == (0.1, 0.2)

    def test_override_prefix_match(self):
        assert resolve_pricing("my-llama-v2", {"my-llama": (0.1, 0.2)}) == (0.1, 0.2)


class TestAnalyzeTrace:
    def test_exact_usage_costing(self):
        report = analyze_trace(make_trace())
        call = report.calls[0]
        assert call.model == "gpt-4o"
        assert call.prompt_tokens == 1000
        assert call.completion_tokens == 500
        # 1000/1M * 2.50 + 500/1M * 10.00
        assert call.cost_usd == pytest.approx(0.0025 + 0.005)
        assert not call.estimated

    def test_total_only_estimated(self):
        report = analyze_trace(make_trace())
        call = report.calls[1]
        assert call.model == "gpt-4o-mini"
        assert call.total_tokens == 2000
        # 2000/1M * avg(0.15, 0.60)
        assert call.cost_usd == pytest.approx(2000 / 1_000_000 * 0.375)
        assert call.estimated
        assert report.has_estimates

    def test_totals(self):
        report = analyze_trace(make_trace())
        assert len(report.calls) == 2
        assert report.total_tokens == 3500
        assert report.total_cost_usd == pytest.approx(
            report.calls[0].cost_usd + report.calls[1].cost_usd
        )

    def test_model_from_response_event_wins(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span.add_event(EventType.LLM_RESPONSE, {"model": "gpt-4.1", "tokens": 100})
        report = analyze_trace(trace)
        assert report.calls[0].model == "gpt-4.1"

    def test_response_without_request_is_unknown_model(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 100})
        report = analyze_trace(trace)
        assert report.calls[0].model == UNKNOWN_MODEL
        assert report.calls[0].cost_usd is None
        assert report.unpriced_models == []

    def test_unpriced_model_reported(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "local-llama"})
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 100})
        report = analyze_trace(trace)
        assert report.calls[0].cost_usd is None
        assert report.unpriced_models == ["local-llama"]
        assert report.total_cost_usd == 0.0

    def test_pricing_overrides(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "local-llama"})
        span.add_event(
            EventType.LLM_RESPONSE,
            {"token_usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}},
        )
        report = analyze_trace(trace, pricing={"local-llama": (1.0, 2.0)})
        assert report.calls[0].cost_usd == pytest.approx(3.0)
        assert report.unpriced_models == []

    def test_response_without_usage_skipped(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span.add_event(EventType.LLM_RESPONSE, {"content": "hi"})
        report = analyze_trace(trace)
        assert report.calls == []
        assert report.calls_without_usage == 1

    def test_model_carries_across_multiple_responses_in_span(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 10})
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 20})
        report = analyze_trace(trace)
        assert [c.model for c in report.calls] == ["gpt-4o", "gpt-4o"]

    def test_model_does_not_leak_across_spans(self):
        trace = Trace(name="t")
        span1 = trace.add_span("a")
        span1.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span1.add_event(EventType.LLM_RESPONSE, {"tokens": 10})
        span2 = trace.add_span("b")
        span2.add_event(EventType.LLM_RESPONSE, {"tokens": 10})
        report = analyze_trace(trace)
        assert report.calls[1].model == UNKNOWN_MODEL

    def test_partial_token_usage(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span.add_event(EventType.LLM_RESPONSE, {"token_usage": {"prompt_tokens": 100}})
        report = analyze_trace(trace)
        call = report.calls[0]
        assert call.prompt_tokens == 100
        assert call.total_tokens == 100
        assert call.cost_usd == pytest.approx(100 / 1_000_000 * 2.50)
        assert not call.estimated

    def test_bad_usage_values_ignored(self):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "gpt-4o"})
        span.add_event(
            EventType.LLM_RESPONSE,
            {"token_usage": {"prompt_tokens": "lots", "completion_tokens": -5}},
        )
        report = analyze_trace(trace)
        assert report.calls == []
        assert report.calls_without_usage == 1

    def test_by_model_aggregation(self):
        report = analyze_trace(make_trace())
        rows = report.by_model()
        assert [r["model"] for r in rows] == ["gpt-4o", "gpt-4o-mini"]
        assert rows[0]["calls"] == 1
        costs = [r["cost_usd"] for r in rows]
        assert costs == sorted(costs, reverse=True)

    def test_by_span_aggregation(self):
        report = analyze_trace(make_trace())
        rows = report.by_span()
        assert [r["span_name"] for r in rows] == ["plan", "execute"]
        assert rows[0]["total_tokens"] == 1500

    def test_to_dict_json_safe(self):
        report = analyze_trace(make_trace())
        payload = report.to_dict()
        json.dumps(payload)
        assert payload["llm_calls"] == 2
        assert payload["trace_name"] == "cost-test"

    def test_empty_trace(self):
        report = analyze_trace(Trace(name="empty"))
        assert report.calls == []
        assert report.total_cost_usd == 0.0
        assert report.total_tokens == 0


class TestCostCli:
    def _save_trace(self, tmp_path, trace=None) -> str:
        path = tmp_path / "trace.jsonl"
        (trace or make_trace()).save(path)
        return str(path)

    def test_cost_output(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", self._save_trace(tmp_path)])
        assert result.exit_code == 0
        assert "Cost by Model" in result.output
        assert "Cost by Span" in result.output
        assert "gpt-4o" in result.output
        assert "estimated" in result.output  # gpt-4o-mini total-only note

    def test_cost_json_output(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", self._save_trace(tmp_path), "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["llm_calls"] == 2
        assert payload["total_tokens"] == 3500
        assert payload["total_cost_usd"] > 0

    def test_cost_price_override(self, tmp_path):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "my-model"})
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 1_000_000})
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "cost",
                self._save_trace(tmp_path, trace),
                "--price",
                "my-model=2:4",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total_cost_usd"] == pytest.approx(3.0)
        assert payload["unpriced_models"] == []

    def test_cost_unpriced_hint(self, tmp_path):
        trace = Trace(name="t")
        span = trace.add_span("s")
        span.add_event(EventType.LLM_REQUEST, {"model": "mystery-model"})
        span.add_event(EventType.LLM_RESPONSE, {"tokens": 100})
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", self._save_trace(tmp_path, trace)])
        assert result.exit_code == 0
        assert "No pricing for: mystery-model" in result.output
        assert "--price" in result.output

    def test_cost_empty_trace(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", self._save_trace(tmp_path, Trace(name="e"))])
        assert result.exit_code == 0
        assert "No LLM calls" in result.output

    def test_cost_bad_price_format(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cost", self._save_trace(tmp_path), "--price", "gpt-4o=oops"]
        )
        assert result.exit_code != 0
        assert "MODEL=INPUT:OUTPUT" in result.output

    def test_cost_bad_price_values(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cost", self._save_trace(tmp_path), "--price", "gpt-4o=a:b"]
        )
        assert result.exit_code != 0
        assert "invalid prices" in result.output

    def test_cost_negative_price(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cost", self._save_trace(tmp_path), "--price", "gpt-4o=-1:2"]
        )
        assert result.exit_code != 0
        assert "non-negative" in result.output

    def test_cost_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "nope.jsonl"])
        assert result.exit_code != 0
