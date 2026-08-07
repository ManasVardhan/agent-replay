"""Trace cost analytics: token and cost breakdowns from recorded LLM events.

Walks a trace's LLM events, associates responses with the model announced by
the preceding request in the same span, and prices token usage against a
built-in per-1M-token table (overridable per call). Works with traces from the
Recorder, the LangChain integration, and the LlamaIndex integration:

- ``token_usage`` dicts (``prompt_tokens`` / ``completion_tokens`` /
  ``total_tokens``) get exact input/output pricing
- a bare ``tokens`` total gets an estimated cost using the average of the
  model's input and output price (flagged as estimated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace import EventType, Trace

UNKNOWN_MODEL = "(unknown)"

# model prefix -> (input USD per 1M tokens, output USD per 1M tokens)
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-haiku-4": (1.00, 5.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


def resolve_pricing(
    model: str, overrides: dict[str, tuple[float, float]] | None = None
) -> tuple[float, float] | None:
    """Look up (input, output) USD per 1M tokens for a model name.

    Tries an exact match first, then the longest matching prefix, so versioned
    names like ``gpt-4o-2024-08-06`` resolve to ``gpt-4o``. Overrides win over
    the built-in table. Returns None for unknown models.
    """
    tables: list[dict[str, tuple[float, float]]] = []
    if overrides:
        tables.append(overrides)
    tables.append(PRICING)

    name = model.lower()
    for table in tables:
        if model in table:
            return table[model]
        lowered = {key.lower(): value for key, value in table.items()}
        if name in lowered:
            return lowered[name]
        best_key = ""
        for key in lowered:
            if name.startswith(key) and len(key) > len(best_key):
                best_key = key
        if best_key:
            return lowered[best_key]
    return None


@dataclass
class LLMCall:
    """One priced LLM response event."""

    span_id: str
    span_name: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8) if self.cost_usd is not None else None,
            "estimated": self.estimated,
        }


@dataclass
class CostReport:
    """Aggregated cost analytics for a trace."""

    trace_name: str
    trace_id: str
    calls: list[LLMCall] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)
    calls_without_usage: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls if c.cost_usd is not None)

    @property
    def has_estimates(self) -> bool:
        return any(c.estimated for c in self.calls)

    def by_model(self) -> list[dict[str, Any]]:
        """Aggregate calls per model, sorted by cost descending."""
        buckets: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            b = buckets.setdefault(
                call.model,
                {
                    "model": call.model,
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": None,
                    "estimated": False,
                },
            )
            b["calls"] += 1
            b["prompt_tokens"] += call.prompt_tokens
            b["completion_tokens"] += call.completion_tokens
            b["total_tokens"] += call.total_tokens
            if call.cost_usd is not None:
                b["cost_usd"] = (b["cost_usd"] or 0.0) + call.cost_usd
            if call.estimated:
                b["estimated"] = True
        rows = list(buckets.values())
        for row in rows:
            if row["cost_usd"] is not None:
                row["cost_usd"] = round(row["cost_usd"], 8)
        rows.sort(key=lambda r: -(r["cost_usd"] or 0.0))
        return rows

    def by_span(self) -> list[dict[str, Any]]:
        """Aggregate calls per span, in trace order."""
        buckets: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            b = buckets.setdefault(
                call.span_id,
                {
                    "span_id": call.span_id,
                    "span_name": call.span_name,
                    "calls": 0,
                    "total_tokens": 0,
                    "cost_usd": None,
                },
            )
            b["calls"] += 1
            b["total_tokens"] += call.total_tokens
            if call.cost_usd is not None:
                b["cost_usd"] = (b["cost_usd"] or 0.0) + call.cost_usd
        rows = list(buckets.values())
        for row in rows:
            if row["cost_usd"] is not None:
                row["cost_usd"] = round(row["cost_usd"], 8)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_name": self.trace_name,
            "trace_id": self.trace_id,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "total_tokens": self.total_tokens,
            "llm_calls": len(self.calls),
            "calls_without_usage": self.calls_without_usage,
            "has_estimates": self.has_estimates,
            "unpriced_models": self.unpriced_models,
            "by_model": self.by_model(),
            "by_span": self.by_span(),
            "calls": [c.to_dict() for c in self.calls],
        }


def _coerce_int(value: Any) -> int | None:
    """Return value as a non-negative int, or None if it is not usable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0:
        return int(value)
    return None


def _extract_usage(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Return (prompt, completion, total) token counts from an LLM response event."""
    usage = data.get("token_usage")
    if isinstance(usage, dict):
        prompt = _coerce_int(usage.get("prompt_tokens"))
        completion = _coerce_int(usage.get("completion_tokens"))
        total = _coerce_int(usage.get("total_tokens"))
        if prompt is not None or completion is not None or total is not None:
            if total is None:
                total = (prompt or 0) + (completion or 0)
            return prompt, completion, total
    total = _coerce_int(data.get("tokens"))
    if total is not None:
        return None, None, total
    return None, None, None


def analyze_trace(
    trace: Trace, pricing: dict[str, tuple[float, float]] | None = None
) -> CostReport:
    """Compute per-call, per-model, and per-span cost analytics for a trace.

    Parameters
    ----------
    trace : the trace to analyze
    pricing : optional overrides mapping model name (or prefix) to
              (input, output) USD per 1M tokens; wins over the built-in table

    Returns
    -------
    A CostReport. Calls against models with no known pricing get
    ``cost_usd=None`` and the model is listed in ``unpriced_models``.
    """
    report = CostReport(trace_name=trace.name, trace_id=trace.trace_id)
    unpriced: dict[str, None] = {}

    for span in trace.spans:
        current_model: str | None = None
        for event in span.events:
            if event.event_type == EventType.LLM_REQUEST:
                model = event.data.get("model")
                if isinstance(model, str) and model:
                    current_model = model
                continue
            if event.event_type != EventType.LLM_RESPONSE:
                continue

            data = event.data
            response_model = data.get("model")
            model = (
                response_model
                if isinstance(response_model, str) and response_model
                else current_model or UNKNOWN_MODEL
            )
            prompt, completion, total = _extract_usage(data)
            if total is None:
                report.calls_without_usage += 1
                continue

            call = LLMCall(
                span_id=span.span_id,
                span_name=span.name,
                model=model,
                prompt_tokens=prompt or 0,
                completion_tokens=completion or 0,
                total_tokens=total,
            )

            prices = resolve_pricing(model, pricing) if model != UNKNOWN_MODEL else None
            if prices is None:
                if model != UNKNOWN_MODEL:
                    unpriced[model] = None
            else:
                input_price, output_price = prices
                if prompt is not None or completion is not None:
                    call.cost_usd = (
                        (prompt or 0) / 1_000_000 * input_price
                        + (completion or 0) / 1_000_000 * output_price
                    )
                else:
                    # Only a total is known: estimate with the average price.
                    call.cost_usd = total / 1_000_000 * (input_price + output_price) / 2
                    call.estimated = True
            report.calls.append(call)

    report.unpriced_models = sorted(unpriced)
    return report
