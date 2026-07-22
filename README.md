
# agent-replay

> **New here?** Start with the [Getting Started Guide](GETTING_STARTED.md).

[![PyPI version](https://img.shields.io/pypi/v/agent-trace-replay.svg)](https://pypi.org/project/agent-trace-replay/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/manasvardhan/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/manasvardhan/agent-replay/actions)

**AI agents are black boxes. agent-replay makes them transparent.**

Record every LLM call, tool use, decision point, and state change during agent execution. Replay them step-by-step. Diff two runs to find exactly where behavior diverged.

## Features

- 🎬 **Record** agent runs with a simple context manager or decorator
- 🔗 **LangChain integration** - capture traces automatically with a callback handler
- ⏯️ **Replay** traces step-by-step in the terminal
- 🔍 **Diff** two traces to find divergence points, with side-by-side HTML comparison reports
- 🌳 **Tree view** of nested spans and events
- 📊 **HTML export** with a self-contained dark-mode timeline
- 📡 **OpenTelemetry export** (OTLP/JSON) for Jaeger, Tempo, and friends
- 🧩 **Structured traces** with spans, events, and metadata
- ⌨️ **CLI** for quick inspection without writing code
- 🐍 **Typed Python 3.10+** with zero heavy dependencies

## Architecture

```
Agent Run ──> Recorder ──> Trace File (.jsonl) ──> Replay Viewer
                                                 ──> Diff Tool
                                                 ──> HTML Export
                                                 ──> OTLP Export
```

```
┌─────────────────────────────────────────────────────────────┐
│  Your Agent Code                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  with Recorder("my-agent") as rec:                    │  │
│  │      with rec.span("planning"):                       │  │
│  │          rec.llm_request(model="gpt-4", ...)          │  │
│  │          rec.llm_response(content="...", tokens=42)   │  │
│  │      with rec.span("tool-use"):                       │  │
│  │          rec.tool_call("search", {"q": "..."})        │  │
│  │          rec.tool_result("search", {...})             │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    trace.jsonl
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         agent-replay  agent-replay  agent-replay
           show          replay        diff
```

## Quick Start

```bash
pip install agent-trace-replay
```

```python
from agent_replay import Recorder

with Recorder("my-agent", output_path="trace.jsonl") as rec:
    with rec.span("planning"):
        rec.llm_request(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
        rec.llm_response(content="Hi there!", tokens=5)
    with rec.span("tool-use"):
        rec.tool_call("search", {"query": "python docs"})
        rec.tool_result("search", {"url": "https://docs.python.org"})
```

Then inspect it:

```bash
agent-replay show trace.jsonl
agent-replay show trace.jsonl --tree
agent-replay replay trace.jsonl
```

## Terminal Viewer

```
╭──────────── Agent Trace ────────────╮
│ my-agent                            │
│ ID: a1b2c3d4e5f67890                │
│ Spans: 2 | Events: 4               │
│ Duration: 1.234s                    │
╰─────────────────────────────────────╯

>>> planning (0.523s)
  🧠 LLM REQUEST  model=gpt-4 messages=1
  💬 LLM RESPONSE "Hi there!" (5 tokens)

>>> tool-use (0.711s)
  🔧 TOOL CALL search({"query": "python docs"})
  📦 TOOL RESULT search -> {"url": "https://docs.python.org"}
```

## Recording

### Context Manager

```python
from agent_replay import Recorder

with Recorder("my-agent", output_path="trace.jsonl") as rec:
    with rec.span("step-1"):
        rec.llm_request(model="gpt-4", messages=[...])
        rec.llm_response(content="...", tokens=10)
        rec.decision("next action", choice="search")
        rec.tool_call("search", {"q": "test"})
        rec.tool_result("search", {"results": [...]})
        rec.state_change("status", old="planning", new="executing")
```

### Decorator

```python
from agent_replay import record_trace, Recorder

@record_trace("my-agent", output_path="trace.jsonl")
def run_agent(task: str, recorder: Recorder = None):
    with recorder.span("work"):
        recorder.llm_request(model="gpt-4")
        recorder.llm_response(content="done")
```

### LangChain Integration

No manual instrumentation needed: attach the callback handler to any
LangChain runnable, chain, agent, or LLM and the full run is captured
automatically. Chain runs become nested spans; LLM requests/responses
(with token usage), tool calls/results, agent decisions, and errors
become events.

```bash
pip install "agent-trace-replay[langchain]"  # pulls in langchain-core
```

```python
from agent_replay.integrations.langchain import AgentReplayCallbackHandler

handler = AgentReplayCallbackHandler("support-agent")
chain.invoke({"question": "..."}, config={"callbacks": [handler]})
agent_executor.invoke({"input": "..."}, config={"callbacks": [handler]})

trace = handler.finish("trace.jsonl")  # close spans and save
```

The saved trace works with every agent-replay feature: `show`, `tree`,
`play`, `diff`, `redact`, HTML export, and OTLP export. Long payloads are
truncated at 500 characters and unknown callback shapes are handled
defensively, so the handler never breaks a run.

### Event Types

| Event | Method | Description |
|-------|--------|-------------|
| `llm_request` | `rec.llm_request()` | LLM API call with model and messages |
| `llm_response` | `rec.llm_response()` | LLM response with content and token count |
| `tool_call` | `rec.tool_call()` | Tool invocation with name and arguments |
| `tool_result` | `rec.tool_result()` | Tool return value |
| `decision` | `rec.decision()` | Agent decision point with chosen action |
| `state_change` | `rec.state_change()` | State mutation with old/new values |
| `error` | `rec.error()` | Error with message and exception info |
| `log` | `rec.log()` | General log message |

## Replay

Step through traces interactively in the terminal:

```bash
agent-replay replay trace.jsonl
```

Commands during replay:
- `n` / `next` - advance one step
- `p` / `prev` - go back one step
- `j N` / `jump N` - jump to step N
- `q` / `quit` - exit

Programmatic replay:

```python
from agent_replay import ReplayEngine

engine = ReplayEngine.from_file("trace.jsonl")
while engine.has_next():
    span, event = engine.step()
    print(f"[{span.name}] {event.event_type.value}")
```

### Streaming Playback

Watch a trace unfold with its original timing, like a terminal screencast.
Great for demos, debugging, and onboarding:

```bash
# Real-time pacing (long gaps capped at 2s by default)
agent-replay play trace.jsonl

# Twice as fast, cap pauses at half a second
agent-replay play trace.jsonl --speed 2 --max-delay 0.5

# Dump the timeline instantly, no pauses
agent-replay play trace.jsonl --no-delay
```

Each line shows the step number, elapsed trace time, span, event type,
and a compact summary:

```
[3/12] +   1.204s agent-loop 🔧 tool_call search({'q': 'weather SF'})
[4/12] +   2.891s agent-loop 📦 tool_result search -> {'temp': 61}
```

Press Ctrl+C to stop playback early. Programmatic access via
`engine.playback_plan(speed=2.0, max_delay=1.0)`, which returns
`PlaybackStep` objects with speed-adjusted delays and elapsed times.

## Diffing

Compare two traces to find where agent behavior diverged:

```bash
agent-replay diff trace_a.jsonl trace_b.jsonl
```

```
╭───────────── Trace Diff ─────────────╮
│ Trace A: a1b2c3d4                    │
│ Trace B: e5f6a7b8                    │
│ Found 2 divergence(s): 1 critical,   │
│ 1 informational.                     │
╰──────────────────────────────────────╯
┌──────────────── Divergences ────────────────┐
│ # │ Severity │ Pos │ Description            │
├───┼──────────┼─────┼────────────────────────┤
│ 1 │ CRITICAL │ 3   │ Different tool called:  │
│   │          │     │ search vs browse        │
│ 2 │ INFO     │ 5   │ LLM response content   │
│   │          │     │ differs                 │
└───┴──────────┴─────┴────────────────────────┘
```

Programmatic diffing:

```python
from agent_replay import Trace, diff_traces

a = Trace.load("trace_a.jsonl")
b = Trace.load("trace_b.jsonl")
result = diff_traces(a, b)

for div in result.divergences:
    print(f"[{div.severity}] Position {div.position}: {div.description}")
```

### Side-by-Side HTML Comparison

Generate a shareable side-by-side comparison report, great for reviewing regressions after a prompt change:

```bash
agent-replay diff before.jsonl after.jsonl --html comparison.html --title "Prompt v2 check"
```

The report is a single self-contained HTML file (inline CSS, no JavaScript, no external assets) that aligns both event streams column by column and highlights every divergence: critical rows in red, warnings in yellow, informational differences in blue, each with the diff description underneath.

Machine-readable output for CI pipelines:

```bash
agent-replay diff before.jsonl after.jsonl --json-output
```

Programmatic access via `render_diff_html(trace_a, trace_b)` (returns the HTML string) and `export_diff_html(trace_a, trace_b, "report.html")`.

## Redaction

Scrub API keys, tokens, and emails from a trace before sharing it in a bug report or public repo:

```bash
agent-replay redact trace.jsonl
```

```
Redacted 3 match(es):
  bearer_token             1
  email                    1
  openai_key               1
Redacted trace written to trace.redacted.jsonl
```

Builtin rules cover OpenAI, Anthropic, AWS, and GitHub keys, bearer tokens, and email addresses. Add custom rules with repeatable `--pattern LABEL=REGEX` options:

```bash
agent-replay redact trace.jsonl -p "ssn=\d{3}-\d{2}-\d{4}" -o clean.jsonl
```

Programmatic redaction:

```python
from agent_replay import Trace, redact_trace

trace = Trace.load("trace.jsonl")
clean, counts = redact_trace(trace, extra_patterns={"acme_id": r"ACME-\d+"})
clean.save("trace.redacted.jsonl")
print(counts)  # {"openai_key": 1, "email": 2, ...}
```

## HTML Export

Generate a self-contained HTML timeline:

```bash
agent-replay export trace.jsonl --format html -o timeline.html
```

The HTML file uses a dark theme with color-coded event types and expandable data sections. No external dependencies needed to view it.

## OpenTelemetry Export

Convert any trace to OTLP/JSON, the standard OpenTelemetry wire format, so it can be ingested by Jaeger, Grafana Tempo, Honeycomb, or any OTEL-compatible backend. No OpenTelemetry SDK required.

```bash
agent-replay export trace.jsonl --format otlp            # writes trace.otlp.json
agent-replay export trace.jsonl --format otlp -o out.json
```

Or programmatically:

```python
from agent_replay import Trace, to_otlp, export_otlp

trace = Trace.load("trace.jsonl")
doc = to_otlp(trace)                      # OTLP dict, ready to POST to a collector
export_otlp(trace, "trace.otlp.json")     # or write straight to a file
```

Mapping details:

- Spans become OTLP spans (ids padded to OTLP lengths, parent links preserved)
- Events become OTLP span events with typed attributes
- Trace and span metadata become resource and span attributes
- Spans containing an `error` event get `STATUS_CODE_ERROR`
- The original agent-replay trace id is kept in the `agent_replay.trace_id` resource attribute

Send the result to a local collector with plain curl:

```bash
curl -X POST http://localhost:4318/v1/traces \
  -H "Content-Type: application/json" \
  --data-binary @trace.otlp.json
```

## Configuration

### Trace Format

Traces are stored as JSONL files. Each line is a JSON object:

- Line 1: Trace header (metadata, trace ID, name)
- Lines 2+: Span records with nested events

```jsonl
{"type": "trace_header", "trace_id": "abc123", "name": "my-agent", ...}
{"type": "span", "name": "planning", "events": [...], ...}
{"type": "span", "name": "tool-use", "events": [...], ...}
```

### Programmatic Access

```python
from agent_replay import Trace

trace = Trace.load("trace.jsonl")
print(f"Spans: {len(trace.spans)}")
print(f"Events: {trace.event_count}")
print(f"Duration: {trace.duration:.3f}s")

for span in trace.spans:
    for event in span.events:
        print(event.event_type, event.data)
```

## Development

```bash
git clone https://github.com/manasvardhan/agent-replay.git
cd agent-replay
pip install -e ".[dev]"
pytest
```

## License

MIT License. See [LICENSE](LICENSE).
