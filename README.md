
# agent-replay

> **New here?** Start with the [Getting Started Guide](GETTING_STARTED.md).

[![PyPI version](https://img.shields.io/pypi/v/agent-trace-replay.svg)](https://pypi.org/project/agent-trace-replay/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/manasvardhan/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/manasvardhan/agent-replay/actions)

**AI agents are black boxes. agent-replay makes them transparent.**

Record every LLM call, tool use, decision point, and state change during agent execution. Replay them step-by-step. Differentiate between two runs to find exactly where behavior diverged.

## Features

- 🎬 **Record** agent runs with a simple context manager or decorator
- 🔗 **LangChain integration** - capture traces automatically with a callback handler
- 🦙 **LlamaIndex integration** - record queries, retrievals, and agent steps automatically
- ⏯️ **Replay** traces step-by-step in the terminal
- 🔴 **Live follow** - `agent-replay follow trace.jsonl` tails a running agent's trace and streams new spans as they land
- 🔍 **Diff** two traces to find divergence points, with side-by-side HTML comparison reports
- 🌳 **Tree view** of nested spans and events
- 💰 **Cost analytics** - per-model and per-span token and cost breakdowns with a `cost` command
- 📊 **HTML export** with a self-contained dark-mode timeline
- 🌐 **Trace server** - `agent-replay serve traces/` to browse timeline, diff, and cost views in the browser
- 🌊 **Live streaming** - the server's live view pushes new spans to the browser over server-sent events while an agent runs
- 🏷️ **Trace tagging** - tag runs at record time, then filter the server index and cross-run search by tag
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

with Recorder("my-agent", output_path="trace.jsonl", tags=["prod", "checkout"]) as rec:
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

### LlamaIndex Integration

Register the handler with LlamaIndex's callback manager and every query,
retrieval, synthesis, sub-question, and agent step is recorded as a
nested span. LLM requests/responses (with token usage), tool calls and
results, embeddings, and exceptions become events on the span they
belong to.

```bash
pip install "agent-trace-replay[llamaindex]"  # pulls in llama-index-core
```

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from agent_replay.integrations.llamaindex import AgentReplayLlamaIndexHandler

handler = AgentReplayLlamaIndexHandler("rag-pipeline")
Settings.callback_manager = CallbackManager([handler])

query_engine = index.as_query_engine()
response = query_engine.query("What changed in Q3?")

trace = handler.finish("trace.jsonl")  # close spans and save
```

Like the LangChain handler, it uses the same defensive payload handling,
so odd shapes never break a run, and the saved trace works with every
CLI command.

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

### Search Across Runs

Find a tool call, error, or response in one trace, or across every trace in
a directory at once:

```bash
agent-replay search trace.jsonl "rate limit"          # one trace
agent-replay search traces/ "rate limit"              # every trace in a directory
agent-replay search traces/ "rate limit" --tag prod   # only traces tagged prod
agent-replay search traces/ "rate limit" --json-output  # machine-readable
```

Directory results are grouped by file, and the JSON output includes the file,
trace name, span, event type, event position, and a data preview for each
match. The same search is available in Python via `search_trace()` and
`search_directory()`.

### Tagging Runs

Tag traces at record time so large trace directories stay navigable:

```python
from agent_replay import Recorder, discover_traces, list_tags

with Recorder("checkout-run", output_path="traces/run1.jsonl",
              tags=["prod", "checkout"]) as rec:
    ...

print(list_tags("traces/"))                       # every tag in a directory
prod = discover_traces("traces/", tag="prod")     # only traces tagged prod
```

Tags are saved in the trace header, shown by `agent-replay info`, filter
`agent-replay search --tag`, and drive the trace server's tag filter. Traces
recorded before tags existed simply load with no tags.

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

### Live Trace Following

Watch a long-running agent without waiting for it to finish. `follow` reads
the spans already in a trace file, then tails it and streams each new span
and its events to the terminal as the agent appends them:

```bash
# Show existing spans, then stream new ones as they are written
agent-replay follow trace.jsonl

# Skip existing spans and only show new activity
agent-replay follow trace.jsonl --from-end

# Check every 0.2s and stop after 30 idle seconds
agent-replay follow trace.jsonl --poll-interval 0.2 --timeout 30
```

```
Following: trace.jsonl (from start)
Press Ctrl+C to stop.
Trace: live-agent tags: demo

>>> plan (0.412s)
  🧠 llm_request model=gpt-4o messages=0
  🔧 tool_call search({'q': 'weather SF'})
```

The follower tracks a byte offset and buffers partial trailing lines, so a
span still being written is not shown until its line is complete. `--timeout`
defaults to 0 (follow until Ctrl+C). Programmatic access via `TraceFollower`:
each `poll()` returns typed `FollowUpdate` records (header, span, or
malformed) that appeared since the previous call.

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

## Cost Analytics

See what a trace cost you, per model and per span:

```bash
agent-replay cost trace.jsonl
```

```
Cost: research-agent
  LLM calls:    4
  Total tokens: 16,300
  Total cost:   $0.050125

              Cost by Model
Model        Calls  Prompt  Completion  Tokens        Cost
gpt-4o           2  10,000       2,400  12,400   $0.049000
gpt-4o-mini      1       0           0   3,000  ~$0.001125
local-llama      1       0           0     900         n/a

     Cost by Span
Span       Calls  Tokens       Cost
plan           1   6,200  $0.024500
research       1   6,200  $0.024500
summarize      1   3,000  $0.001125
local          1     900        n/a
```

Costs come from token usage recorded in `llm_response` events. Traces from the
LangChain and LlamaIndex integrations carry full `token_usage` dicts and get
exact input/output pricing; hand-recorded traces with only a `tokens` total
get an estimate at the average of the model's input and output rates (marked
with `~`). The built-in table covers common OpenAI, Anthropic, and Google
models, and versioned names like `gpt-4o-2024-08-06` match by prefix.

Price local or fine-tuned models with repeatable `--price` overrides (USD per
1M tokens), and use `--json-output` for scripting:

```bash
agent-replay cost trace.jsonl --price my-llama=0.10:0.25
agent-replay cost trace.jsonl --json-output
```

Programmatic access:

```python
from agent_replay import Trace, analyze_trace

report = analyze_trace(Trace.load("trace.jsonl"))
print(report.total_cost_usd, report.total_tokens)
print(report.by_model())  # sorted by cost, descending
print(report.by_span())
```

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

## Trace Server

Browse a whole directory of traces in the browser without exporting files one by one:

```bash
agent-replay serve traces/                  # http://127.0.0.1:8600/
agent-replay serve traces/ --port 9000      # custom port
agent-replay serve traces/ --price my-model=1.50:6.00   # pricing for cost views
```

The index page lists every `.jsonl` trace in the directory with tags, span, event, and duration summaries. From there each trace links to:

- **timeline** - the same dark-mode HTML timeline as `export --format html`, rendered on the fly
- **live** - a live view that streams new spans into the page over server-sent events while an agent is still writing the trace
- **cost** - per-model and per-span token and cost tables from the cost analyzer
- **diff** - side-by-side comparison of any two traces via `/diff?a=<file>&b=<file>`
- **search** - a search box on the index scans every trace at once via `/search?q=<query>`
- **tags** - click any tag (or use `/?tag=<tag>`) to filter the index; the search box keeps the active tag filter

A JSON listing is available at `/api/traces` (each entry includes its tags, `?tag=` filters), and cross-trace search results at `/api/search?q=<query>&tag=<tag>`, for scripting. The directory is re-scanned on every request, so traces saved while the server runs appear on refresh. Only files inside the served directory are ever read, and the server binds to 127.0.0.1 by default. Built entirely on the standard library, no extra dependencies.

### Live Streaming

The live view at `/trace/<file>/live` is the browser twin of the terminal `follow` command: spans already in the file are rendered server-side, then the page subscribes to `/trace/<file>/events` and appends each new span the moment the writer saves it, with a live/reconnecting status pill. The events endpoint is plain server-sent events, so it is also scriptable:

```bash
# Stream every span in the file, then new ones as they land
curl -N http://127.0.0.1:8600/trace/run.jsonl/events

# Only new activity, checked every 0.2s, closing after 30 idle seconds
curl -N "http://127.0.0.1:8600/trace/run.jsonl/events?from_end=1&poll=0.2&timeout=30"
```

Each SSE message carries an `event:` type (`header`, `span`, `malformed`, or `end` when an idle `?timeout=` expires) and a JSON `data:` payload; span payloads match `Span.to_dict()`. Malformed trailing lines are reported instead of crashing the stream, and a mid-write span is not parsed until its newline arrives, using the same tailing logic as `follow`.

Programmatic use:

```python
from agent_replay import TraceServer, discover_traces

print([i.name for i in discover_traces("traces/")])

server = TraceServer("traces/", port=0)   # port 0 picks a free port
server.serve_in_background()
print(server.url)
```

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
