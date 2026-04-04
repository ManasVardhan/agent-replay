# AGENTS.md - agent-replay

## Overview
- Record, replay, and diff AI agent execution traces. Captures every LLM call, tool use, decision point, and state change during agent execution, then lets you replay step-by-step or diff two runs to find exactly where behavior diverged.
- For developers debugging AI agents who need full visibility into agent behavior across runs.
- Core value: structured tracing with spans and events, interactive step-through replay, automated divergence detection, and rich terminal + HTML output.

## Architecture

```
Agent Run --> Recorder --> Trace File (.jsonl) --> Replay Viewer
                                                --> Diff Tool
                                                --> HTML Export
                                                --> Stats / Search
```

```
+------------------------------------------------------------+
|  Your Agent Code                                           |
|  +------------------------------------------------------+  |
|  |  with Recorder("my-agent") as rec:                   |  |
|  |      with rec.span("planning"):                      |  |
|  |          rec.llm_request(model="gpt-4", ...)         |  |
|  |          rec.llm_response(content="...", tokens=42)  |  |
|  |      with rec.span("tool-use"):                      |  |
|  |          rec.tool_call("search", {"q": "..."})       |  |
|  +------------------------------------------------------+  |
+---------------------------+--------------------------------+
                            |
                            v
                     trace.jsonl
                            |
               +------------+------------+
               v            v            v
          agent-replay  agent-replay  agent-replay
            show          replay        diff
```

**Data flow:**
1. `Recorder` wraps agent code as a context manager or decorator
2. Events are added to `Span` objects within a `Trace`
3. On exit, the trace is serialized to JSONL (one header line + one line per span)
4. CLI commands load the JSONL and provide show, replay, diff, export, stats, search

## Directory Structure

```
agent-replay/
  .github/workflows/ci.yml        -- CI: test with coverage on Python 3.10-3.12
  src/agent_replay/
    __init__.py                    -- Public API re-exports, __version__ = "0.1.1"
    __main__.py                    -- python -m agent_replay entry
    trace.py                       -- Trace, Span, Event, EventType data models
    recorder.py                    -- Recorder context manager + record_trace decorator
    replay.py                      -- ReplayEngine for step-through navigation
    diff.py                        -- diff_traces(), DiffResult, Divergence
    exporters.py                   -- export_json(), export_html() (self-contained dark-mode timeline)
    viewer.py                      -- TraceViewer: Rich terminal rendering (show, tree, diff, step)
    cli.py                         -- Click CLI: show, replay, diff, export, info, stats, search
  examples/
    basic_agent.py                 -- Example agent with recorder
  tests/                           -- 166 tests across 12 test files
    test_trace.py                  -- Trace/Span/Event data model tests
    test_recorder.py               -- Recorder context manager tests
    test_recorder_extended.py      -- Extended recorder coverage
    test_replay.py                 -- ReplayEngine tests
    test_diff.py                   -- Diff algorithm tests
    test_cli.py                    -- CLI command tests
    test_cli_replay.py             -- Replay CLI tests
    test_cli_stats.py              -- Stats CLI tests
    test_exporters.py              -- JSON/HTML export tests
    test_viewer.py                 -- Viewer rendering tests
    test_edge_cases.py             -- Edge case coverage
    test_trace_filter.py           -- Trace filter/search tests
    test_stats_search.py           -- Stats and search tests
  pyproject.toml                   -- Hatchling build, metadata
  README.md                        -- Full docs
  ROADMAP.md                       -- v0.2 plans
  CONTRIBUTING.md                  -- Contribution guidelines
  GETTING_STARTED.md               -- Quick start guide
  LICENSE                          -- MIT
```

## Core Concepts

- **Trace**: Top-level container. Has a `trace_id`, `name`, ordered list of `Span` objects, start/end times, metadata. Serializes to JSONL.
- **Span**: Named execution phase (e.g. "planning", "tool-use"). Contains a list of `Event` objects. Spans can nest via `parent_id`. Has start/end times and metadata.
- **Event**: A single thing that happened. Has `EventType`, timestamp, data dict, event_id.
- **EventType**: Enum with values: `llm_request`, `llm_response`, `tool_call`, `tool_result`, `decision`, `state_change`, `error`, `log`.
- **Recorder**: Context manager that builds a Trace. Provides convenience methods (`llm_request`, `tool_call`, `decision`, etc.). Supports nested spans via `with rec.span("name")`.
- **ReplayEngine**: Flattens all span/event pairs into a sorted list, provides `step()`, `step_back()`, `jump()`, `peek()`, `search()`, `reset()`.
- **DiffResult / Divergence**: Compares two traces event-by-event. Classifies divergences as critical (different event types, different tool calls, different decisions) or informational (different LLM response content).

## API Reference

### Recorder
```python
class Recorder:
    def __init__(self, name="agent-run", metadata=None, output_path=None)
    def __enter__(self) -> Recorder
    def __exit__(self, *exc) -> None
    def span(self, name, metadata=None) -> ContextManager[Span]  # nestable
    def llm_request(self, model="", messages=None, **kwargs) -> None
    def llm_response(self, content="", tokens=None, **kwargs) -> None
    def tool_call(self, tool, args=None, **kwargs) -> None
    def tool_result(self, tool, result=None, **kwargs) -> None
    def decision(self, description, choice="", **kwargs) -> None
    def state_change(self, key, old=None, new=None, **kwargs) -> None
    def log(self, message, level="info", **kwargs) -> None
    def error(self, message, exception=None, **kwargs) -> None
    def finish(self) -> Trace
```

### record_trace (decorator)
```python
@record_trace("my-agent", output_path="trace.jsonl")
def run_agent(task: str, recorder: Recorder = None): ...
```

### Trace
```python
class Trace:
    def save(self, path) -> Path           # JSONL output
    @classmethod def load(cls, path) -> Trace  # JSONL input
    def all_events() -> list[Event]        # sorted by timestamp
    def filter_events(*event_types) -> list[Event]
    def event_type_counts() -> dict[str, int]
    def to_dict() -> dict
    @classmethod def from_dict(cls, d) -> Trace
```

### ReplayEngine
```python
class ReplayEngine:
    @classmethod def from_file(cls, path) -> ReplayEngine
    def step() -> tuple[Span, Event] | None
    def step_back() -> tuple[Span, Event] | None
    def jump(position) -> tuple[Span, Event] | None
    def peek() -> tuple[Span, Event] | None
    def search(query) -> list[int]         # positions matching query
    def reset() -> None
    @property total_steps -> int
    @property position -> int
    def has_next() -> bool
    def has_prev() -> bool
```

### Diff
```python
def diff_traces(trace_a: Trace, trace_b: Trace) -> DiffResult
```

### Exporters
```python
def export_json(trace, path) -> Path
def export_html(trace, path) -> Path  # self-contained dark-mode HTML timeline
```

## CLI Commands

```bash
# Show a trace file (flat or tree view)
agent-replay show trace.jsonl
agent-replay show trace.jsonl --tree

# Interactive step-through replay
agent-replay replay trace.jsonl
# Commands during replay: n/next, p/prev, j N/jump N, q/quit

# Diff two traces
agent-replay diff trace_a.jsonl trace_b.jsonl

# Export to JSON or HTML
agent-replay export trace.jsonl --format html -o timeline.html
agent-replay export trace.jsonl --format json -o trace.json

# Show summary info
agent-replay info trace.jsonl

# Detailed stats (event counts, models used, tools used, span durations)
agent-replay stats trace.jsonl
agent-replay stats trace.jsonl --json-output

# Search events by query string
agent-replay search trace.jsonl "search"

# Version
agent-replay --version
```

## Configuration

- **Trace format**: JSONL. Line 1 is trace header, subsequent lines are span records.
- **Output path**: Set via `Recorder(output_path="trace.jsonl")` or `record_trace(output_path=...)`.
- **No config files or env vars needed.**

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=agent_replay -v
```

- **166 tests** across 12 test files
- All tests are pure unit tests, no external dependencies
- Located in `tests/`

## Dependencies

- **rich>=13.0**: Terminal rendering (panels, tables, trees, colors)
- **click>=8.0**: CLI framework
- **Python >=3.10**

## CI/CD

- **GitHub Actions** (`.github/workflows/ci.yml`)
- Matrix: Python 3.10, 3.11, 3.12
- Steps: install, pytest with coverage
- Triggers: push/PR to main

## Current Status

- **Version**: 0.1.1
- **Published on PyPI**: yes (`pip install agent-trace-replay`)
- **What works**: Full recording (context manager + decorator), JSONL persistence, interactive replay, trace diffing with severity classification, HTML export (dark-mode timeline), Rich terminal viewer (flat + tree), stats, search, info commands
- **Known limitations**: No LangChain/LlamaIndex auto-instrumentation yet. No OpenTelemetry export. Replay is terminal-only (no web UI).
- **Roadmap (v0.2)**: LangChain/LlamaIndex integration, streaming replay, trace comparison UI, OpenTelemetry export

## Development Guide

```bash
git clone https://github.com/manasvardhan/agent-replay.git
cd agent-replay
pip install -e ".[dev]"
pytest
```

- **Build system**: Hatchling
- **Source layout**: `src/agent_replay/`
- **Adding a new EventType**: Add to `EventType` enum in `trace.py`, add convenience method in `recorder.py`, update viewer icons/colors in `viewer.py`
- **Adding a new CLI command**: Add `@cli.command()` in `cli.py`
- **Adding a new export format**: Add function in `exporters.py`, wire into CLI `export` command
- **Code style**: Ruff, line length 100, target Python 3.10

## Git Conventions

- **Branch**: main
- **Commits**: Imperative style ("Add feature X", "Fix bug Y")
- Never use em dashes in commit messages or docs

## Context

- **Author**: Manas Vardhan (ManasVardhan on GitHub)
- **Part of**: A suite of AI agent tooling
- **Related repos**: llm-cost-guardian (cost tracking), agent-sentry (crash reporting), llm-shelter (safety guardrails), promptdiff (prompt versioning), mcp-forge (MCP server scaffolding), bench-my-llm (benchmarking)
- **PyPI package**: `agent-trace-replay` (note: PyPI name differs from repo name)
- **Import as**: `agent_replay`
