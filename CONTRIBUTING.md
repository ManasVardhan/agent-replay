# Contributing to agent-replay

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/ManasVardhan/agent-replay.git
cd agent-replay
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
pytest --cov=agent_replay --cov-report=term-missing
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Request Guidelines

1. Fork the repo and create a feature branch
2. Write tests for any new functionality
3. Make sure all tests pass (`pytest -v`)
4. Run `ruff check` and fix any issues
5. Keep commits focused and descriptive
6. Open a PR against `main`

## What to Work On

Check the [Issues](https://github.com/ManasVardhan/agent-replay/issues) page for open tasks. Issues labeled `good-first-issue` are great starting points.

Some areas that could use help:

- **More export formats** (CSV, Parquet, OpenTelemetry)
- **Framework integrations** (LangChain, CrewAI, AutoGen)
- **Trace filtering** (by event type, time range, span name)
- **Statistics** (token usage summaries, latency breakdowns)
- **Async support** (async context managers for recording)

## Architecture

```
src/agent_replay/
    __init__.py      # Public API exports
    cli.py           # Click CLI commands
    diff.py          # Trace comparison engine
    exporters.py     # JSON and HTML export
    recorder.py      # Recorder context manager and decorator
    replay.py        # Step-through replay engine
    trace.py         # Core data model (Event, Span, Trace)
    viewer.py        # Rich terminal rendering
```

## Questions?

Open an issue or reach out to [@vardhan_manas](https://twitter.com/vardhan_manas).
