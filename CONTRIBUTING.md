# Contributing to agent-replay

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/manasvardhan/agent-replay.git
cd agent-replay
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=agent_replay
```

## Code Style

- Python 3.10+ with type annotations
- Follow existing code patterns
- Keep functions focused and well-documented

## Pull Requests

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit a PR with a clear description

## Reporting Issues

Open an issue with:
- What you expected
- What actually happened
- Steps to reproduce
- Python version and OS
