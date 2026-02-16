# Getting Started with agent-replay

A step-by-step guide to get up and running from scratch.

## Prerequisites

You need **Python 3.10 or newer** installed on your machine.

**Check if you have Python:**
```bash
python3 --version
```
If you see `Python 3.10.x` or higher, you're good. If not, download it from [python.org](https://www.python.org/downloads/).

## Step 1: Clone the repository

```bash
git clone https://github.com/ManasVardhan/agent-replay.git
cd agent-replay
```

## Step 2: Create a virtual environment

```bash
python3 -m venv venv
```

**Activate it:**

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

## Step 3: Install the package

```bash
pip install -e ".[dev]"
```

## Step 4: Run the tests

```bash
pytest tests/ -v
```

All tests should pass. This confirms everything is installed correctly.

## Step 5: Try it out

### 5a. Run the example

```bash
python examples/basic_agent.py
```

This simulates a simple agent run, records it as a trace, and shows you the output.

### 5b. Explore the CLI

```bash
agent-replay --help
```

### 5c. Write your own trace

Create a file called `test_it.py`:

```python
from agent_replay import Recorder, Trace

# Record an agent session
with Recorder() as rec:
    # Log an LLM call
    rec.llm_call(
        model="gpt-4o",
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        tokens_in=10,
        tokens_out=8
    )

    # Log a tool call
    rec.tool_call(
        name="web_search",
        input={"query": "Paris population"},
        output={"result": "2.1 million"}
    )

    # Log a decision
    rec.decision("Provide final answer with population data")

# Get the trace
trace = rec.trace

print(f"Trace ID: {trace.trace_id}")
print(f"Events recorded: {len(trace.events)}")
print(f"Duration: {trace.duration_ms:.0f}ms")

# Save it
trace.save("my_trace.jsonl")
print("Trace saved to my_trace.jsonl")
```

Run it:
```bash
python test_it.py
```

### 5d. Replay a trace

After saving a trace, view it:
```bash
agent-replay view my_trace.jsonl
```

### 5e. Export to HTML

```bash
agent-replay export my_trace.jsonl -o trace_report.html
```

Open `trace_report.html` in your browser to see a visual timeline.

## Step 6: Run the linter (optional)

```bash
ruff check src/ tests/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3: command not found` | Install Python from [python.org](https://www.python.org/downloads/) |
| `pip: command not found` | Try `python3 -m pip` instead of `pip` |
| `No module named agent_replay` | Make sure you ran `pip install -e ".[dev]"` with the venv activated |
| Tests fail | Make sure you're on the latest `main` branch: `git pull origin main` |

## What's next?

- Read the full [README](README.md) for the diff tool, rich terminal viewer, and advanced recording options
- Check `examples/` for more patterns
- Try recording your own agent workflows
