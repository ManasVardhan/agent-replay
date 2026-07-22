# Roadmap - agent-replay

## Shipped

### ▶️ Streaming Replay
Replay agent traces step-by-step in the terminal with configurable playback speed - great for demos, debugging, and onboarding. Shipped as the `play` CLI command (`--speed`, `--max-delay`, `--no-delay`) backed by `ReplayEngine.playback_plan()`.

### 📤 OpenTelemetry Export
Export traces in OpenTelemetry OTLP/JSON format for integration with Jaeger, Grafana Tempo, or any OTEL-compatible backend. Shipped as `export --format otlp` backed by `to_otlp()` / `export_otlp()`, with no OTEL SDK dependency.

### 🔀 Trace Comparison UI
Side-by-side diff view to compare two agent runs, highlighting where decisions diverged. Useful for regression testing prompt changes. Shipped as `diff --html report.html` (plus `--title` and `--json-output`) backed by `render_diff_html()` / `export_diff_html()`: a self-contained HTML report with aligned event columns and severity-highlighted divergences.

### 🔗 LangChain Integration
Native callbacks so LangChain agent traces are captured automatically without manual instrumentation. Shipped as `AgentReplayCallbackHandler` in `agent_replay.integrations.langchain` (optional `[langchain]` extra): chain runs become nested spans, LLM requests/responses with token usage, tool calls/results, agent decisions, and errors become events, and `finish(path)` saves a trace compatible with every CLI command.

---

## v0.2 (Planned)

### 🦙 LlamaIndex Integration
Native hooks for LlamaIndex so query and agent traces are captured automatically, matching the LangChain integration.

---

Have ideas? Open an issue or start a discussion!
