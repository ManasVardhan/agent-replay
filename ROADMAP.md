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

### 🦙 LlamaIndex Integration
Native hooks for LlamaIndex so query and agent traces are captured automatically, matching the LangChain integration. Shipped as `AgentReplayLlamaIndexHandler` in `agent_replay.integrations.llamaindex` (optional `[llamaindex]` extra): query, retrieval, synthesis, sub-question, tree, and agent-step events become nested spans; LLM requests/responses with token usage, tool calls/results, embeddings, and exceptions become events; `finish(path)` saves a trace compatible with every CLI command.

### 💰 Trace Cost Analytics
Per-span and per-model token and cost breakdowns computed from recorded LLM events. Shipped as the `cost` CLI command backed by `analyze_trace()` / `CostReport`: exact input/output pricing for `token_usage` dicts (LangChain and LlamaIndex traces), average-rate estimates for bare token totals (marked `~`), a built-in per-1M-token table for common OpenAI, Anthropic, and Google models with prefix matching for versioned names, repeatable `--price MODEL=INPUT:OUTPUT` overrides, `--json-output`, and clear reporting of unpriced models and responses without usage data.

---

## v0.2 (Planned)

### 🌐 Trace Server
A tiny local web server (`agent-replay serve traces/`) that lists saved traces and renders the HTML timeline, diff, and cost views in the browser, so a team can browse a directory of agent runs without exporting files one by one.

---

Have ideas? Open an issue or start a discussion!
