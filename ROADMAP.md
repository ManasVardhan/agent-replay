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

### 🌐 Trace Server
A tiny local web server that lists saved traces and renders the HTML timeline, diff, and cost views in the browser, so a team can browse a directory of agent runs without exporting files one by one. Shipped as the `serve` CLI command (`--host`, `--port`, `--price`, `--verbose`) backed by `TraceServer` / `discover_traces()`: an index of every `.jsonl` trace with links to on-the-fly timeline, cost, and side-by-side diff pages, a `/api/traces` JSON listing, directory re-scan on every request, and standard library only.

### 🔎 Trace Search Across Runs
Find a specific tool call or error across many recorded runs. Shipped in v0.2.0: `search` now accepts a directory target (`agent-replay search traces/ "rate limit"`) and scans every trace it contains, grouping matches by file, with `--json-output` for scripting; the trace server gained a search box on the index page, a `/search?q=` results view, and an `/api/search?q=` JSON endpoint; `search_trace()` / `search_directory()` and the `SearchMatch` dataclass expose the same search in Python.

### 🏷️ Trace Tagging and Filtering
Attach tags to traces at record time so large trace directories stay navigable. Shipped in v0.3.0: `Recorder("run", tags=["prod", "checkout"])` and `record_trace(..., tags=[...])` save normalized tags in the trace header (older traces load with no tags), `agent-replay info` shows them, `agent-replay search --tag` scopes cross-run search, `discover_traces(dir, tag=...)`, `search_directory(dir, q, tag=...)`, and `list_tags(dir)` expose the same filtering in Python, and the trace server renders clickable tag chips on the index with `/?tag=`, tag-aware search, and tags plus `?tag=` filtering on `/api/traces` and `/api/search`.

---

## v0.4 (Planned)

### 🔁 Live Trace Following
Tail a trace file as an agent writes it (`agent-replay follow trace.jsonl`), streaming new spans and events to the terminal as they happen, so long-running agents can be watched without waiting for the run to finish.

---

Have ideas? Open an issue or start a discussion!
