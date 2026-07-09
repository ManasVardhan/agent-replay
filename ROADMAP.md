# Roadmap - agent-replay

## Shipped

### ▶️ Streaming Replay
Replay agent traces step-by-step in the terminal with configurable playback speed - great for demos, debugging, and onboarding. Shipped as the `play` CLI command (`--speed`, `--max-delay`, `--no-delay`) backed by `ReplayEngine.playback_plan()`.

---

## v0.2 (Planned)

### 🔗 LangChain / LlamaIndex Integration
Native callbacks and hooks for LangChain and LlamaIndex so agent traces are captured automatically without manual instrumentation.

### 🔀 Trace Comparison UI
Side-by-side diff view to compare two agent runs, highlighting where decisions diverged. Useful for regression testing prompt changes.

### 📤 OpenTelemetry Export
Export traces in OpenTelemetry format for integration with Jaeger, Grafana Tempo, or any OTEL-compatible backend.

---

Have ideas? Open an issue or start a discussion!
