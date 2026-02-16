"""Example: Recording a simple agent loop and replaying it.

Run:
    python examples/basic_agent.py
"""

from agent_replay import Recorder, ReplayEngine, Trace, export_html


def simulate_agent():
    """Simulate an agent that answers a question using tools."""
    with Recorder("research-agent", output_path="example_trace.jsonl") as rec:

        # Step 1: Receive user query
        with rec.span("receive-query"):
            rec.state_change("task", old=None, new="What is the capital of France?")
            rec.log("Received user query")

        # Step 2: LLM decides to use a tool
        with rec.span("llm-planning"):
            rec.llm_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
            )
            rec.llm_response(
                content="I'll search for this information.",
                tokens=12,
            )
            rec.decision(
                description="Choose next action",
                choice="use_tool:search",
                alternatives=["direct_answer", "ask_clarification"],
            )

        # Step 3: Tool execution
        with rec.span("tool-execution"):
            rec.tool_call("search", {"query": "capital of France"})
            rec.tool_result("search", {
                "results": [
                    {"title": "Paris - Capital of France", "snippet": "Paris is the capital..."}
                ]
            })

        # Step 4: Final response
        with rec.span("final-response"):
            rec.llm_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "I'll search for this information."},
                    {"role": "tool", "content": "Paris is the capital..."},
                ],
            )
            rec.llm_response(
                content="The capital of France is Paris.",
                tokens=8,
            )
            rec.state_change("status", old="working", new="complete")

    return rec.trace


def replay_trace():
    """Load and replay the trace."""
    engine = ReplayEngine.from_file("example_trace.jsonl")
    print(f"\nReplaying: {engine.trace.name}")
    print(f"Total steps: {engine.total_steps}\n")

    while engine.has_next():
        span, event = engine.step()
        print(f"  [{span.name}] {event.event_type.value}: {event.data}")


if __name__ == "__main__":
    print("=== Recording Agent Run ===")
    trace = simulate_agent()
    print(f"Trace saved: {trace.trace_id} ({trace.event_count} events)")

    print("\n=== Replaying Trace ===")
    replay_trace()

    print("\n=== Exporting HTML ===")
    export_html(Trace.load("example_trace.jsonl"), "example_trace.html")
    print("Exported to example_trace.html")
