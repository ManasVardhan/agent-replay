"""Rich terminal viewer for agent traces."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .diff import DiffResult, Divergence
from .replay import ReplayEngine
from .trace import EventType, Span, Trace

# Color mapping for event types
EVENT_COLORS: dict[EventType, str] = {
    EventType.LLM_REQUEST: "cyan",
    EventType.LLM_RESPONSE: "green",
    EventType.TOOL_CALL: "yellow",
    EventType.TOOL_RESULT: "blue",
    EventType.DECISION: "magenta",
    EventType.STATE_CHANGE: "white",
    EventType.ERROR: "red",
    EventType.LOG: "dim",
}

EVENT_ICONS: dict[EventType, str] = {
    EventType.LLM_REQUEST: "🧠",
    EventType.LLM_RESPONSE: "💬",
    EventType.TOOL_CALL: "🔧",
    EventType.TOOL_RESULT: "📦",
    EventType.DECISION: "🔀",
    EventType.STATE_CHANGE: "📝",
    EventType.ERROR: "❌",
    EventType.LOG: "📋",
}


class TraceViewer:
    """Rich terminal viewer for traces."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def show_trace(self, trace: Trace) -> None:
        """Display a full trace overview."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]{trace.name}[/bold]\n"
            f"ID: [dim]{trace.trace_id}[/dim]\n"
            f"Spans: {len(trace.spans)} | Events: {trace.event_count}\n"
            f"Duration: {trace.duration:.3f}s" if trace.duration else "Duration: running",
            title="[bold cyan]Agent Trace[/bold cyan]",
            border_style="cyan",
        ))

        for span in trace.spans:
            self._show_span(span)

    def _show_span(self, span: Span, indent: int = 0) -> None:
        prefix = "  " * indent
        duration = f" ({span.duration:.3f}s)" if span.duration else ""
        self.console.print(f"\n{prefix}[bold yellow]>>> {span.name}[/bold yellow]{duration}")

        for event in span.events:
            icon = EVENT_ICONS.get(event.event_type, "?")
            color = EVENT_COLORS.get(event.event_type, "white")
            label = event.event_type.value.replace("_", " ").upper()
            self.console.print(f"{prefix}  {icon} [{color}]{label}[/{color}]", end="")

            # Show key data inline
            data = event.data
            if event.event_type == EventType.LLM_REQUEST:
                model = data.get("model", "")
                n_msgs = len(data.get("messages", []))
                self.console.print(f" [dim]model={model} messages={n_msgs}[/dim]")
            elif event.event_type == EventType.LLM_RESPONSE:
                content = data.get("content", "")
                preview = content[:80] + "..." if len(content) > 80 else content
                tokens = data.get("tokens")
                tok_str = f" [dim]({tokens} tokens)[/dim]" if tokens else ""
                self.console.print(f' "{preview}"{tok_str}')
            elif event.event_type == EventType.TOOL_CALL:
                tool = data.get("tool", "")
                args = data.get("args", {})
                self.console.print(f" [bold]{tool}[/bold]({args})")
            elif event.event_type == EventType.TOOL_RESULT:
                tool = data.get("tool", "")
                result = str(data.get("result", ""))
                preview = result[:60] + "..." if len(result) > 60 else result
                self.console.print(f" [bold]{tool}[/bold] -> {preview}")
            elif event.event_type == EventType.DECISION:
                desc = data.get("description", "")
                choice = data.get("choice", "")
                self.console.print(f" {desc} -> [bold]{choice}[/bold]")
            elif event.event_type == EventType.ERROR:
                msg = data.get("message", "")
                self.console.print(f" [red]{msg}[/red]")
            else:
                msg = data.get("message", str(data)[:80])
                self.console.print(f" {msg}")

    def show_tree(self, trace: Trace) -> None:
        """Show trace as a tree structure."""
        tree = Tree(f"[bold cyan]{trace.name}[/bold cyan] ({trace.trace_id})")
        span_nodes: dict[str, Tree] = {}

        for span in trace.spans:
            parent = span_nodes.get(span.parent_id, tree) if span.parent_id else tree  # type: ignore
            duration = f" [{span.duration:.3f}s]" if span.duration else ""
            node = parent.add(f"[yellow]{span.name}[/yellow]{duration}")
            span_nodes[span.span_id] = node

            for event in span.events:
                icon = EVENT_ICONS.get(event.event_type, "?")
                color = EVENT_COLORS.get(event.event_type, "white")
                node.add(f"{icon} [{color}]{event.event_type.value}[/{color}]")

        self.console.print(tree)

    def show_diff(self, diff_result: DiffResult) -> None:
        """Display a diff result."""
        self.console.print()
        style = "green" if diff_result.identical else "red"
        self.console.print(Panel(
            f"Trace A: [dim]{diff_result.trace_a_id}[/dim]\n"
            f"Trace B: [dim]{diff_result.trace_b_id}[/dim]\n"
            f"[{style}]{diff_result.summary}[/{style}]",
            title="[bold]Trace Diff[/bold]",
            border_style=style,
        ))

        if diff_result.divergences:
            table = Table(title="Divergences", show_lines=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Severity", width=10)
            table.add_column("Position", width=8)
            table.add_column("Description")

            severity_colors = {"critical": "red", "warning": "yellow", "info": "blue"}
            for i, div in enumerate(diff_result.divergences, 1):
                color = severity_colors.get(div.severity, "white")
                table.add_row(
                    str(i),
                    f"[{color}]{div.severity.upper()}[/{color}]",
                    str(div.position),
                    div.description,
                )

            self.console.print(table)

    def show_step(self, engine: ReplayEngine) -> None:
        """Show the current step in a replay."""
        result = engine.peek()
        if result is None:
            self.console.print("[dim]End of trace[/dim]")
            return

        span, event = result
        pos = engine.position
        total = engine.total_steps
        icon = EVENT_ICONS.get(event.event_type, "?")
        color = EVENT_COLORS.get(event.event_type, "white")

        self.console.print(
            f"[dim][{pos + 1}/{total}][/dim] "
            f"[yellow]{span.name}[/yellow] "
            f"{icon} [{color}]{event.event_type.value}[/{color}]"
        )
