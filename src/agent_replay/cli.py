"""CLI interface for agent-replay."""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import click
from rich.console import Console

from .diff import diff_traces
from .exporters import export_html, export_json
from .replay import ReplayEngine
from .trace import EventType, Trace
from .viewer import TraceViewer

console = Console()


@click.group()
@click.version_option(package_name="agent-trace-replay", prog_name="agent-replay")
def cli() -> None:
    """Record, replay, and debug AI agent execution traces."""


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("--tree", is_flag=True, help="Show trace as a tree structure.")
def show(trace_file: Path, tree: bool) -> None:
    """Display a trace file."""
    trace = Trace.load(trace_file)
    viewer = TraceViewer(console)
    if tree:
        viewer.show_tree(trace)
    else:
        viewer.show_trace(trace)


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
def replay(trace_file: Path) -> None:
    """Step through a trace interactively."""
    engine = ReplayEngine.from_file(trace_file)
    viewer = TraceViewer(console)

    console.print(f"[bold cyan]Replay: {engine.trace.name}[/bold cyan]")
    console.print(f"[dim]{engine.total_steps} events. Commands: (n)ext, (p)rev, (j)ump N, (q)uit[/dim]\n")

    while True:
        viewer.show_step(engine)
        try:
            cmd = click.prompt("", prompt_suffix="> ", default="n", show_default=False)
        except (EOFError, KeyboardInterrupt):
            break

        cmd = cmd.strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("n", "next", ""):
            engine.step()
        elif cmd in ("p", "prev", "back"):
            engine.step_back()
        elif cmd.startswith("j ") or cmd.startswith("jump "):
            try:
                pos = int(cmd.split()[-1]) - 1
                engine.jump(pos)
            except (ValueError, IndexError):
                console.print("[red]Usage: j <position>[/red]")
        else:
            console.print("[dim]Unknown command[/dim]")


@cli.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
def diff(trace_a: Path, trace_b: Path) -> None:
    """Compare two trace files and show divergences."""
    a = Trace.load(trace_a)
    b = Trace.load(trace_b)
    result = diff_traces(a, b)
    viewer = TraceViewer(console)
    viewer.show_diff(result)


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["json", "html"]), default="json", help="Export format.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path.")
def export(trace_file: Path, fmt: str, output: Path | None) -> None:
    """Export a trace to JSON or HTML."""
    trace = Trace.load(trace_file)
    if output is None:
        output = trace_file.with_suffix(f".{fmt}")

    if fmt == "json":
        export_json(trace, output)
    else:
        export_html(trace, output)

    console.print(f"[green]Exported to {output}[/green]")


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
def info(trace_file: Path) -> None:
    """Show summary information about a trace."""
    trace = Trace.load(trace_file)
    console.print(f"[bold]{trace.name}[/bold] ({trace.trace_id})")
    console.print(f"  Spans:    {len(trace.spans)}")
    console.print(f"  Events:   {trace.event_count}")
    duration = f"{trace.duration:.3f}s" if trace.duration else "N/A"
    console.print(f"  Duration: {duration}")
    console.print(f"  Metadata: {trace.metadata}")


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def stats(trace_file: Path, as_json: bool) -> None:
    """Show detailed statistics for a trace file."""
    trace = Trace.load(trace_file)
    events = trace.all_events()

    # Count by type
    type_counts = trace.event_type_counts()
    total_tokens = 0
    models_used: set[str] = set()
    tools_used: set[str] = set()

    for event in events:
        if event.event_type == EventType.LLM_REQUEST:
            model = event.data.get("model", "")
            if model:
                models_used.add(model)
        elif event.event_type == EventType.LLM_RESPONSE:
            tokens = event.data.get("tokens")
            if tokens and isinstance(tokens, int):
                total_tokens += tokens
        elif event.event_type == EventType.TOOL_CALL:
            tool = event.data.get("tool", "")
            if tool:
                tools_used.add(tool)

    span_durations: dict[str, float | None] = {}
    for span in trace.spans:
        span_durations[span.name] = span.duration

    if as_json:
        data = {
            "name": trace.name,
            "trace_id": trace.trace_id,
            "spans": len(trace.spans),
            "events": trace.event_count,
            "duration": trace.duration,
            "total_tokens": total_tokens,
            "models_used": sorted(models_used),
            "tools_used": sorted(tools_used),
            "event_type_counts": type_counts,
            "span_durations": span_durations,
        }
        console.print(json_mod.dumps(data, indent=2, default=str))
        return

    console.print(f"\n[bold cyan]Stats: {trace.name}[/bold cyan]")
    duration = f"{trace.duration:.3f}s" if trace.duration else "N/A"
    console.print(f"  Duration:     {duration}")
    console.print(f"  Spans:        {len(trace.spans)}")
    console.print(f"  Total events: {len(events)}")

    if type_counts:
        console.print("\n[bold]Event breakdown:[/bold]")
        for etype, count in sorted(type_counts.items()):
            console.print(f"  {etype:<20} {count:>5}")

    if total_tokens:
        console.print(f"\n  Total LLM tokens: {total_tokens:,}")

    if models_used:
        console.print(f"\n[bold]Models used:[/bold] {', '.join(sorted(models_used))}")

    if tools_used:
        console.print(f"[bold]Tools used:[/bold]  {', '.join(sorted(tools_used))}")

    if span_durations:
        console.print("\n[bold]Span durations:[/bold]")
        for name, dur in span_durations.items():
            dur_str = f"{dur:.3f}s" if dur is not None else "open"
            console.print(f"  {name:<30} {dur_str}")


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.argument("query")
def search(trace_file: Path, query: str) -> None:
    """Search for events matching a query string."""
    trace = Trace.load(trace_file)
    engine = ReplayEngine(trace)
    positions = engine.search(query)

    if not positions:
        console.print(f"[dim]No events matching '{query}'[/dim]")
        return

    console.print(f"[bold cyan]Found {len(positions)} match(es) for '{query}':[/bold cyan]\n")
    for pos in positions:
        pair = engine.jump(pos)
        if pair:
            span, event = pair
            console.print(
                f"  [{pos + 1}] [yellow]{span.name}[/yellow] "
                f"[dim]{event.event_type.value}[/dim]"
            )
            # Show relevant data preview
            data_str = str(event.data)
            if len(data_str) > 120:
                data_str = data_str[:120] + "..."
            console.print(f"      {data_str}")
