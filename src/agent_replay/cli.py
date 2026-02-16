"""CLI interface for agent-replay."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from .diff import diff_traces
from .exporters import export_html, export_json
from .replay import ReplayEngine
from .trace import Trace
from .viewer import TraceViewer

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agent-replay")
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
