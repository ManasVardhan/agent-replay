"""CLI interface for agent-replay."""

from __future__ import annotations

import json as json_mod
import re
import time
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .diff import diff_traces
from .diff_html import export_diff_html
from .exporters import export_html, export_json
from .otel import export_otlp
from .redact import redact_trace
from .replay import ReplayEngine
from .trace import EventType, Trace
from .viewer import TraceViewer

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="agent-replay")
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
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--speed",
    "-s",
    type=float,
    default=1.0,
    show_default=True,
    help="Playback speed multiplier; 2.0 plays twice as fast.",
)
@click.option(
    "--max-delay",
    type=float,
    default=2.0,
    show_default=True,
    help="Cap each pause at this many seconds (after speed adjustment).",
)
@click.option("--no-delay", is_flag=True, help="Print every step instantly, no pauses.")
def play(trace_file: Path, speed: float, max_delay: float, no_delay: bool) -> None:
    """Replay a trace in the terminal with its original timing.

    Streams events in timestamp order, pausing between them to mirror
    the recorded pacing. Great for demos, debugging, and onboarding.
    Press Ctrl+C to stop playback early.
    """
    if speed <= 0:
        raise click.BadParameter("must be > 0", param_hint="--speed")
    if max_delay < 0:
        raise click.BadParameter("must be >= 0", param_hint="--max-delay")

    engine = ReplayEngine.from_file(trace_file)
    plan = engine.playback_plan(speed=speed, max_delay=max_delay)
    if not plan:
        console.print("[dim]Trace has no events to play.[/dim]")
        return

    viewer = TraceViewer(console)
    total = len(plan)
    trace_span = plan[-1].elapsed
    console.print(f"[bold cyan]Playing: {engine.trace.name}[/bold cyan]")
    speed_note = "instant" if no_delay else f"speed {speed:g}x, pauses capped at {max_delay:g}s"
    console.print(f"[dim]{total} events over {trace_span:.3f}s ({speed_note})[/dim]\n")

    played = 0
    try:
        for i, step in enumerate(plan, 1):
            if not no_delay and step.delay > 0:
                time.sleep(step.delay)
            viewer.show_play_step(i, total, step.elapsed, step.span, step.event)
            played += 1
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Playback interrupted at {played}/{total} events.[/yellow]")
        return

    console.print(f"\n[green]Played {played} events (trace span {trace_span:.3f}s).[/green]")


@cli.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--html",
    "html_output",
    type=click.Path(path_type=Path),
    help="Also write a side-by-side HTML comparison report to this file.",
)
@click.option("--title", default="Trace Comparison", show_default=True,
              help="Title for the HTML report.")
@click.option("--json-output", "as_json", is_flag=True, help="Output the diff as JSON.")
def diff(
    trace_a: Path,
    trace_b: Path,
    html_output: Path | None,
    title: str,
    as_json: bool,
) -> None:
    """Compare two trace files and show divergences.

    Use --html report.html to get a shareable side-by-side comparison
    view with every divergence highlighted.
    """
    a = Trace.load(trace_a)
    b = Trace.load(trace_b)
    result = diff_traces(a, b)

    if as_json:
        # click.echo, not console.print: rich soft-wraps long lines, which
        # would inject newlines inside JSON strings and corrupt the output.
        click.echo(json_mod.dumps(result.to_dict(), indent=2, default=str))
    else:
        viewer = TraceViewer(console)
        viewer.show_diff(result)

    if html_output is not None:
        export_diff_html(a, b, html_output, result=result, title=title)
        console.print(f"[green]HTML comparison written to {html_output}[/green]")


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "html", "otlp"]),
    default="json",
    help="Export format (otlp = OpenTelemetry OTLP/JSON).",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path.")
def export(trace_file: Path, fmt: str, output: Path | None) -> None:
    """Export a trace to JSON, HTML, or OpenTelemetry OTLP/JSON."""
    trace = Trace.load(trace_file)
    if output is None:
        suffix = ".otlp.json" if fmt == "otlp" else f".{fmt}"
        output = trace_file.with_suffix(suffix)

    if fmt == "json":
        export_json(trace, output)
    elif fmt == "otlp":
        export_otlp(trace, output)
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
    duration = f"{trace.duration:.3f}s" if trace.duration is not None else "N/A"
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
        # click.echo, not console.print: rich soft-wraps long lines, which
        # would inject newlines inside JSON strings and corrupt the output.
        click.echo(json_mod.dumps(data, indent=2, default=str))
        return

    console.print(f"\n[bold cyan]Stats: {trace.name}[/bold cyan]")
    duration = f"{trace.duration:.3f}s" if trace.duration is not None else "N/A"
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
@click.option(
    "--price",
    "prices",
    multiple=True,
    help="Pricing override as MODEL=INPUT:OUTPUT in USD per 1M tokens (repeatable).",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def cost(trace_file: Path, prices: tuple[str, ...], as_json: bool) -> None:
    """Show token and cost breakdowns for a trace's LLM calls.

    Prices token usage recorded in llm_response events (token_usage dicts or
    bare token totals) against a built-in per-1M-token table covering common
    OpenAI, Anthropic, and Google models. Versioned names match by prefix.
    Use --price to add or override pricing, for example:

        agent-replay cost trace.jsonl --price my-model=1.50:6.00
    """
    from .cost import analyze_trace

    overrides: dict[str, tuple[float, float]] = {}
    for spec in prices:
        model, sep, price_part = spec.partition("=")
        input_str, price_sep, output_str = price_part.partition(":")
        if not sep or not model or not price_sep:
            raise click.BadParameter(f"'{spec}' is not MODEL=INPUT:OUTPUT", param_hint="--price")
        try:
            input_price, output_price = float(input_str), float(output_str)
        except ValueError as exc:
            raise click.BadParameter(
                f"invalid prices for '{model}': {price_part}", param_hint="--price"
            ) from exc
        if input_price < 0 or output_price < 0:
            raise click.BadParameter(
                f"prices must be non-negative for '{model}'", param_hint="--price"
            )
        overrides[model] = (input_price, output_price)

    trace = Trace.load(trace_file)
    report = analyze_trace(trace, pricing=overrides or None)

    if as_json:
        click.echo(json_mod.dumps(report.to_dict(), indent=2))
        return

    console.print(f"\n[bold cyan]Cost: {report.trace_name}[/bold cyan]")
    if not report.calls:
        console.print("[dim]No LLM calls with token usage found in this trace.[/dim]")
        if report.calls_without_usage:
            console.print(
                f"[dim]{report.calls_without_usage} llm_response event(s) "
                f"had no token usage data.[/dim]"
            )
        return

    console.print(f"  LLM calls:    {len(report.calls)}")
    console.print(f"  Total tokens: {report.total_tokens:,}")
    console.print(f"  Total cost:   ${report.total_cost_usd:.6f}")

    from rich.table import Table

    model_table = Table(title="Cost by Model")
    model_table.add_column("Model", style="yellow")
    model_table.add_column("Calls", justify="right")
    model_table.add_column("Prompt", justify="right")
    model_table.add_column("Completion", justify="right")
    model_table.add_column("Tokens", justify="right")
    model_table.add_column("Cost", justify="right")
    for row in report.by_model():
        cost_str = f"${row['cost_usd']:.6f}" if row["cost_usd"] is not None else "n/a"
        if row["estimated"]:
            cost_str = f"~{cost_str}"
        model_table.add_row(
            row["model"],
            str(row["calls"]),
            f"{row['prompt_tokens']:,}",
            f"{row['completion_tokens']:,}",
            f"{row['total_tokens']:,}",
            cost_str,
        )
    console.print(model_table)

    span_table = Table(title="Cost by Span")
    span_table.add_column("Span", style="cyan")
    span_table.add_column("Calls", justify="right")
    span_table.add_column("Tokens", justify="right")
    span_table.add_column("Cost", justify="right")
    for row in report.by_span():
        cost_str = f"${row['cost_usd']:.6f}" if row["cost_usd"] is not None else "n/a"
        span_table.add_row(
            row["span_name"], str(row["calls"]), f"{row['total_tokens']:,}", cost_str
        )
    console.print(span_table)

    if report.has_estimates:
        console.print(
            "[dim]~ = estimated: only a token total was recorded, priced at the "
            "average of input and output rates.[/dim]"
        )
    if report.unpriced_models:
        models_str = ", ".join(report.unpriced_models)
        console.print(
            f"[yellow]No pricing for: {models_str}. "
            f"Add it with --price MODEL=INPUT:OUTPUT (USD per 1M tokens).[/yellow]"
        )
    if report.calls_without_usage:
        console.print(
            f"[dim]{report.calls_without_usage} llm_response event(s) "
            f"had no token usage data and were skipped.[/dim]"
        )


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path.")
@click.option(
    "--pattern",
    "-p",
    "patterns",
    multiple=True,
    help="Extra redaction rule as LABEL=REGEX (repeatable).",
)
@click.option("--placeholder", default="[REDACTED:{label}]", show_default=True,
              help="Replacement text, may contain {label}.")
def redact(trace_file: Path, output: Path | None, patterns: tuple[str, ...], placeholder: str) -> None:
    """Scrub API keys, tokens, and emails from a trace file.

    Writes a redacted copy so the trace can be shared safely.
    Builtin rules cover OpenAI, Anthropic, AWS, and GitHub keys,
    bearer tokens, and email addresses.
    """
    extra: dict[str, str] = {}
    for spec in patterns:
        label, sep, regex = spec.partition("=")
        if not sep or not label or not regex:
            raise click.BadParameter(
                f"'{spec}' is not LABEL=REGEX", param_hint="--pattern"
            )
        try:
            re.compile(regex)
        except re.error as exc:
            raise click.BadParameter(
                f"invalid regex for '{label}': {exc}", param_hint="--pattern"
            ) from exc
        extra[label] = regex

    trace = Trace.load(trace_file)
    redacted_trace, counts = redact_trace(trace, extra_patterns=extra, placeholder=placeholder)

    if output is None:
        output = trace_file.with_suffix(".redacted.jsonl")
    redacted_trace.save(output)

    total = sum(counts.values())
    if total == 0:
        console.print(f"[green]No sensitive data found. Clean copy written to {output}[/green]")
        return

    console.print(f"[bold cyan]Redacted {total} match(es):[/bold cyan]")
    for label, n in sorted(counts.items()):
        console.print(f"  {label:<20} {n:>5}")
    console.print(f"[green]Redacted trace written to {output}[/green]")


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
