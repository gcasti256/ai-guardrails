"""CLI interface for AI Guardrails."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from guardrails import GuardrailsEngine
from guardrails.policy.loader import PolicyLoader
from guardrails.types import RedactionStrategy, Severity

app = typer.Typer(
    name="guardrails",
    help="AI Guardrails — Production-grade AI safety and content guardrails",
    no_args_is_help=True,
)
console = Console()


def _get_engine(policy_path: str | None = None) -> GuardrailsEngine:
    """Create and configure a GuardrailsEngine instance."""
    engine = GuardrailsEngine()
    if policy_path:
        p = Path(policy_path)
        if p.is_dir():
            engine.load_policies_dir(p)
        elif p.is_file():
            engine.load_policy(p)
        else:
            console.print(f"[red]Policy path not found: {policy_path}[/red]")
            raise typer.Exit(1)
    else:
        # Try default policies directory
        default_dir = Path(__file__).parent.parent.parent / "policies"
        if default_dir.exists():
            engine.load_policies_dir(default_dir)
    return engine


@app.command()
def scan(
    text: Optional[str] = typer.Argument(None, help="Text to scan (or pipe via stdin)"),
    policy: Optional[str] = typer.Option(None, "--policy", "-p", help="Policy file or directory"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Scan text against active policies."""
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            console.print("[red]Error: provide text as argument or pipe via stdin[/red]")
            raise typer.Exit(1)

    engine = _get_engine(policy)
    result = asyncio.run(engine.scan(text))

    if output_json:
        output = {
            "is_safe": result.is_safe,
            "action": result.action.value,
            "detections": [
                {
                    "entity_type": d.entity_type,
                    "text": d.text,
                    "confidence": d.confidence,
                    "severity": d.severity.value,
                }
                for d in result.detections
            ],
            "violations": [
                {
                    "rule": v.rule_name,
                    "policy": v.policy_name,
                    "action": v.action.value,
                    "message": v.message,
                }
                for v in result.policy_violations
            ],
        }
        console.print_json(json.dumps(output))
        return

    # Rich table output
    if result.is_safe:
        console.print("[green]✓ Text is safe[/green]")
    else:
        console.print(f"[red]✗ Text flagged — Action: {result.action.value.upper()}[/red]")

    if result.detections:
        table = Table(title="Detections")
        table.add_column("Entity Type", style="cyan")
        table.add_column("Text", style="white")
        table.add_column("Confidence", style="yellow")
        table.add_column("Severity", style="red")
        for d in result.detections:
            severity_color = {
                Severity.LOW: "green",
                Severity.MEDIUM: "yellow",
                Severity.HIGH: "red",
                Severity.CRITICAL: "bold red",
            }.get(d.severity, "white")
            table.add_row(
                d.entity_type,
                d.text[:50] + ("..." if len(d.text) > 50 else ""),
                f"{d.confidence:.2f}",
                f"[{severity_color}]{d.severity.value}[/{severity_color}]",
            )
        console.print(table)

    if result.policy_violations:
        table = Table(title="Policy Violations")
        table.add_column("Rule", style="cyan")
        table.add_column("Policy", style="white")
        table.add_column("Action", style="red")
        table.add_column("Message", style="yellow")
        for v in result.policy_violations:
            table.add_row(v.rule_name, v.policy_name, v.action.value, v.message)
        console.print(table)


@app.command()
def redact(
    file_path: Optional[str] = typer.Argument(None, help="File to redact (or pipe via stdin)"),
    strategy: str = typer.Option("replace", "--strategy", "-s", help="Redaction strategy: mask, hash, replace, remove"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    policy: Optional[str] = typer.Option(None, "--policy", "-p", help="Policy file or directory"),
) -> None:
    """Redact PII from a file or stdin."""
    if file_path:
        p = Path(file_path)
        if not p.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            raise typer.Exit(1)
        text = p.read_text()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        console.print("[red]Error: provide a file path or pipe via stdin[/red]")
        raise typer.Exit(1)

    try:
        redaction_strategy = RedactionStrategy(strategy)
    except ValueError:
        console.print(f"[red]Invalid strategy: {strategy}. Use: mask, hash, replace, remove[/red]")
        raise typer.Exit(1)

    engine = _get_engine(policy)
    result = asyncio.run(engine.redact(text, strategy=redaction_strategy))

    if output:
        Path(output).write_text(result.redacted_text)
        console.print(f"[green]Redacted output written to {output}[/green]")
    else:
        console.print(result.redacted_text)

    if result.redactions:
        console.print(f"\n[yellow]Redacted {len(result.redactions)} items:[/yellow]")
        for r in result.redactions:
            console.print(f"  • {r.entity_type}: '{r.original}' → '{r.replacement}'")


# Sub-command group for policy operations
policy_app = typer.Typer(help="Policy management commands")
app.add_typer(policy_app, name="policy")


@policy_app.command("validate")
def policy_validate(
    config_path: str = typer.Argument(..., help="Policy YAML file to validate"),
) -> None:
    """Validate a policy YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        console.print(f"[red]File not found: {config_path}[/red]")
        raise typer.Exit(1)

    loader = PolicyLoader()
    try:
        policy = loader.load_file(p)
        errors = loader.validate(policy)
        if errors:
            console.print(f"[yellow]Policy '{policy.name}' has {len(errors)} issue(s):[/yellow]")
            for err in errors:
                console.print(f"  [red]• {err}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]✓ Policy '{policy.name}' is valid ({len(policy.rules)} rules)[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load policy: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
