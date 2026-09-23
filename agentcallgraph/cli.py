"""CLI entrypoint for agent-call-graph."""

from __future__ import annotations

import click

from agentcallgraph.detectors.budget import find_budget_anomalies
from agentcallgraph.detectors.circular import find_circular_calls
from agentcallgraph.detectors.redundant import find_redundant_calls
from agentcallgraph.parsers.generic import parse_generic_jsonl
from agentcallgraph.parsers.hermes import parse_hermes_session


@click.command()
@click.argument("session_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
)
@click.option(
    "--source",
    type=click.Choice(
        ["auto", "hermes", "claude-code", "codex", "opencode", "generic"], case_sensitive=False
    ),
    default="auto",
)
@click.option("--threshold", type=float, default=3.0, help="Budget anomaly sigma threshold")
@click.option(
    "--loop-window",
    type=int,
    default=10,
    help="Number of recent tool calls to analyze for loops",
)
@click.option(
    "--loop-threshold",
    type=int,
    default=3,
    help="Minimum number of repetitions to report",
)
@click.option("--fail-on-findings/--no-fail-on-findings", is_flag=True, default=False)
@click.version_option()
def main(
    session_file: str,
    output_format: str,
    source: str,
    threshold: float,
    fail_on_findings: bool,
    loop_window: int,
    loop_threshold: int,
):
    """Analyze an AI agent session log for wasteful tool call patterns.

    SESSION_FILE is the path to a session log file (JSONL, SQLite DB, etc.)
    """
    session = _parse_session(session_file, source)

    findings = []
    findings.extend(find_redundant_calls(session))
    findings.extend(find_budget_anomalies(session, sigma_threshold=threshold))
    findings.extend(
        find_circular_calls(session, loop_window=loop_window, loop_threshold=loop_threshold)
    )

    if output_format == "json":
        import json

        result = {
            "session_id": session.session_id,
            "source_format": session.source_format,
            "total_events": len(session.events),
            "total_tool_calls": len(session.tool_calls),
            "duration_seconds": session.duration_seconds,
            "total_tokens": session.total_tokens,
            "findings": [
                {
                    "type": f.type,
                    "severity": f.severity,
                    "message": f.message,
                    "details": f.details,
                }
                for f in findings
            ],
        }
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        _render_text(session, findings)

    if fail_on_findings and findings:
        raise click.Exit(1)


def _parse_session(path: str, source: str):
    if source == "hermes":
        return parse_hermes_session(path)
    elif source == "generic":
        return parse_generic_jsonl(path)
    else:
        # Auto-detect: try hermes first, fallback to generic
        from pathlib import Path as _Path

        p = _Path(path)
        if p.suffix in (".db", ".sqlite", ".sqlite3") or "hermes" in p.name.lower():
            try:
                return parse_hermes_session(path)
            except (ValueError, OSError):
                pass
        return parse_generic_jsonl(path)


def _render_text(session, findings):
    click.echo(f"Session: {session.session_id}")
    click.echo(f"Format: {session.source_format}")
    click.echo(f"Events: {len(session.events)} ({len(session.tool_calls)} tool calls)")
    click.echo(f"Duration: {session.duration_seconds:.1f}s")
    click.echo(f"Total tokens: {session.total_tokens}")
    click.echo("")

    if not findings:
        click.echo("✅ No findings — session looks clean.")
        return

    click.echo(f"⚠️  {len(findings)} finding(s):\n")
    for f in findings:
        severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(f.severity, "⚪")
        click.echo(f"  {severity_icon} [{f.severity.upper()}] {f.type}")
        click.echo(f"     {f.message}")
        if f.details:
            click.echo(f"     Details: {f.details}")
        click.echo("")


if __name__ == "__main__":
    main()
