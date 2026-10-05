"""Detector for budget anomalies (unusual token spend)."""

from __future__ import annotations

import statistics

from agentcallgraph.detectors.redundant import Finding
from agentcallgraph.graph.types import Event, Session, usage_tokens


def find_budget_anomalies(session: Session, sigma_threshold: float = 3.0) -> list[Finding]:
    """Find turns or events that exceed expected token budget by N sigma."""
    findings = []

    # Group by turn. Events with no turn_id get their own group instead of
    # sharing turn 0: collapsing them made len(turns) == 1, the early return
    # below fired, and a log in the Claude Code / Codex shape -- which carries no
    # turn_id at all -- could never report a budget anomaly (issue #15).
    turns: dict[int, list[Event]] = {}
    for position, e in enumerate(session.events):
        tid = e.turn_id if e.turn_id is not None else position
        turns.setdefault(tid, []).append(e)

    if len(turns) < 2:
        return findings

    # Compute per-turn token totals
    turn_totals: dict[int, int] = {}
    for tid, events in turns.items():
        turn_totals[tid] = sum(usage_tokens(e.token_usage) for e in events)

    totals = list(turn_totals.values())
    if len(totals) < 2:
        return findings

    mean = statistics.mean(totals)
    stdev = statistics.stdev(totals) if len(totals) > 1 else 0

    if stdev == 0:
        return findings

    for tid, total in turn_totals.items():
        z_score = (total - mean) / stdev
        if z_score > sigma_threshold:
            findings.append(
                Finding(
                    type="budget_anomaly",
                    severity="warning",
                    message=f"Turn {tid} used {total} tokens (avg: {mean:.0f}, σ: {stdev:.0f}, z={z_score:.1f})",
                    details={
                        "turn_id": tid,
                        "tokens": total,
                        "mean_tokens": mean,
                        "stdev_tokens": stdev,
                        "z_score": z_score,
                    },
                )
            )

    return findings
