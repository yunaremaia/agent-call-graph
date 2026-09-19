"""Detector for budget anomalies (unusual token spend)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from agentcallgraph.detectors.redundant import Finding
from agentcallgraph.graph.types import Event, Session


def find_budget_anomalies(session: Session, sigma_threshold: float = 3.0) -> list[Finding]:
    """Find turns or events that exceed expected token budget by N sigma."""
    findings = []

    # Group by turn
    turns: dict[int, list[Event]] = {}
    for e in session.events:
        tid = e.turn_id if e.turn_id is not None else 0
        turns.setdefault(tid, []).append(e)

    if len(turns) < 2:
        return findings

    # Compute per-turn token totals
    turn_totals: dict[int, int] = {}
    for tid, events in turns.items():
        total = 0
        for e in events:
            if e.token_usage:
                total += e.token_usage.get("total_tokens", 0)
        turn_totals[tid] = total

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
