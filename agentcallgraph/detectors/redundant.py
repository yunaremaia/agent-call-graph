"""Detector for redundant/duplicate tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentcallgraph.graph.types import Event, Session


@dataclass
class Finding:
    type: str
    severity: str  # error, warning, info
    message: str
    details: dict[str, Any] | None = None


def find_redundant_calls(session: Session) -> list[Finding]:
    """Find tool calls with identical (tool_name, args) executed more than once."""
    findings = []
    seen: dict[str, list[Event]] = {}

    for event in session.tool_calls:
        key = event.tool_call_key
        if key is None:
            continue
        seen.setdefault(key, []).append(event)

    for key, events in seen.items():
        if len(events) < 2:
            continue

        tool_name = events[0].tool_name
        # Check if outputs were actually identical (true redundancy)
        outputs = set()
        for e in events:
            if e.tool_output is not None:
                outputs.add(_canonicalize(e.tool_output))

        if len(outputs) <= 1:
            severity = "warning"
            msg = f"Tool '{tool_name}' called {len(events)}× with identical arguments and same result"
        else:
            severity = "info"
            msg = f"Tool '{tool_name}' called {len(events)}× with identical arguments but different results"

        # Count wasted calls (all but the first)
        wasted = len(events) - 1
        total_cost = _estimate_cost(events)

        findings.append(
            Finding(
                type="redundant_call",
                severity=severity,
                message=msg,
                details={
                    "tool_name": tool_name,
                    "call_count": len(events),
                    "wasted_calls": wasted,
                    "estimated_cost_usd": total_cost,
                    "first_timestamp": events[0].timestamp,
                    "last_timestamp": events[-1].timestamp,
                },
            )
        )

    return findings


def _canonicalize(value: Any) -> str:
    """Create a stable string representation for comparison."""
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _estimate_cost(events: list[Event]) -> float:
    """Rough cost estimate: $0.003 per 1K input tokens + $0.015 per 1K output tokens."""
    total = 0.0
    for e in events:
        if e.token_usage:
            inp = e.token_usage.get("input_tokens", 0)
            out = e.token_usage.get("output_tokens", 0)
            total += (inp / 1000) * 0.003 + (out / 1000) * 0.015
    return round(total, 4)
