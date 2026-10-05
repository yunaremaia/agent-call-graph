"""Detector for redundant/duplicate tool calls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from agentcallgraph.graph.types import Event, EventType, Session


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
    calls = session.tool_calls
    call_id_counts = Counter(e.event_id for e in calls if isinstance(e.event_id, str))
    results_by_call: dict[str, set[str]] = {}

    for event in session.events:
        if (
            event.event_type == EventType.TOOL_RESULT
            and isinstance(event.parent_event_id, str)
            and event.tool_output is not None
        ):
            results_by_call.setdefault(event.parent_event_id, set()).add(
                _canonicalize(event.tool_output)
            )

    for event in calls:
        key = event.tool_call_key
        if key is None:
            continue
        seen.setdefault(key, []).append(event)

    for key, events in seen.items():
        if len(events) < 2:
            continue

        tool_name = events[0].tool_name
        # Only compare evidence that can be attributed to one call. Keep all
        # outputs for a link so conflicting result records cannot overwrite one
        # another and silently turn into an identical-result finding.
        outputs = set()
        complete = True
        has_result_data = False
        for e in events:
            call_outputs = set()
            if e.tool_output is not None:
                call_outputs.add(_canonicalize(e.tool_output))
                has_result_data = True
            if isinstance(e.event_id, str):
                linked_outputs = results_by_call.get(e.event_id, set())
                has_result_data |= bool(linked_outputs)
                if call_id_counts[e.event_id] == 1:
                    call_outputs.update(linked_outputs)

            if len(call_outputs) == 1:
                outputs.update(call_outputs)
            else:
                complete = False

        if len(outputs) > 1:
            severity = "info"
            msg = f"Tool '{tool_name}' called {len(events)}× with identical arguments but different results"
        elif complete:
            severity = "warning"
            msg = (
                f"Tool '{tool_name}' called {len(events)}× with identical arguments and same result"
            )
        else:
            severity = "info"
            availability = "incomplete result data" if has_result_data else "no result data"
            msg = f"Tool '{tool_name}' called {len(events)}× with identical arguments but {availability}"

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
