"""Detector for circular reasoning patterns."""

from __future__ import annotations

import json
import re
from typing import Any

from agentcallgraph.detectors.redundant import Finding
from agentcallgraph.graph.types import Session


def find_circular_calls(
    session: Session,
    loop_window: int = 10,
    loop_threshold: int = 3,
) -> list[Finding]:
    """Detect repeated sequences of tool calls."""

    if loop_window < 1:
        raise ValueError("loop_window must be positive")

    if loop_threshold < 2:
        raise ValueError("loop_threshold must be at least 2")

    tool_calls = session.tool_calls[-loop_window:]

    path_map: dict[str, str] = {}
    # A call with no tool_name has no signature, and comparing None against None
    # made every nameless call match its neighbour -- so any two or more of them
    # were reported as a repeated sequence the agent never ran (issue #21). Keep
    # the original position so the reported indices still index tool_calls.
    indexed = [
        (position, signature)
        for position, signature in enumerate(
            _call_signature(event, path_map) for event in tool_calls
        )
        if signature is not None
    ]
    signatures = [signature for _, signature in indexed]

    findings = []
    n = len(signatures)
    # Index of the first call not already covered by a reported loop. A repeated
    # sequence is periodic, so every rotation of it (and every longer slice of
    # it) names the same physical calls: reporting each start position turned one
    # loop into a quadratic pile of duplicates of itself (issue #23).
    covered_until = 0

    for pattern_length in range(2, min(5, n // loop_threshold) + 1):
        for start in range(n - pattern_length + 1):
            if start < covered_until:
                continue
            pattern = signatures[start : start + pattern_length]

            repetitions = 1
            next_start = start + pattern_length

            while (
                next_start + pattern_length <= n
                and signatures[next_start : next_start + pattern_length] == pattern
            ):
                repetitions += 1
                next_start += pattern_length

            if repetitions < loop_threshold:
                continue

            findings.append(
                Finding(
                    type="circular_reasoning",
                    severity="warning",
                    message=(
                        f"Tool sequence repeated {repetitions} times: "
                        f"{pattern_length} calls per loop"
                    ),
                    details={
                        "loop_sequence": pattern,
                        "repetitions": repetitions,
                        "start_index": indexed[start][0],
                        "end_index": indexed[next_start - 1][0],
                    },
                )
            )
            covered_until = next_start

    return findings


def _normalize(
    value: Any,
    path_map: dict[str, str],
) -> Any:
    """Normalize arguments while preserving path identity."""

    if isinstance(value, dict):
        return {
            key.lower().strip(): _normalize(item, path_map)
            for key, item in sorted(value.items())
        }

    if isinstance(value, list):
        return [_normalize(item, path_map) for item in value]

    if isinstance(value, str):
        value = value.lower().strip()

        # Normalize file paths while preserving identity.
        def replace_path(match: re.Match[str]) -> str:
            path = match.group(0)

            if path not in path_map:
                path_map[path] = f"<path_{len(path_map)}>"

            return path_map[path]

        value = re.sub(
            r"(?:[a-zA-Z]:[\\/]|/|\./|(?:[\w.-]+[\\/]))[^\s\"']+",
            replace_path,
            value,
        )

        # Normalize numbers.
        value = re.sub(r"\b\d+\b", "<num>", value)

        return value

    return value


def _call_signature(
    event: Any,
    path_map: dict[str, str],
) -> str | None:
    """Create a comparable signature for a tool call."""

    if not event.tool_name:
        return None

    normalized_args = _normalize(
        event.tool_input or {},
        path_map,
    )

    return json.dumps(
        {
            "tool": event.tool_name.lower().strip(),
            "args": normalized_args,
        },
        sort_keys=True,
        default=str,
    )