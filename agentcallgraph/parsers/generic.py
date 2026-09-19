"""Generic JSONL parser for agent session logs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from agentcallgraph.graph.types import Event, EventType, Session


def parse_generic_jsonl(path: str | Path) -> Session:
    """Parse a generic JSONL file with agent session events.

    Expected event shape (any superset accepted):
    {"type": "tool_call", "tool_name": "bash", "tool_input": {...}, "timestamp": 1234567890.0}
    {"type": "tool_result", "tool_output": {...}, "tool_call_id": "call_xxx"}
    {"type": "llm_call", "token_usage": {"input_tokens": 100, "output_tokens": 50}}
    """
    events: list[Event] = []
    path = Path(path)

    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = _dict_to_event(raw, line_num)
            if event:
                events.append(event)

    return Session(
        session_id=path.stem + "-" + str(uuid.uuid4())[:8],
        source_format="generic_jsonl",
        events=events,
        metadata={"source_file": str(path)},
    )


def _dict_to_event(raw: dict[str, Any], line_num: int) -> Event | None:
    """Convert a raw dict to an Event, or None if unrecognizable."""
    event_type = _infer_type(raw)
    if event_type is None:
        return None

    import time

    ts = raw.get("timestamp", time.time() if "timestamp" not in raw else 0)
    if isinstance(ts, str):
        from datetime import datetime

        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            ts = 0.0

    tool_name = raw.get("tool_name")
    tool_input = raw.get("tool_input")
    tool_output = raw.get("tool_output")
    token_usage = raw.get("token_usage")

    return Event(
        event_id=raw.get("event_id", raw.get("tool_call_id", f"ev-{line_num}")),
        event_type=event_type,
        timestamp=float(ts),
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        token_usage=token_usage,
        parent_event_id=raw.get("parent_event_id"),
        turn_id=raw.get("turn_id"),
        metadata=raw.get("metadata", {}),
    )


def _infer_type(raw: dict[str, Any]) -> EventType | None:
    """Infer the event type from a raw dict."""
    raw_type = raw.get("type", "").lower()

    if raw_type in ("tool_call", "tool use", "function_call", "action"):
        return EventType.TOOL_CALL
    if raw_type in ("tool_result", "tool_output", "function_output", "observation"):
        return EventType.TOOL_RESULT
    if raw_type in ("llm_call", "llm", "assistant"):
        return EventType.LLM_CALL
    if raw_type in ("user", "user_input", "user_message"):
        return EventType.USER_INPUT
    if raw_type in ("system", "system_message"):
        return EventType.SYSTEM

    # Heuristic: has tool_name => tool_call, has tool_output => tool_result
    if "tool_name" in raw:
        return EventType.TOOL_CALL
    if "tool_output" in raw:
        return EventType.TOOL_RESULT
    if "token_usage" in raw:
        return EventType.LLM_CALL

    return None
