"""Hermes Agent session parser."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agentcallgraph.graph.types import Event, EventType, Session


def parse_hermes_session(path: str | Path) -> Session:
    """Parse a Hermes Agent session DB (SQLite) or exported JSONL."""
    path = Path(path)

    if path.suffix == ".jsonl":
        from agentcallgraph.parsers.generic import parse_generic_jsonl

        session = parse_generic_jsonl(path)
        session.source_format = "hermes_jsonl"
        return session

    if path.suffix in (".db", ".sqlite", ".sqlite3"):
        return _parse_hermes_sqlite(path)

    # Try JSON first, then SQLite
    try:
        with path.open() as f:
            json.load(f)
        return parse_generic_jsonl(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _parse_hermes_sqlite(path)


def _parse_hermes_sqlite(path: Path) -> Session:
    """Parse Hermes Agent SQLite session store."""
    events: list[Event] = []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    # Hermes stores messages in a messages table
    try:
        cursor = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, timestamp FROM messages ORDER BY id"
        )
        for idx, row in enumerate(cursor):
            role = row["role"]
            if role == "assistant" and row["tool_calls"]:
                # tool_calls is JSON array of {name, arguments, id}
                tcs = json.loads(row["tool_calls"])
                for tc in tcs:
                    events.append(
                        Event(
                            event_id=tc.get("id", f"ev-{idx}"),
                            event_type=EventType.TOOL_CALL,
                            timestamp=float(row["timestamp"] or 0),
                            tool_name=tc.get("name"),
                            tool_input=_parse_args(tc.get("arguments", "")),
                        )
                    )
            elif role == "tool" and row["tool_call_id"]:
                events.append(
                    Event(
                        event_id=row["tool_call_id"],
                        event_type=EventType.TOOL_RESULT,
                        timestamp=float(row["timestamp"] or 0),
                        tool_output={"content": row["content"]},
                    )
                )
            elif role == "assistant":
                events.append(
                    Event(
                        event_id=f"ev-{idx}",
                        event_type=EventType.LLM_CALL,
                        timestamp=float(row["timestamp"] or 0),
                    )
                )
            elif role == "user":
                events.append(
                    Event(
                        event_id=f"ev-{idx}",
                        event_type=EventType.USER_INPUT,
                        timestamp=float(row["timestamp"] or 0),
                    )
                )
    except sqlite3.OperationalError:
        # Fallback: empty session
        pass
    finally:
        conn.close()

    return Session(
        session_id=path.stem,
        source_format="hermes_sqlite",
        events=events,
        metadata={"source_file": str(path)},
    )


def _parse_args(arguments: str) -> dict[str, Any]:
    """Parse tool arguments (may be JSON string or dict)."""
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"raw": arguments}
