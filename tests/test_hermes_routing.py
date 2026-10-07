"""Routing and crash guards for ``parse_hermes_session``.

Issue #29: a Hermes-named file with no recognised suffix and multi-line JSONL
content crashed with ``sqlite3.DatabaseError: file is not a database``. Two
distinct defects sat behind that one traceback, so these tests pin both:

1. ``_parse_hermes_sqlite`` caught ``sqlite3.OperationalError``, which is the
   *subclass* that covers "no such table"-style failures. "File is not a
   database" is raised as ``sqlite3.DatabaseError`` itself, so the handler
   never fired and the exception escaped.
2. The unknown-suffix sniff called ``json.load()``, which rejects multi-line
   JSONL with "Extra data" and routed a perfectly valid text log into the
   SQLite parser.

Both defects failed *silently* in the same way: a session with zero events and
exit status 0 for a file that contains real tool calls. These assertions are on
the parsed events, not on the absence of an exception, so they keep guarding
the real behaviour if the empty-session fallback ever moves.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentcallgraph.parsers.hermes import parse_hermes_session

# A JSONL body whose lines are individually valid JSON. json.load() cannot read
# this ("Extra data"), which is exactly what routed it to the SQLite parser.
_MULTILINE_JSONL = [
    {"type": "tool_call", "tool_name": "bash", "tool_input": {"command": "ls"}, "timestamp": 1.0},
    {"type": "tool_result", "tool_output": {"content": "a.txt"}, "tool_call_id": "c1", "timestamp": 2.0},
]


def _write_unsuffixed_jsonl(tmp_path: Path) -> Path:
    """A Hermes-named file whose content is multi-line JSONL."""
    path = tmp_path / "hermes_session"
    path.write_text("".join(json.dumps(e) + "\n" for e in _MULTILINE_JSONL))
    return path


def _write_unsuffixed_sqlite(tmp_path: Path) -> Path:
    """The same name, but with a real SQLite header and a messages table."""
    path = tmp_path / "hermes_store"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, role TEXT, content TEXT,"
        " tool_calls TEXT, tool_call_id TEXT, timestamp REAL)"
    )
    conn.execute(
        "INSERT INTO messages (role, content, tool_calls, tool_call_id, timestamp)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            "assistant",
            "",
            json.dumps([{"name": "bash", "arguments": {"command": "ls"}, "id": "tc1"}]),
            None,
            1.0,
        ),
    )
    conn.commit()
    conn.close()
    return path


def test_multiline_jsonl_without_suffix_is_parsed(tmp_path):
    """Regression for issue #29: the crash, and the empty-session lie after it.

    Before the fix this raised ``sqlite3.DatabaseError``; with only the
    exception handler widened it returned zero events for a two-event file.
    Both are wrong, so the assertion is on the two tool calls themselves.
    """
    session = parse_hermes_session(_write_unsuffixed_jsonl(tmp_path))

    assert session.source_format == "generic_jsonl"
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0].tool_name == "bash"
    assert session.tool_calls[0].tool_input == {"command": "ls"}


def test_real_sqlite_without_suffix_still_routes_to_sqlite(tmp_path):
    """Header sniffing must not misroute a genuine database to the JSON parser.

    The routing decision was inverted from "parse as SQLite unless it parses as
    JSON" to "read the 16-byte header", so both directions need pinning.
    """
    session = parse_hermes_session(_write_unsuffixed_sqlite(tmp_path))

    assert session.source_format == "hermes_sqlite"
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0].tool_name == "bash"


def test_db_suffix_with_text_content_does_not_crash(tmp_path):
    """A .db-suffixed file that is not a database must degrade, not raise.

    The suffix branch calls ``_parse_hermes_sqlite`` directly, so it reaches the
    handler with no sniff in front of it. Before the fix this propagated
    ``sqlite3.DatabaseError`` out of the parser.
    """
    path = tmp_path / "hermes.db"
    path.write_text('{"role": "user", "content": "oi"}\n')

    session = parse_hermes_session(path)

    assert session.events == []


def test_sqlite_without_messages_table_raises_not_degrades(tmp_path):
    """Issue #18: a real DB with no messages table must raise, not return empty.

    The old behaviour returned an empty Session, which the CLI rendered as
    "No findings — session looks clean" with exit 0 -- a false negative
    presented as a positive result.
    """
    import pytest

    path = tmp_path / "hermes_empty.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE unrelated (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="no 'messages' table"):
        parse_hermes_session(path)