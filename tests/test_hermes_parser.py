"""Tests for the Hermes session parser.

The JSON-fallback branch of ``parse_hermes_session`` used to be dead code:
``parse_generic_jsonl`` was imported *inside* the ``.jsonl`` branch, which made
it a function-local name for the whole body. Any path that reached line 31 --
a Hermes-named file whose contents are valid JSON but whose suffix is not
``.jsonl`` -- therefore raised ``UnboundLocalError`` instead of parsing.
These tests pin both halves of the dispatch.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentcallgraph.parsers.hermes import parse_hermes_session

_JSON_FALLBACK_EVENTS = [
    {"type": "tool_call", "tool_name": "bash", "tool_input": {"command": "ls"}, "timestamp": 1.0},
    {
        "type": "tool_result",
        "tool_output": {"content": "a"},
        "tool_call_id": "c1",
        "timestamp": 2.0,
    },
]


def _write_json_fallback(tmp_path: Path) -> Path:
    """A Hermes-named file with a non-.jsonl suffix holding one JSON document.

    The suffix checks do not match, so the parser sniffs the contents: ``json.load``
    succeeds and hands the file to ``parse_generic_jsonl``. That call site is the
    one the function-local import shadowed, which is what issue #16 reported.
    """
    path = tmp_path / "hermes_session.log"
    path.write_text(json.dumps(_JSON_FALLBACK_EVENTS[0]))
    return path


def test_json_fallback_does_not_raise_unbound_local(tmp_path):
    """Regression: the fallback branch must not blow up with UnboundLocalError.

    This is the exact crash reported in issue #16. The assertion is on the
    absence of the exception rather than on the parsed shape, so it keeps
    guarding the bug even if the parser's output shape changes later.
    """
    session = parse_hermes_session(_write_json_fallback(tmp_path))

    assert len(session.events) == 1


def test_json_fallback_parses_tool_calls(tmp_path):
    """The fallback branch must actually parse, not just avoid raising."""
    session = parse_hermes_session(_write_json_fallback(tmp_path))

    assert session.source_format == "generic_jsonl"
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0].tool_name == "bash"
    assert session.tool_calls[0].tool_input == {"command": "ls"}


def test_jsonl_suffix_branch_keeps_hermes_format(tmp_path):
    """The original .jsonl branch must keep stamping the hermes format."""
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in _JSON_FALLBACK_EVENTS))

    session = parse_hermes_session(path)

    assert session.source_format == "hermes_jsonl"
    assert len(session.tool_calls) == 1


def test_sqlite_suffix_branch_does_not_use_fallback(tmp_path):
    """A .db suffix must route to SQLite parsing even when the file is JSON.

    Guards the dispatch order: the suffix checks win over content sniffing,
    so a stray JSON file named ``.db`` cannot reach the fallback branch.
    """
    path = tmp_path / "hermes.db"
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

    session = parse_hermes_session(path)

    assert session.source_format == "hermes_sqlite"
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0].tool_name == "bash"
