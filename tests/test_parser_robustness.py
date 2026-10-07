"""Regression tests: a hostile or truncated record must not kill the session.

Covers issues #19, #20, #33 and #34 -- all of them the same parser trusting a
field it never validated, or decoding text with whatever the locale happens to
be.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentcallgraph.detectors.budget import find_budget_anomalies
from agentcallgraph.parsers.generic import parse_generic_jsonl


def _tool_call(event_id: str, timestamp: float) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "tool_name": "bash",
            "tool_input": {"command": "ls"},
            "timestamp": timestamp,
            "event_id": event_id,
        }
    )


def _write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def test_non_dict_json_lines_are_skipped(tmp_path):
    # A single-line JSON array export, a null flush, a bare scalar and a stray
    # string: none of them is an event, and none of them may abort the parse.
    path = _write(
        tmp_path,
        "mixed.jsonl",
        "\n".join(
            [
                '[{"type": "tool_call", "tool_name": "bash", "timestamp": 1.0}]',
                "null",
                '"a stray string"',
                "42",
                _tool_call("t1", 2.0),
            ]
        )
        + "\n",
    )

    session = parse_generic_jsonl(path)

    assert [e.event_id for e in session.events] == ["t1"]


def test_null_timestamp_keeps_the_surrounding_events(tmp_path):
    path = _write(
        tmp_path,
        "ts_null.jsonl",
        "\n".join(
            [
                json.dumps({"type": "user_input", "timestamp": 1.0, "event_id": "u1"}),
                json.dumps(
                    {
                        "type": "llm_call",
                        "timestamp": None,
                        "token_usage": {"total_tokens": 5000},
                    }
                ),
                _tool_call("t1", 3.0),
            ]
        )
        + "\n",
    )

    session = parse_generic_jsonl(path)

    assert [e.event_id for e in session.events] == ["u1", "ev-2", "t1"]
    assert session.total_tokens == 5000


def test_object_timestamp_keeps_the_event(tmp_path):
    path = _write(
        tmp_path,
        "ts_dict.jsonl",
        json.dumps(
            {"type": "llm_call", "timestamp": {"seconds": 5}, "token_usage": {"total_tokens": 10}}
        )
        + "\n",
    )

    session = parse_generic_jsonl(path)

    assert len(session.events) == 1
    assert session.total_tokens == 10


def test_unparseable_string_timestamp_is_zero_not_a_crash(tmp_path):
    path = _write(
        tmp_path,
        "ts_str.jsonl",
        json.dumps({"type": "llm_call", "timestamp": "not-a-date", "token_usage": {"total_tokens": 7}})
        + "\n",
    )

    assert parse_generic_jsonl(path).total_tokens == 7


def test_utf8_bom_does_not_drop_the_first_event(tmp_path):
    # What every Windows editor and PowerShell 5.1 `Set-Content -Encoding utf8`
    # writes. The BOM must be consumed as an encoding signature, not read as a
    # leading byte of the first JSON line.
    text = "\n".join([_tool_call("b0", 1.0), _tool_call("b1", 2.0)]) + "\n"

    with_bom = _write(tmp_path, "bom.jsonl", text, encoding="utf-8-sig")
    without_bom = _write(tmp_path, "plain.jsonl", text)

    assert len(parse_generic_jsonl(with_bom).events) == len(parse_generic_jsonl(without_bom).events) == 2


def test_undecodable_byte_does_not_abort_the_parse(tmp_path):
    path = tmp_path / "latin1.jsonl"
    payload = (_tool_call("t0", 1.0) + "\n" + _tool_call("t1", 2.0) + "\n").encode()
    path.write_bytes(payload + b"\xff\xfe\n")

    session = parse_generic_jsonl(path)

    assert [e.event_id for e in session.events] == ["t0", "t1"]


def test_sqlite_without_messages_table_raises_not_empty_session(tmp_path):
    # Issue #18: a SQLite DB without a messages table was silently swallowed,
    # returning an empty Session that the CLI reported as "clean" with exit 0.
    import sqlite3

    from agentcallgraph.parsers.hermes import parse_hermes_session

    db_path = tmp_path / "other.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sessions(id INTEGER)")
    conn.commit()
    conn.close()

    import pytest

    with pytest.raises(ValueError, match="no 'messages' table"):
        parse_hermes_session(db_path)


def test_non_object_token_usage_is_ignored_not_fatal(tmp_path):
    # A scalar or list where a usage object is documented: the record is still a
    # valid event, and the budget detector must not raise on it.
    path = _write(
        tmp_path,
        "tok.jsonl",
        "\n".join(
            [
                json.dumps({"type": "llm_call", "turn_id": 0, "timestamp": 1.0, "token_usage": "1500"}),
                json.dumps({"type": "llm_call", "turn_id": 1, "timestamp": 2.0, "token_usage": [1500]}),
                json.dumps(
                    {
                        "type": "llm_call",
                        "turn_id": 2,
                        "timestamp": 3.0,
                        "token_usage": {"total_tokens": 4},
                    }
                ),
            ]
        )
        + "\n",
    )

    session = parse_generic_jsonl(path)

    assert len(session.events) == 3
    assert find_budget_anomalies(session) == [] or all(
        f.type == "budget_anomaly" for f in find_budget_anomalies(session)
    )
    assert all(isinstance(e.token_usage, dict | None) for e in session.events)
