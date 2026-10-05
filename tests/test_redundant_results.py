"""Result evidence must survive parsing before redundant calls are classified."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentcallgraph.cli import main
from agentcallgraph.detectors.redundant import find_redundant_calls
from agentcallgraph.graph.types import Event, EventType, Session
from agentcallgraph.parsers.generic import parse_generic_jsonl
from agentcallgraph.parsers.hermes import parse_hermes_session


def _call(call_id, **extra):
    return {
        "type": "tool_call",
        "event_id": call_id,
        "tool_name": "read_file",
        "tool_input": {"path": "file.txt"},
        "timestamp": 1,
        **extra,
    }


def _result(call_id, output, **extra):
    return {
        "type": "tool_result",
        "event_id": f"result-{call_id}",
        "tool_call_id": call_id,
        "tool_output": output,
        "timestamp": 2,
        **extra,
    }


def _parse(tmp_path, records):
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return parse_generic_jsonl(path)


def _finding(session):
    findings = find_redundant_calls(session)
    assert len(findings) == 1
    return findings[0]


def test_result_parent_links_preserve_explicit_parent(tmp_path):
    session = _parse(
        tmp_path,
        [
            _result("a", "A", parent_event_id="explicit-parent"),
            _result("b", "B", parent_event_id=None),
            _result("c", "C"),
            _call("d", tool_call_id="not-a-parent"),
        ],
    )
    assert [event.parent_event_id for event in session.events] == [
        "explicit-parent",
        "b",
        "c",
        None,
    ]


@pytest.mark.parametrize(
    ("results", "severity", "message"),
    [
        ([], "info", "no result data"),
        ([_result("a", "A")], "info", "incomplete result data"),
        ([_result("a", "A"), _result("b", "A")], "warning", "same result"),
        ([_result("a", "A"), _result("b", "B")], "info", "different results"),
        (
            [_result("a", "A"), _result("a", "A"), _result("b", "A")],
            "warning",
            "same result",
        ),
        (
            [_result("a", "A"), _result("a", "B"), _result("b", "A")],
            "info",
            "incomplete result data",
        ),
        ([_result("a", "A"), _result("other", "A")], "info", "incomplete result data"),
        (
            [_result(None, "A", event_id="a"), _result(None, "A", event_id="b")],
            "info",
            "no result data",
        ),
    ],
    ids=[
        "missing",
        "partial",
        "same",
        "different",
        "repeated-result",
        "conflict",
        "orphan",
        "id-only",
    ],
)
def test_separate_result_classification(tmp_path, results, severity, message):
    finding = _finding(_parse(tmp_path, [_call("a"), _call("b"), *results]))
    assert finding.severity == severity
    assert message in finding.message


@pytest.mark.parametrize("output", ["", {}, 0, False])
def test_falsey_outputs_are_result_evidence(tmp_path, output):
    finding = _finding(
        _parse(tmp_path, [_call("a"), _result("a", output), _call("b"), _result("b", output)])
    )
    assert finding.severity == "warning"
    assert "same result" in finding.message


@pytest.mark.parametrize("outputs", [("{}", {}), ("0", 0), ("false", False)])
def test_result_comparison_preserves_json_types(tmp_path, outputs):
    finding = _finding(
        _parse(
            tmp_path, [_call("a"), _result("a", outputs[0]), _call("b"), _result("b", outputs[1])]
        )
    )
    assert finding.severity == "info"
    assert "different results" in finding.message


def test_result_comparison_ignores_object_key_order(tmp_path):
    finding = _finding(
        _parse(
            tmp_path,
            [
                _call("a"),
                _result("a", {"first": 1, "second": 2}),
                _call("b"),
                _result("b", {"second": 2, "first": 1}),
            ],
        )
    )
    assert finding.severity == "warning"
    assert "same result" in finding.message


@pytest.mark.parametrize(("outputs", "severity"), [(["A", "A"], "warning"), (["A", "B"], "info")])
def test_inline_output_api_is_preserved(outputs, severity):
    session = Session(
        "inline",
        "test",
        [
            Event(
                str(i),
                EventType.TOOL_CALL,
                float(i),
                tool_name="read_file",
                tool_input={"path": "file.txt"},
                tool_output=output,
            )
            for i, output in enumerate(outputs)
        ],
    )
    assert _finding(session).severity == severity


def test_conflicting_inline_and_linked_output_is_not_identical(tmp_path):
    finding = _finding(
        _parse(
            tmp_path, [_call("a", tool_output="A"), _call("b", tool_output="A"), _result("a", "B")]
        )
    )
    assert finding.severity == "info"
    assert "incomplete result data" in finding.message


def test_duplicate_call_id_in_another_group_cannot_borrow_result(tmp_path):
    finding = _finding(
        _parse(
            tmp_path,
            [
                _call("a"),
                _call("b"),
                _call("a", tool_input={"path": "other.txt"}),
                _result("a", "A"),
                _result("b", "A"),
            ],
        )
    )
    assert finding.severity == "info"
    assert "incomplete result data" in finding.message


@pytest.mark.parametrize("invalid_id", [[], {}, None])
def test_unusable_result_links_remain_unknown_without_crashing(tmp_path, invalid_id):
    finding = _finding(
        _parse(
            tmp_path, [_call(invalid_id), _call("b"), _result(invalid_id, "A"), _result("b", "A")]
        )
    )
    assert finding.severity == "info"
    assert "same result" not in finding.message


def test_known_different_outputs_remain_different_with_missing_result(tmp_path):
    finding = _finding(
        _parse(tmp_path, [_call("a"), _call("b"), _call("c"), _result("b", "B"), _result("a", "A")])
    )
    assert finding.severity == "info"
    assert "different results" in finding.message


def test_hermes_sqlite_links_results_to_calls(tmp_path):
    path = tmp_path / "session.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE messages (id INTEGER, role TEXT, content TEXT, "
            "tool_calls TEXT, tool_call_id TEXT, timestamp REAL)"
        )
        for i, output in enumerate(["A", "B"]):
            call_id = f"call-{i}"
            calls = json.dumps(
                [{"id": call_id, "name": "read_file", "arguments": {"path": "file.txt"}}]
            )
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (2 * i, "assistant", None, calls, None, 2 * i),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (2 * i + 1, "tool", output, None, call_id, 2 * i + 1),
            )
        conn.commit()
    session = parse_hermes_session(path)
    results = [event for event in session.events if event.event_type == EventType.TOOL_RESULT]
    assert [event.parent_event_id for event in results] == ["call-0", "call-1"]
    finding = _finding(session)
    assert finding.severity == "info"
    assert "different results" in finding.message


def test_cli_reports_different_separate_results():
    fixture = Path(__file__).parent / "fixtures" / "different_results.jsonl"
    result = CliRunner().invoke(main, [str(fixture), "--format", "json"])
    assert result.exit_code == 0, result.output
    findings = json.loads(result.output)["findings"]
    redundant = [finding for finding in findings if finding["type"] == "redundant_call"]
    assert len(redundant) == 1
    assert redundant[0]["severity"] == "info"
    assert "different results" in redundant[0]["message"]
