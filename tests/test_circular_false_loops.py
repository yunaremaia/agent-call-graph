"""Regression tests: the loop detector must only report loops the agent ran."""

from __future__ import annotations

import json
from pathlib import Path

from agentcallgraph.detectors.circular import find_circular_calls
from agentcallgraph.parsers.generic import parse_generic_jsonl


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _one_loop(repeats: int) -> list[dict]:
    records = []
    for i in range(repeats):
        records.append({"type": "tool_call", "tool_name": "bash",
                        "tool_input": {"command": "ls"},
                        "timestamp": 1000.0 + i * 2, "event_id": f"bash-{i}"})
        records.append({"type": "tool_call", "tool_name": "read",
                        "tool_input": {"file_path": "src/mod.py"},
                        "timestamp": 1001.0 + i * 2, "event_id": f"read-{i}"})
    return records


def _other_loop(repeats: int) -> list[dict]:
    records = []
    for i in range(repeats):
        records.append({"type": "tool_call", "tool_name": "grep",
                        "tool_input": {"pattern": "foo"},
                        "timestamp": 2000.0 + i * 2, "event_id": f"grep-{i}"})
        records.append({"type": "tool_call", "tool_name": "edit",
                        "tool_input": {"file_path": "src/other.py"},
                        "timestamp": 2001.0 + i * 2, "event_id": f"edit-{i}"})
    return records


def test_nameless_tool_calls_are_not_a_loop(tmp_path):
    # Six different commands, no tool_name, no repetition at all. The signature
    # of each is None, and None == None made them look identical (issue #21).
    path = _write(tmp_path, "nameless.jsonl",
                  [{"type": "tool_call", "tool_input": {"command": f"go {i}"},
                    "timestamp": float(i)} for i in range(6)])

    assert find_circular_calls(parse_generic_jsonl(path)) == []


def test_one_physical_loop_is_reported_once(tmp_path):
    # One logical loop (bash(ls) -> read(src/mod.py)) executed 5 times. Every
    # rotation of it, and every shorter/longer pattern slice of it, describes the
    # same 10 physical calls and must collapse into one finding (issue #23).
    path = _write(tmp_path, "one_loop.jsonl", _one_loop(5))

    findings = find_circular_calls(parse_generic_jsonl(path), loop_window=10, loop_threshold=3)

    assert len(findings) == 1, [f.details for f in findings]
    assert findings[0].details["repetitions"] == 5
    assert len(findings[0].details["loop_sequence"]) == 2


def test_two_distinct_loops_are_both_reported(tmp_path):
    # Positive control: dedup must not swallow a second, genuinely separate loop.
    path = _write(tmp_path, "two_loops.jsonl", _one_loop(3) + _other_loop(3))

    findings = find_circular_calls(parse_generic_jsonl(path), loop_window=20, loop_threshold=3)

    sequences = {tuple(f.details["loop_sequence"]) for f in findings}
    assert len(sequences) == 2, findings
