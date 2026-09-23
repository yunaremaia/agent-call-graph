"""Tests for detectors."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcallgraph.detectors.budget import find_budget_anomalies
from agentcallgraph.detectors.circular import find_circular_calls
from agentcallgraph.detectors.redundant import find_redundant_calls
from agentcallgraph.graph.types import Event, EventType, Session
from agentcallgraph.parsers.generic import parse_generic_jsonl


def test_redundant_detector_basic():
    fixture = Path(__file__).parent / "fixtures" / "redundant_ls.jsonl"
    session = parse_generic_jsonl(fixture)
    findings = find_redundant_calls(session)

    assert len(findings) >= 1
    assert findings[0].type == "redundant_call"
    assert findings[0].details["call_count"] == 3
    assert findings[0].details["wasted_calls"] == 2


def test_redundant_detector_no_duplicates():
    session = Session(
        session_id="clean",
        source_format="test",
        events=[
            Event(
                event_id="a",
                event_type=EventType.TOOL_CALL,
                timestamp=1.0,
                tool_name="bash",
                tool_input={"command": "ls"},
            ),
            Event(
                event_id="b",
                event_type=EventType.TOOL_CALL,
                timestamp=2.0,
                tool_name="bash",
                tool_input={"command": "pwd"},
            ),
        ],
    )
    findings = find_redundant_calls(session)
    assert len(findings) == 0


def test_redundant_detector_different_outputs():
    session = Session(
        session_id="diff",
        source_format="test",
        events=[
            Event(
                event_id="a",
                event_type=EventType.TOOL_CALL,
                timestamp=1.0,
                tool_name="bash",
                tool_input={"command": "cat file.txt"},
                tool_output={"output": "hello"},
            ),
            Event(
                event_id="b",
                event_type=EventType.TOOL_CALL,
                timestamp=2.0,
                tool_name="bash",
                tool_input={"command": "cat file.txt"},
                tool_output={"output": "world"},
            ),
        ],
    )
    findings = find_redundant_calls(session)
    assert len(findings) == 1
    assert findings[0].severity == "info"  # same args but different outputs


def test_budget_anomaly_normal():
    """No anomalies when all turns are within normal range."""
    events = []
    for i in range(10):
        events.append(
            Event(
                event_id=f"ev-{i}",
                event_type=EventType.LLM_CALL,
                timestamp=float(i),
                token_usage={"total_tokens": 1000},
                turn_id=i,
            )
        )
    session = Session(session_id="normal", source_format="test", events=events)
    findings = find_budget_anomalies(session, sigma_threshold=3.0)
    assert len(findings) == 0


def test_budget_anomaly_spike():
    """Detect a token spike in one turn."""
    events = []
    for i in range(10):
        tokens = 1000 if i != 5 else 50000
        events.append(
            Event(
                event_id=f"ev-{i}",
                event_type=EventType.LLM_CALL,
                timestamp=float(i),
                token_usage={"total_tokens": tokens},
                turn_id=i,
            )
        )
    session = Session(session_id="spike", source_format="test", events=events)
    findings = find_budget_anomalies(session, sigma_threshold=2.0)
    assert len(findings) >= 1
    assert findings[0].type == "budget_anomaly"


def test_circular_detector_basic():
    """Detect a repeated two-call sequence."""
    events = []

    tools = [
        ("read_file", {"path": "main.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "main.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "main.py"}),
        ("grep", {"pattern": "TODO"}),
    ]

    for i, (tool_name, tool_input) in enumerate(tools):
        events.append(
            Event(
                event_id=f"event-{i}",
                event_type=EventType.TOOL_CALL,
                timestamp=float(i),
                tool_name=tool_name,
                tool_input=tool_input,
            )
        )

    session = Session(
        session_id="circular-test",
        source_format="test",
        events=events,
    )

    findings = find_circular_calls(session)

    assert len(findings) >= 1
    assert findings[0].type == "circular_reasoning"
    assert findings[0].details["repetitions"] >= 3
def test_circular_detector_no_loop():
    """Do not detect non-repeating calls."""
    events = [
        Event(
            event_id=str(i),
            event_type=EventType.TOOL_CALL,
            timestamp=float(i),
            tool_name=f"tool_{i}",
            tool_input={"value": i},
        )
        for i in range(6)
    ]

    session = Session(
        session_id="no-loop",
        source_format="test",
        events=events,
    )

    findings = find_circular_calls(session)

    assert findings == []


def test_circular_detector_different_paths():
    """Do not treat different file paths as identical."""
    tools = [
        ("read_file", {"path": "/project/a.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "/project/b.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "/project/c.py"}),
        ("grep", {"pattern": "TODO"}),
    ]

    events = [
        Event(
            event_id=str(i),
            event_type=EventType.TOOL_CALL,
            timestamp=float(i),
            tool_name=name,
            tool_input=tool_input,
        )
        for i, (name, tool_input) in enumerate(tools)
    ]

    session = Session(
        session_id="different-paths",
        source_format="test",
        events=events,
    )

    findings = find_circular_calls(session)

    assert findings == []


def test_circular_detector_invalid_parameters():
    """Reject invalid detector parameters."""
    session = Session(
        session_id="invalid",
        source_format="test",
        events=[],
    )

    with pytest.raises(ValueError):
        find_circular_calls(session, loop_window=0)

    with pytest.raises(ValueError):
        find_circular_calls(session, loop_threshold=1)

def test_circular_detector_relative_paths():
    """Do not treat different relative paths as identical."""
    tools = [
        ("read_file", {"path": "src/a.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "src/b.py"}),
        ("grep", {"pattern": "TODO"}),
        ("read_file", {"path": "src/c.py"}),
        ("grep", {"pattern": "TODO"}),
    ]

    events = [
        Event(
            event_id=str(i),
            event_type=EventType.TOOL_CALL,
            timestamp=float(i),
            tool_name=name,
            tool_input=tool_input,
        )
        for i, (name, tool_input) in enumerate(tools)
    ]

    session = Session(
        session_id="relative-paths",
        source_format="test",
        events=events,
    )

    findings = find_circular_calls(session)

    assert findings == []