"""Tests for detectors."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcallgraph.parsers.generic import parse_generic_jsonl
from agentcallgraph.detectors.redundant import find_redundant_calls
from agentcallgraph.detectors.budget import find_budget_anomalies
from agentcallgraph.graph.types import Event, EventType, Session


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
            Event(event_id="a", event_type=EventType.TOOL_CALL, timestamp=1.0, tool_name="bash", tool_input={"command": "ls"}),
            Event(event_id="b", event_type=EventType.TOOL_CALL, timestamp=2.0, tool_name="bash", tool_input={"command": "pwd"}),
        ],
    )
    findings = find_redundant_calls(session)
    assert len(findings) == 0


def test_redundant_detector_different_outputs():
    session = Session(
        session_id="diff",
        source_format="test",
        events=[
            Event(event_id="a", event_type=EventType.TOOL_CALL, timestamp=1.0, tool_name="bash", tool_input={"command": "cat file.txt"}, tool_output={"output": "hello"}),
            Event(event_id="b", event_type=EventType.TOOL_CALL, timestamp=2.0, tool_name="bash", tool_input={"command": "cat file.txt"}, tool_output={"output": "world"}),
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
