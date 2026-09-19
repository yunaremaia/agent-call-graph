"""Tests for parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcallgraph.parsers.generic import parse_generic_jsonl
from agentcallgraph.graph.types import EventType


def test_generic_parser_redundant_ls():
    fixture = Path(__file__).parent / "fixtures" / "redundant_ls.jsonl"
    session = parse_generic_jsonl(fixture)

    assert session.source_format == "generic_jsonl"
    assert len(session.tool_calls) == 3
    assert session.tool_calls[0].tool_name == "bash"
    assert session.tool_calls[0].tool_input == {"command": "ls"}


def test_generic_parser_empty():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        f.flush()
        session = parse_generic_jsonl(f.name)
        assert len(session.events) == 0


def test_generic_parser_malformed_lines():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("not json\n")
        f.write('{"type": "tool_call", "tool_name": "bash", "timestamp": 1.0}\n')
        f.write("\n")
        f.flush()
        session = parse_generic_jsonl(f.name)
        assert len(session.events) == 1
