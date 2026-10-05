"""Regression tests: token accounting must read the keys real logs emit.

Covers issue #15 -- `total_tokens` was read from a single key that real session
logs do not use, and events without a turn_id were collapsed into one group,
which made the budget detector unable to report anything at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentcallgraph.detectors.budget import find_budget_anomalies
from agentcallgraph.parsers.generic import parse_generic_jsonl

SPIKE = [(100, 50), (120, 60), (9000, 4000), (110, 55), (130, 65)]


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _split_usage_log(tmp_path: Path, with_turn_id: bool) -> Path:
    records = []
    for n, (inp, out) in enumerate(SPIKE):
        record = {
            "type": "llm_call",
            "token_usage": {"input_tokens": inp, "output_tokens": out},
            "timestamp": float(n),
            "event_id": f"llm{n}",
        }
        if with_turn_id:
            record["turn_id"] = n
        records.append(record)
    return _write(tmp_path, f"tokens_{with_turn_id}.jsonl", records)


def test_total_tokens_sums_input_and_output(tmp_path):
    # The exact shape parse_generic_jsonl's own docstring documents.
    session = parse_generic_jsonl(_split_usage_log(tmp_path, True))

    assert session.total_tokens == sum(i + o for i, o in SPIKE) == 13690


def test_precomputed_total_wins_over_split_fields():
    from agentcallgraph.graph.types import usage_tokens

    assert usage_tokens({"total_tokens": 5, "input_tokens": 1, "output_tokens": 1}) == 5


def test_split_fields_are_summed_with_cache_counters():
    from agentcallgraph.graph.types import usage_tokens

    assert (
        usage_tokens(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 3,
            }
        )
        == 160
    )


def test_non_numeric_usage_values_are_ignored():
    from agentcallgraph.graph.types import usage_tokens

    assert usage_tokens({"input_tokens": "many", "output_tokens": None}) == 0
    assert usage_tokens(None) == 0


def test_budget_detector_flags_a_spike_without_turn_id(tmp_path):
    # Every event without a turn_id collapsed into turn 0, so len(turns) == 1 and
    # the detector returned before computing anything (issue #15).
    session = parse_generic_jsonl(_split_usage_log(tmp_path, False))

    findings = find_budget_anomalies(session, sigma_threshold=1.0)

    assert len(findings) == 1
    assert findings[0].details["tokens"] == 13000


def test_budget_detector_still_separates_real_turns(tmp_path):
    # Positive control: an explicit turn_id must still group the way it did.
    session = parse_generic_jsonl(_split_usage_log(tmp_path, True))

    findings = find_budget_anomalies(session, sigma_threshold=1.0)

    assert [f.details["turn_id"] for f in findings] == [2]
