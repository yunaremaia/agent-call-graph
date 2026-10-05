"""CLI regression tests, exercised through real subprocess invocations.

Why subprocess and not ``click.testing.CliRunner``: ``CliRunner`` catches
unhandled exceptions and reports ``exit_code == 1`` for them, so a command that
raises ``SystemExit(1)`` deliberately and a command that explodes with an
``AttributeError`` look identical to it. That is exactly how ``raise
click.Exit(1)`` (PR #25, issue #22) survived a green test suite: the exit code
was right by accident while the real traceback was being swallowed.

Running the CLI as a child process is the only way to observe the actual
process exit status and the actual stderr. See ``test_fail_on_findings_exits_1_cleanly``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# The strict, non-Python JSON reader. `json.loads` cannot be the gate here: it
# accepts NaN/Infinity as an extension, which is exactly how issue #32 hid.
NODE = shutil.which("node")

# A committed fixture known to produce exactly one redundant_call finding.
FINDINGS_SESSION = FIXTURES / "redundant_ls.jsonl"

# A committed fixture that produces no findings at all.
CLEAN_SESSION = FIXTURES / "clean_session.jsonl"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a child process and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "agentcallgraph.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # the exit code is the assertion, not an exception
    )


def assert_no_traceback(result: subprocess.CompletedProcess[str]) -> None:
    """Fail if the child process died from an unhandled exception.

    ``SystemExit(1)`` and an unhandled ``AttributeError`` both surface as
    ``exit_code == 1``; only the absence of a traceback tells them apart.
    """
    assert "Traceback" not in result.stderr, result.stderr
    assert "AttributeError" not in result.stderr, result.stderr


def test_fail_on_findings_exits_1_cleanly():
    """--fail-on-findings with findings: exit 1 and no unhandled exception.

    Regression guard for issue #22. With ``raise click.Exit(1)`` this raised
    ``AttributeError: Exit`` (``click.Exit`` does not exist), which escaped as a
    raw traceback and still happened to exit 1 -- indistinguishable by exit code
    alone.
    """
    result = run_cli(str(FINDINGS_SESSION), "--fail-on-findings")

    assert result.returncode == 1
    assert_no_traceback(result)
    assert "redundant_call" in result.stdout


def test_without_fail_on_findings_exits_0():
    """The same findings session without the flag is a successful run."""
    result = run_cli(str(FINDINGS_SESSION))

    assert result.returncode == 0
    assert_no_traceback(result)
    assert "redundant_call" in result.stdout


def test_no_fail_on_findings_flag_exits_0():
    """The explicit --no-fail-on-findings flag also exits 0."""
    result = run_cli(str(FINDINGS_SESSION), "--no-fail-on-findings")

    assert result.returncode == 0
    assert_no_traceback(result)


def test_clean_session_with_fail_on_findings_exits_0():
    """--fail-on-findings must not fail the run when there is nothing to report."""
    result = run_cli(str(CLEAN_SESSION), "--fail-on-findings")

    assert result.returncode == 0
    assert_no_traceback(result)
    assert "No findings" in result.stdout


def test_json_format_with_fail_on_findings_still_emits_valid_json():
    """A non-zero exit must not cost the caller the report on stdout."""
    result = run_cli(str(FINDINGS_SESSION), "--format", "json", "--fail-on-findings")

    assert result.returncode == 1
    assert_no_traceback(result)

    payload = json.loads(result.stdout)
    assert payload["source_format"] == "generic_jsonl"
    assert payload["total_tool_calls"] == 3
    assert [f["type"] for f in payload["findings"]] == ["redundant_call"]
    assert payload["findings"][0]["details"]["call_count"] == 3


def test_json_format_clean_session_with_fail_on_findings():
    """Same JSON contract on the clean session, with a zero exit."""
    result = run_cli(str(CLEAN_SESSION), "--format", "json", "--fail-on-findings")

    assert result.returncode == 0
    assert_no_traceback(result)
    assert json.loads(result.stdout)["findings"] == []


def test_invalid_threshold_exits_2():
    """Bad option value is a usage error: exit 2, no traceback (README contract)."""
    result = run_cli(str(FINDINGS_SESSION), "--threshold", "notanumber")

    assert result.returncode == 2
    assert_no_traceback(result)
    assert "Invalid value for '--threshold'" in result.stderr


def test_missing_session_file_exits_2():
    """A nonexistent path is a usage error: exit 2, no traceback."""
    result = run_cli("/nonexistent/session.jsonl")

    assert result.returncode == 2
    assert_no_traceback(result)
    assert "does not exist" in result.stderr


def test_invalid_format_choice_exits_2():
    """An unknown --format value is a usage error: exit 2, no traceback."""
    result = run_cli(str(FINDINGS_SESSION), "--format", "yaml")

    assert result.returncode == 2
    assert_no_traceback(result)
    assert "Invalid value for '--format'" in result.stderr


def test_help_exits_0():
    """--help is a successful run regardless of the session file."""
    result = run_cli("--help")

    assert result.returncode == 0
    assert_no_traceback(result)
    assert "--fail-on-findings" in result.stdout


# --- issue #32: the JSON report must be valid JSON for STRICT parsers ---------
#
# `json.loads` accepts the bare literals `NaN` / `Infinity` / `-Infinity` as a
# Python extension, so a `json.loads` round-trip proves nothing -- that is how
# this defect survived a green suite. `node -e "JSON.parse"` is the real gate.

NONFINITE_SESSION = FIXTURES / "bad_timestamps.jsonl"

# A non-finite literal as a JSON *value*: not preceded or followed by a word
# character, so it cannot be the inside of a quoted string such as "NaN-ish".
NONFINITE_LITERAL = re.compile(r"(?<![\w\"])(NaN|-?Infinity)(?![\w\"])")


def assert_strictly_valid_json(raw: str) -> None:
    """Assert `raw` is JSON that a strict, non-Python parser accepts."""
    assert not NONFINITE_LITERAL.search(raw), f"non-JSON literal in report: {raw}"
    if not NODE:
        pytest.skip("node is not installed; token-absence check already ran")
    proc = subprocess.run(
        [NODE, "-e", "JSON.parse(process.argv[1])", raw],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"strict parser rejected the report: {proc.stderr}"


def test_json_report_has_no_nonfinite_literals():
    """Issue #32: a hand-written NaN and a 1e999 overflow must not reach stdout."""
    result = run_cli(str(NONFINITE_SESSION), "--format", "json")

    assert result.returncode == 0
    assert_no_traceback(result)
    assert_strictly_valid_json(result.stdout)


def test_json_report_parses_under_a_strict_parser():
    """Every output path in the report is a real JSON value, not a Python float."""
    result = run_cli(str(NONFINITE_SESSION), "--format", "json")

    assert_strictly_valid_json(result.stdout)
    # json.loads is the lenient reader, but here it confirms the sanitized values
    # are null -- the shape choice the fix makes for an unrepresentable float.
    payload = json.loads(result.stdout)
    assert payload["duration_seconds"] is None
    details = {f["type"]: f["details"] for f in payload["findings"]}
    assert details["redundant_call"]["first_timestamp"] == 1.0
    assert details["redundant_call"]["last_timestamp"] is None


def test_text_report_does_not_print_nan():
    """Issue #32: `Duration: nans` is not a duration. The key still prints."""
    result = run_cli(str(NONFINITE_SESSION))

    assert result.returncode == 0
    assert_no_traceback(result)
    duration_line = next(
        line for line in result.stdout.splitlines() if line.startswith("Duration:")
    )
    assert duration_line == "Duration: unknown"
    assert "nan" not in result.stdout.lower()


def test_json_safe_maps_nonfinite_floats_to_none():
    """The guard itself, in-process: nested floats, lists and tuples included.

    In-process on purpose. The CLI tests run the child process, which coverage
    cannot attribute, so without this the new guard's lines are invisible to the
    coverage gate (pyproject: fail_under = 69).
    """
    from agentcallgraph.cli import _json_safe

    value = {
        "finite": 1.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": [{"deep": float("nan")}, ("tuple", float("inf")), 3],
        "int": 7,
        "str": "1e999",
    }
    safe = _json_safe(value)

    assert safe["finite"] == 1.5
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["neg_inf"] is None
    assert safe["nested"] == [{"deep": None}, ["tuple", None], 3]
    assert safe["int"] == 7 and safe["str"] == "1e999"
    # Key sets are untouched: consumers see the same schema either way.
    assert set(safe) == set(value)
    assert _json_safe("plain") == "plain"