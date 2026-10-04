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
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

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