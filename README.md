# agent-call-graph

Analyze a coding session log and report the tool calls that were repeated for no reason, the loops that went nowhere, and the turns that blew the token budget.

[![CI](https://github.com/yunaremaia/agent-call-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/yunaremaia/agent-call-graph/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/yunaremaia/agent-call-graph)](https://github.com/yunaremaia/agent-call-graph/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/yunaremaia/agent-call-graph/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/yunaremaia/agent-call-graph)](https://github.com/yunaremaia/agent-call-graph/blob/main/LICENSE)

## Why

A long session produces a log of every tool call it made. Reading that log by hand does not scale, and a token-cost dashboard only shows the total bill — it cannot tell you *which* calls made the bill. Four questions the log answers and the bill does not:

- **Which calls were identical?** A tool invoked three times with the same arguments and the same result pays three times for one answer. The detector canonicalizes arguments as sorted JSON, so `{"b":1,"a":2}` and `{"a":2,"b":1}` count as the same call.
- **Where did the session loop?** Retrying a search that already failed is normal once and a bug three times. The loop detector normalizes arguments before comparing — numbers and file paths are replaced with placeholders — so `Read src/a.py` followed by `Read src/b.py` is recognized as the same step repeated, not two different steps.
- **Which turn cost too much?** Per-turn token spend is scored against the session's own mean and standard deviation, so an outlier is judged relative to the shape of that session rather than a hardcoded budget.
- **Can this run in CI?** The exit code is a real gate (`--fail-on-findings`), so a session that starts repeating itself fails a pipeline instead of quietly costing money.

Everything runs on the log file after the session is over: no live capture, no proxy, no instrumentation of the tool itself.

## Problem

Coding sessions routinely execute **thousands of tool calls** across many turns. Cost trackers ([agentcost](https://github.com/yunaremaia/agentcost)), log visualizers ([causetrace](https://github.com/milkoor/causetrace)) and trace debuggers ([agent-replay](https://github.com/Zijian-Ni/agent-replay)) all report what happened — but none of them read a finished session log and report *which* calls were structurally wasted.

Patterns that cost real tokens:

- **Redundant calls**: the same `(tool, args)` invoked 3-5× for one answer.
- **Circling**: a search fails, is retried with a slightly different pattern, then retried again.
- **Over-reading**: a single `grep` returning 50K tokens where a targeted `Read` would return 500.
- **Cascading chains**: one expensive result feeding 2-3 follow-up calls where a single compound call would do.

None of these show up in a cost dashboard — you see the **bill**, not the **waste**.

## Solution

`agent-call-graph` parses a session log, reconstructs the tool calls it contains, and reports:

| Detection | Status | What it flags |
|-----------|--------|---------------|
| `redundant_call` | implemented | Same `(tool, args)` called more than once. `warning` when the result was identical, `info` when the repeated call returned something different. |
| `circular_reasoning` | implemented | A sequence of 2–4 tool calls repeated `>=3` times in the last `--loop-window` calls, after arguments are normalized. |
| `budget_anomaly` | implemented | A turn whose token spend is more than `--threshold` standard deviations above the session mean. |
| `over_fetched` | planned | Tool returned far more data than the run consumed. |
| `unused_output` | planned | Tool result never referenced by any later step. |
| `expensive_chain` | planned | Linear chain of calls that one compound call would replace. |

### Install

`agent-call-graph` is not published on PyPI. Install it straight from the repository:

```bash
pip install git+https://github.com/yunaremaia/agent-call-graph.git
```

The console script is named `agent-call-graph` and the import package is `agentcallgraph`.

### Usage

```bash
# Human-readable report
agent-call-graph session.jsonl

# Structured findings for CI
agent-call-graph session.jsonl --format json > findings.json

# Fail the build when anything is found
agent-call-graph session.jsonl --fail-on-findings

# Tune the detectors
agent-call-graph session.jsonl --threshold 2.5          # budget sigma
agent-call-graph session.jsonl --loop-window 20 --loop-threshold 4

# Parse a Hermes session store instead of a JSONL log
agent-call-graph sessions.db --source hermes
```

### Input formats (v0.1)

- **Generic JSONL** — one JSON object per line; any of `tool_name`, `tool_input`, `tool_output`, `timestamp`, `turn_id`, `token_usage`. Unrecognized lines are skipped.
- **Hermes session store** — a SQLite session database with a `messages` table (`role`, `content`, `tool_calls`, `tool_call_id`, `timestamp`), or a JSONL export.

`--source` also accepts `claude-code`, `codex` and `opencode`, but those parsers are not implemented yet: today they fall back to the generic JSONL parser. Point the tool at a log in one of those formats only after confirming the generic parser reads it correctly.

### Output formats

- **Text** — severity-ranked findings (default).
- **JSON** — `session_id`, `source_format`, event/tool-call counts, duration, total tokens, and the full finding list.

### Exit codes

- `0` — no findings (also `0` when findings exist but `--fail-on-findings` was not passed)
- `1` — findings detected **and** `--fail-on-findings` was passed
- `2` — input error (for example, the session file does not exist)

## Roadmap

### v0.1 — "First Light"
- [x] Generic JSONL parser + Hermes session-store parser
- [x] `redundant_call` detector
- [x] `circular_reasoning` detector
- [x] `budget_anomaly` detector
- [x] Text + JSON reporters
- [ ] `pip install agent-call-graph` (not yet published — install from the repository)

### v0.2 — "Runtime Coverage"
- [ ] Claude Code native parser
- [ ] Codex rollout parser
- [ ] OpenCode log parser
- [ ] `unused_output` detector
- [ ] SARIF reporter (GitHub Code Scanning)

### v0.3 — "Optimization"
- [ ] `over_fetched` detector (output size vs consumed analysis)
- [ ] `expensive_chain` detector (compound-call suggestion)
- [ ] HTML interactive viewer
- [ ] `agent-call-graph --watch` real-time mode
- [ ] Diff mode: compare two session graphs

### v0.4 — "Integration"
- [ ] `agent-call-graph fix` — auto-suggest prompt/config changes
- [ ] CI action: `yunaremaia/action-call-graph`
- [ ] MCP server mode: expose findings to other tools

## Differentiation

| Tool | What it does | What `agent-call-graph` adds |
|------|-------------|------------------------------|
| [agent-replay](https://github.com/Zijian-Ni/agent-replay) | Record + replay traces | Static analysis *before* replay, no recording needed |
| [causetrace](https://github.com/milkoor/causetrace) | Causal tree visualization | Quantitative waste detection with severity ranking |
| [agentcost](https://github.com/yunaremaia/agentcost) | Token cost tracking | Structural root-cause of cost, not just reporting |
| [AgentFlow](https://github.com/patoles/agent-flow) | Real-time visualization | Post-session analysis, no live capture needed |
| [agentlint](https://github.com/kcotias/agentlint) | CLAUDE.md linting | Runtime session analysis, not static instruction linting |

## Contributing

1. Fork + branch from `main`
2. `pip install -e ".[dev]"`
3. `pytest` — must pass before PR
4. Add a fixture under `tests/fixtures/` for any new input format
5. PRs welcome — see [open issues](https://github.com/yunaremaia/agent-call-graph/issues)

## License

MIT — see [LICENSE](LICENSE)

## Built by

[Yunare Maia](https://github.com/yunaremaia) — the tooling behind [driftcheck](https://github.com/yunaremaia/driftcheck), [aipr](https://github.com/yunaremaia/aipr), [depscan](https://github.com/yunaremaia/depscan), and friends.
