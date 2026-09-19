# agent-call-graph

> Static analysis of AI agent session logs — detect wasteful tool call patterns, circular reasoning, and budget anomalies before they burn tokens.

## Problem

Modern AI coding agents (Claude Code, Codex, Cursor, OpenCode, Aider) routinely execute **thousands of tool calls** across multi-turn sessions. We have cost trackers ([agentcost](https://github.com/yunaremaia/agentcost)), log visualizers ([causetrace](https://github.com/milkoor/causetrace)), and trace debuggers ([agent-replay](https://github.com/Zijian-Ni/agent-replay)) — but **nothing that statically analyzes a session log to detect structural waste before the next run**.

Real patterns observed in production:
- **Redundant tool calls**: the same `(tool, args)` invoked 3-5× in a single turn (Kimi Code's dedup was merged Aug 2026, Hermes' per-turn dedup is still open).
- **Circular reasoning**: agent searches for a file → not found → searches again with slightly different pattern → loops.
- **Budget-blind execution**: a single `grep` that returns 50K tokens when a targeted `Read` would cost 500.
- **Cascading tool chains**: one expensive tool result feeds into 2-3 follow-up tools when a single compound call would suffice.
- **Zombie tool calls**: calls whose output is never consumed by any subsequent LLM reasoning step.

None of these show up in cost dashboards — you see the **bill**, not the **waste**.

## Solution

`agent-call-graph` ingests session logs from any major agent runtime (JSONL, SQLite, or exported transcript), reconstructs the tool-call DAG, and reports:

| Detection | Description | Impact |
|-----------|-------------|--------|
| `redundant_call` | Same `(tool, args)` executed >1× with identical result | Removes redundant compute |
| `circular_reasoning` | Agent retries same search/action after failure with no strategy change | Saves 10-50 wasted turns |
| `over_fetched` | Tool returns far more data than the agent actually uses | Identifies token waste |
| `unused_output` | Tool result produced but never referenced in subsequent reasoning | Dead computation |
| `expensive_chain` | Linear chain of N tools where 1 compound call would work | Reduces API round-trips |
| `budget_anomaly` | Single turn or session exceeds expected token budget by >3σ | Cost control |

### Input formats (v0.1)
- Claude Code: `~/.claude/projects/**/*.jsonl` (native)
- Codex: `~/.codex/sessions/**/rollout-*.jsonl` (native)
- Hermes: session DB exports (`*.jsonl`)
- OpenCode: `~/.local/share/opencode/log/*.log`
- Generic: any JSONL with `{tool_name, tool_input, tool_output, timestamp}` records

### Output formats
- **Text**: human-readable report with severity-ranked findings
- **JSON**: structured findings for CI integration (`agent-call-graph --format json session.jsonl > findings.json`)
- **SARIF**: GitHub Code Scanning integration
- **HTML**: interactive DAG viewer (like agent-replay but for structural analysis)

### Exit codes
- `0` — no actionable findings
- `1` — findings detected (CI gate)
- `2` — input error

## Proposed Architecture

```
agent-call-graph/
├── agentcallgraph/
│   ├── __init__.py
│   ├── cli.py              # entrypoint: agent-call-graph
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── claude_code.py  # Claude Code JSONL parser
│   │   ├── codex.py        # Codex rollout parser
│   │   ├── hermes.py       # Hermes session DB parser
│   │   ├── opencode.py     # OpenCode log parser
│   │   └── generic.py      # fallback JSONL parser
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py      # construct tool-call DAG from events
│   │   └── types.py        # ToolCallNode, Edge, SessionGraph
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── redundant.py    # same (tool, args) >1×
│   │   ├── circular.py     # retry loops with no strategy change
│   │   ├── over_fetch.py   # output size >> consumed size
│   │   ├── unused.py       # orphaned tool outputs
│   │   ├── chain.py        # expensive linear chains
│   │   └── budget.py       # statistical budget anomalies
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   ├── json.py
│   │   ├── sarif.py
│   │   └── html.py
│   └── config.py           # detector thresholds, input format config
├── tests/
│   ├── fixtures/           # synthetic session logs per runtime
│   ├── test_parsers.py
│   ├── test_graph.py
│   ├── test_detectors.py
│   └── test_reporters.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── README.md
└── LICENSE (MIT)
```

## Recommended Stack

- **Language**: Python 3.11+ (matches driftcheck/aipr ecosystem, broad agent-runtime compatibility)
- **Graph**: [networkx](https://networkx.org/) for DAG construction and cycle detection
- **CLI**: [click](https://click.palletsprojects.com/) (same as driftcheck)
- **Validation**: [pydantic](https://docs.pydantic.dev/) for input schema per runtime
- **Testing**: pytest with parametrized fixtures
- **CI**: GitHub Actions (Linux/macOS, Python 3.11-3.13)
- **Output**: stdlib `html` for viewer, `sarif-sdk` equivalent for SARIF

## Roadmap

### v0.1 — "First Light"
- [ ] Generic JSONL parser + Hermes native parser
- [ ] DAG builder with parent→child inference
- [ ] `redundant_call` detector
- [ ] `budget_anomaly` detector
- [ ] Text + JSON reporters
- [ ] 80%+ test coverage on synthetic fixtures
- [ ] `pip install agent-call-graph`

### v0.2 — "Runtime Coverage"
- [ ] Claude Code native parser
- [ ] Codex rollout parser
- [ ] OpenCode log parser
- [ ] `circular_reasoning` detector
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
- [ ] Hermes Agent skill/cron integration
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

[Yunare Maia](https://github.com/yunaremaia) — the agentic infrastructure behind [driftcheck](https://github.com/yunaremaia/driftcheck), [aipr](https://github.com/yunaremaia/aipr), [depscan](https://github.com/yunaremaia/depscan), and friends.
