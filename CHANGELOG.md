# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- CLI regression tests covering `--fail-on-findings` exit codes end-to-end via subprocess
- `tests/test_dependencies.py`: asserts every runtime dependency declared in
  `[project.dependencies]` is actually imported by the package (parsed with `ast`)

### Changed

### Fixed

- Link separate JSONL and Hermes SQLite tool results to their calls before
  classifying redundant calls. Missing or ambiguous results no longer produce
  an unsupported "same result" warning (#14).

- Removed `networkx>=3.0` and `pydantic>=2.0` from `[project.dependencies]`. No
  module under `agentcallgraph/` imported either of them, so installing the
  package pulled 13.7 MB of unused libraries into the user's environment.

## [Initial Release]

- Initial project release
