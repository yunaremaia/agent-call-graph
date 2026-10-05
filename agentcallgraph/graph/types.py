"""Core data types for agent-call-graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"
    SYSTEM = "system"


#: Fields that add up to a total when a log does not pre-compute one.
_SPLIT_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def usage_tokens(usage: dict[str, Any] | None) -> int:
    """Total tokens in one usage record, whatever key names the log format uses.

    A pre-computed ``total_tokens`` wins when present. Otherwise the split
    fields are summed -- those are the ones the generic parser documents and the
    ones exporters actually write, so reading ``total_tokens`` alone reported 0
    for every real session log (issue #15).
    """
    if not isinstance(usage, dict):
        return 0
    keys = ("total_tokens",) if "total_tokens" in usage else _SPLIT_TOKEN_KEYS
    return sum(v for v in (usage.get(k, 0) for k in keys) if isinstance(v, (int, float)))


@dataclass
class Event:
    """A single event in an agent session."""

    event_id: str
    event_type: EventType
    timestamp: float
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | str | None = None
    token_usage: dict[str, int] | None = None
    parent_event_id: str | None = None
    turn_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call_key(self) -> str | None:
        """Canonical key for dedup: tool_name + sorted JSON args."""
        if self.event_type != EventType.TOOL_CALL or not self.tool_name:
            return None
        import json

        args = self.tool_input or {}
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"{self.tool_name}:{canonical}"


@dataclass
class Session:
    """A parsed agent session."""

    session_id: str
    source_format: str
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[Event]:
        return [e for e in self.events if e.event_type == EventType.TOOL_CALL]

    @property
    def duration_seconds(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return self.events[-1].timestamp - self.events[0].timestamp

    @property
    def total_tokens(self) -> int:
        return sum(usage_tokens(e.token_usage) for e in self.events)
