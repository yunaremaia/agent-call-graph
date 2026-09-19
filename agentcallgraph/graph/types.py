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
        total = 0
        for e in self.events:
            if e.token_usage:
                total += e.token_usage.get("total_tokens", 0)
        return total
