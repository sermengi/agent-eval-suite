from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A model-requested tool call."""

    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass(frozen=True)
class ModelMessage:
    """Normalized model response for the agent runner."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw_response: dict[str, Any] | None = None


class ModelClient(Protocol):
    """Protocol implemented by model clients used by the agent runner."""

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return the next model message for the conversation."""
