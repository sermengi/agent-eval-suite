from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DebugEventType = Literal[
    "run_started",
    "llm_request",
    "llm_response",
    "tool_call_parsed",
    "tool_executed",
    "final_response",
    "run_error",
]


class DebugEvent(BaseModel):
    """One timestamped event in an agent debug trace."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: DebugEventType
    step: int = Field(ge=0)
    timestamp: datetime
    title: str
    summary: str
    payload: dict[str, Any]

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require timezone-aware timestamps and normalize them to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


class DebugTrace(BaseModel):
    """Serializable debug trace for one agent run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    model_name: str
    user_message: str
    started_at: datetime
    completed_at: datetime | None
    events: list[DebugEvent]

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_trace_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware trace timestamps and normalize them to UTC."""

        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trace timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)
