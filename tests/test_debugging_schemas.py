from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.debugging.schemas import DebugEvent, DebugTrace


def make_trace_data() -> dict[str, object]:
    """Build valid debug trace data for schema tests."""

    return {
        "run_id": "20260430-debug-001",
        "task_id": "task_001",
        "model_name": "gpt-4o-mini",
        "user_message": "What was total revenue by category?",
        "started_at": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        "completed_at": None,
        "events": [
            {
                "event_id": "evt-001",
                "event_type": "run_started",
                "step": 1,
                "timestamp": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                "title": "Run started",
                "summary": "Debug trace collection started.",
                "payload": {"config_hash": "a" * 64},
            },
            {
                "event_id": "evt-002",
                "event_type": "tool_executed",
                "step": 2,
                "timestamp": datetime(2025, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
                "title": "SQL query executed",
                "summary": "Read-only revenue query returned one row.",
                "payload": {
                    "tool_name": "sql_query",
                    "arguments": {"query": "SELECT SUM(revenue) FROM sales"},
                    "result_preview": "100.0",
                },
            },
        ],
    }


def test_valid_trace_serializes_and_deserializes() -> None:
    trace = DebugTrace(**make_trace_data())

    serialized = trace.model_dump_json()
    payload = json.loads(serialized)

    assert payload["run_id"] == "20260430-debug-001"
    assert payload["completed_at"] is None
    assert payload["events"][0]["event_type"] == "run_started"
    assert payload["events"][1]["payload"]["tool_name"] == "sql_query"

    restored = DebugTrace.model_validate_json(serialized)

    assert restored == trace
    assert isinstance(restored.events[0], DebugEvent)


def test_rejects_unknown_event_type() -> None:
    event_data = make_trace_data()["events"][0]
    assert isinstance(event_data, dict)
    event_data = {**event_data, "event_type": "unknown_event"}

    with pytest.raises(ValidationError, match="event_type"):
        DebugEvent(**event_data)


@pytest.mark.parametrize("field_name", ["run_id", "task_id", "model_name", "user_message"])
def test_rejects_missing_run_metadata(field_name: str) -> None:
    trace_data = make_trace_data()
    trace_data.pop(field_name)

    with pytest.raises(ValidationError, match=field_name):
        DebugTrace(**trace_data)
