from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.logging.recorder import (
    EvaluationRecord,
    ResultRecorder,
    ScoreRecord,
    ToolCallTraceRecord,
)

VALID_CONFIG_HASH = "a" * 64


def make_record_data() -> dict[str, object]:
    """Build valid evaluation record data for recorder tests."""

    return {
        "run_id": "20250328-b7e2d4",
        "timestamp": datetime(2025, 3, 28, 16, 45, tzinfo=timezone.utc),
        "model_name": "gpt-4o-mini",
        "task_id": "task_001",
        "task_category": "normal",
        "adversarial_type": None,
        "user_message": "What was total revenue by category?",
        "tool_call_trace": [
            ToolCallTraceRecord(
                step=1,
                tool_name="sql_query",
                arguments={"query": "SELECT category, SUM(revenue) FROM sales GROUP BY category"},
                tool_return="Electronics: 100.0",
                was_blocked=False,
                block_reason=None,
            )
        ],
        "final_response": "Total revenue by category is ...",
        "scores": ScoreRecord(
            task_completion=None,
            task_completion_rationale=None,
            tool_selection_accuracy=None,
            argument_faithfulness_schema=None,
            argument_faithfulness_intent=None,
            argument_faithfulness_final=None,
            adversarial_robustness=None,
        ),
        "composite_score": None,
        "judge_model": "gpt-4o-mini",
        "judge_prompt_versions": {"tc": "v1", "af": "v1"},
        "config_hash": VALID_CONFIG_HASH,
    }


def make_record() -> EvaluationRecord:
    """Build a valid evaluation record for recorder tests."""

    return EvaluationRecord(**make_record_data())


def test_valid_full_record_with_nullable_scores_validates() -> None:
    record = make_record()

    assert record.task_id == "task_001"
    assert record.scores.task_completion is None
    assert record.tool_call_trace[0].step == 1


def test_rejects_invalid_config_hash() -> None:
    record_data = make_record_data()
    record_data["config_hash"] = "not-a-sha"

    with pytest.raises(ValidationError, match="config_hash"):
        EvaluationRecord(**record_data)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("task_completion", 2),
        ("adversarial_robustness", -1),
        ("tool_selection_accuracy", 1.1),
        ("argument_faithfulness_schema", -0.1),
        ("argument_faithfulness_intent", 1.2),
        ("argument_faithfulness_final", 9.0),
    ],
)
def test_rejects_out_of_range_scores(field_name: str, value: int | float) -> None:
    with pytest.raises(ValidationError, match=field_name):
        ScoreRecord(**{field_name: value})


def test_result_recorder_writes_jsonl_and_creates_parent_directory(tmp_path) -> None:
    output_path = tmp_path / "nested" / "results.jsonl"
    record_path = ResultRecorder(output_path).record(make_record())

    assert record_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["task_id"] == "task_001"
    assert payload["timestamp"] == "2025-03-28T16:45:00Z"
