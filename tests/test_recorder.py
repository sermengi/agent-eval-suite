from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.logging.recorder import (
    AdversarialRobustnessScore,
    ArgumentFaithfulnessScore,
    EvaluationRecord,
    ResultRecorder,
    ScoreRecord,
    TaskCompletionScore,
    ToolCallTraceRecord,
    ToolSelectionAccuracyScore,
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
            task_completion=TaskCompletionScore(
                score=1,
                rationale="The answer matches the reference.",
                judge_model="gpt-4o-mini",
                prompt_version="v1",
            ),
            tool_selection_accuracy=ToolSelectionAccuracyScore(
                score=1.0,
                rationale="The agent called the expected SQL tool.",
                expected_sequences=[["sql_query"], ["sql_query", "summarize"]],
                actual_sequence=["sql_query"],
            ),
            argument_faithfulness=ArgumentFaithfulnessScore(
                schema_score=1.0,
                schema_rationale="The SQL references valid schema elements.",
                intent_score=1.0,
                intent_rationale="The SQL answers the requested revenue breakdown.",
                final_score=1.0,
                judge_model="gpt-4o-mini",
                prompt_version="v1",
            ),
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
    assert record.scores.task_completion.score == 1
    assert record.scores.adversarial_robustness is None
    assert record.tool_call_trace[0].step == 1


def test_rejects_invalid_config_hash() -> None:
    record_data = make_record_data()
    record_data["config_hash"] = "not-a-sha"

    with pytest.raises(ValidationError, match="config_hash"):
        EvaluationRecord(**record_data)


def test_rejects_naive_timestamp() -> None:
    record_data = make_record_data()
    record_data["timestamp"] = datetime(2025, 3, 28, 16, 45)

    with pytest.raises(ValidationError, match="timestamp"):
        EvaluationRecord(**record_data)


def test_normalizes_aware_timestamp_to_utc() -> None:
    record_data = make_record_data()
    record_data["timestamp"] = datetime(
        2025,
        3,
        28,
        22,
        15,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )

    record = EvaluationRecord(**record_data)
    payload = json.loads(record.model_dump_json())

    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None
    assert datetime.fromisoformat(payload["timestamp"]) == datetime(
        2025,
        3,
        28,
        16,
        45,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize(
    ("score_model", "field_name", "payload"),
    [
        (
            TaskCompletionScore,
            "score",
            {
                "score": 2,
                "rationale": "Invalid binary score.",
                "judge_model": "gpt-4o-mini",
                "prompt_version": "v1",
            },
        ),
        (
            AdversarialRobustnessScore,
            "score",
            {
                "score": -1,
                "rationale": "Invalid binary score.",
                "detected_failure_modes": [],
            },
        ),
        (
            ToolSelectionAccuracyScore,
            "score",
            {
                "score": 1.1,
                "rationale": "Invalid unit interval score.",
                "expected_sequences": [["sql_query"]],
                "actual_sequence": ["sql_query"],
            },
        ),
        (
            ArgumentFaithfulnessScore,
            "schema_score",
            {
                "schema_score": -0.1,
                "schema_rationale": "Invalid unit interval score.",
                "intent_score": 1.0,
                "intent_rationale": "Valid intent score.",
                "final_score": 1.0,
                "judge_model": "gpt-4o-mini",
                "prompt_version": "v1",
            },
        ),
        (
            ArgumentFaithfulnessScore,
            "intent_score",
            {
                "schema_score": 1.0,
                "schema_rationale": "Valid schema score.",
                "intent_score": 1.2,
                "intent_rationale": "Invalid unit interval score.",
                "final_score": 1.0,
                "judge_model": "gpt-4o-mini",
                "prompt_version": "v1",
            },
        ),
        (
            ArgumentFaithfulnessScore,
            "final_score",
            {
                "schema_score": 1.0,
                "schema_rationale": "Valid schema score.",
                "intent_score": 1.0,
                "intent_rationale": "Valid intent score.",
                "final_score": 9.0,
                "judge_model": "gpt-4o-mini",
                "prompt_version": "v1",
            },
        ),
    ],
)
def test_rejects_out_of_range_scores(
    score_model: type,
    field_name: str,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=field_name):
        score_model(**payload)


def test_result_recorder_writes_jsonl_and_creates_parent_directory(tmp_path) -> None:
    output_path = tmp_path / "nested" / "results.jsonl"
    record_path = ResultRecorder(output_path).record(make_record())

    assert record_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["task_id"] == "task_001"
    parsed_timestamp = datetime.fromisoformat(payload["timestamp"])
    assert parsed_timestamp == datetime(2025, 3, 28, 16, 45, tzinfo=timezone.utc)


def test_result_recorder_uses_single_write_call(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "results.jsonl"
    writes: list[str] = []

    class WriteSpy:
        def __enter__(self) -> "WriteSpy":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def write(self, text: str) -> int:
            writes.append(text)
            return len(text)

    def open_spy(self: Path, *args: Any, **kwargs: Any) -> WriteSpy:
        assert self == output_path
        return WriteSpy()

    monkeypatch.setattr(Path, "open", open_spy)

    ResultRecorder(output_path).record(make_record())

    assert len(writes) == 1
    assert writes[0].endswith("\n")
    payload = json.loads(writes[0])
    assert payload["task_id"] == "task_001"
