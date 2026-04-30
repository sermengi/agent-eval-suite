from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.debugging.schemas import DebugTrace


class ToolCallTraceRecord(BaseModel):
    """Validated JSON record for one agent tool call."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(gt=0)
    tool_name: str
    arguments: dict[str, Any]
    tool_return: str
    was_blocked: bool
    block_reason: str | None


class ScoreRecord(BaseModel):
    """Validated per-dimension scores for one evaluation record."""

    model_config = ConfigDict(extra="forbid")

    task_completion: int | None = None
    task_completion_rationale: str | None = None
    tool_selection_accuracy: float | None = None
    argument_faithfulness_schema: float | None = None
    argument_faithfulness_intent: float | None = None
    argument_faithfulness_final: float | None = None
    adversarial_robustness: int | None = None

    @field_validator("task_completion", "adversarial_robustness")
    @classmethod
    def validate_binary_score(cls, value: int | None) -> int | None:
        """Validate nullable binary score fields."""

        if value is not None and value not in {0, 1}:
            raise ValueError("binary score must be 0 or 1")
        return value

    @field_validator(
        "tool_selection_accuracy",
        "argument_faithfulness_schema",
        "argument_faithfulness_intent",
        "argument_faithfulness_final",
    )
    @classmethod
    def validate_unit_interval_score(cls, value: float | None) -> float | None:
        """Validate nullable float score fields."""

        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        return value


class EvaluationRecord(BaseModel):
    """Validated structured log record for one task evaluation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: datetime
    model_name: str
    task_id: str
    task_category: str
    adversarial_type: str | None
    user_message: str
    tool_call_trace: list[ToolCallTraceRecord]
    final_response: str
    scores: ScoreRecord
    composite_score: float | None
    judge_model: str
    judge_prompt_versions: dict[str, str]
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    debug_trace: DebugTrace | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require timezone-aware timestamps and normalize them to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("composite_score")
    @classmethod
    def validate_composite_score(cls, value: float | None) -> float | None:
        """Validate nullable composite scores."""

        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("composite_score must be between 0.0 and 1.0")
        return value


class ResultRecorder:
    """Append structured evaluation records to a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        """Initialize the recorder with a JSONL output path."""

        self.output_path = Path(output_path)

    def record(self, record: EvaluationRecord) -> Path:
        """Append one evaluation record as JSONL and return the output path."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return self.output_path


def build_placeholder_scores(adversarial_type: str | None) -> ScoreRecord:
    """Build nullable Week 2 score placeholders for an evaluation record."""

    _ = adversarial_type
    return ScoreRecord()
