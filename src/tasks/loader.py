from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

TaskCategory = Literal["normal", "ambiguous", "adversarial"]
AdversarialType = Literal[
    "prompt_injection",
    "goal_hijacking",
    "unsafe_delegation",
    "argument_manipulation",
]
ALLOWED_EXPECTED_TOOL_NAMES = frozenset({"sql_query", "python_exec", "summarize"})


class ExpectedToolSequence(BaseModel):
    """Allowed tool-call sequences for a task evaluation."""

    acceptable_sequences: list[list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_acceptable_sequences(self) -> ExpectedToolSequence:
        """Validate that each acceptable sequence contains at least one tool name."""

        for sequence in self.acceptable_sequences:
            if not sequence:
                raise ValueError("acceptable_sequences entries must be non-empty")
            if any(not tool_name for tool_name in sequence):
                raise ValueError("acceptable_sequences tool names must be non-empty")
            unknown_tool_names = [
                tool_name for tool_name in sequence if tool_name not in ALLOWED_EXPECTED_TOOL_NAMES
            ]
            if unknown_tool_names:
                raise ValueError("unknown expected tool name: " + ", ".join(unknown_tool_names))
        return self


class SqlValidationHints(BaseModel):
    """Rule-based SQL validation hints for argument faithfulness scoring."""

    model_config = ConfigDict(extra="forbid")

    required_tables: list[str] | None = None
    required_columns: list[str] | None = None
    required_clauses: list[str] | None = None


class SummarizeValidationHints(BaseModel):
    """Rule-based summarize validation hints for argument faithfulness scoring."""

    model_config = ConfigDict(extra="forbid")

    required_format: str | None = None


class PythonValidationHints(BaseModel):
    """Rule-based Python validation hints for argument faithfulness scoring."""

    model_config = ConfigDict(extra="forbid")

    required_variables: list[str] | None = None
    required_modules: list[str] | None = None


class ValidationHints(BaseModel):
    """Optional task-level hints for rule-based argument validation."""

    model_config = ConfigDict(extra="forbid")

    sql: SqlValidationHints | None = None
    summarize: SummarizeValidationHints | None = None
    python: PythonValidationHints | None = None


class TaskDefinition(BaseModel):
    """Validated YAML definition for one evaluation task."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: TaskCategory
    description: str
    reference_answer: str
    expected_tool_sequence: list[ExpectedToolSequence] = Field(min_length=1)
    adversarial_type: AdversarialType | None
    notes: str
    validation_hints: ValidationHints | None = None

    @model_validator(mode="after")
    def validate_adversarial_type(self) -> TaskDefinition:
        """Validate category-specific adversarial type requirements."""

        if self.category in {"normal", "ambiguous"} and self.adversarial_type is not None:
            raise ValueError("adversarial_type must be null for normal and ambiguous tasks")
        if self.category == "adversarial" and self.adversarial_type is None:
            raise ValueError("adversarial_type is required for adversarial tasks")
        return self


def load_task_file(path: str | Path) -> list[TaskDefinition]:
    """Load and validate one YAML task file.

    The YAML document may contain either a single task object or a list of task objects.
    """

    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as handle:
        raw_tasks = yaml.safe_load(handle)

    if isinstance(raw_tasks, list):
        return TypeAdapter(list[TaskDefinition]).validate_python(raw_tasks)
    return [TaskDefinition.model_validate(raw_tasks)]


def load_tasks(paths: Sequence[str | Path]) -> list[TaskDefinition]:
    """Load and validate tasks from multiple YAML files in path order."""

    tasks: list[TaskDefinition] = []
    for path in paths:
        tasks.extend(load_task_file(path))
    return tasks
