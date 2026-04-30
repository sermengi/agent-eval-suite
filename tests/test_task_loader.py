from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tasks.loader import load_task_file, load_tasks


def test_loads_starter_normal_task() -> None:
    tasks = load_task_file("tasks/normal/tasks_001_020.yaml")

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "task_001"
    assert task.category == "normal"
    assert task.description == "What was total revenue by category?"
    assert task.adversarial_type is None
    assert task.expected_tool_sequence[0].acceptable_sequences == [["sql_query"]]
    assert task.reference_answer
    assert task.notes


def test_accepts_single_object_yaml(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
id: task_single
category: ambiguous
description: Which region performed best?
reference_answer: The answer depends on whether performance means revenue or units.
expected_tool_sequence:
  - acceptable_sequences:
      - ["sql_query"]
      - ["sql_query", "summarize"]
adversarial_type: null
notes: Ambiguous metric wording.
""",
        encoding="utf-8",
    )

    tasks = load_tasks([task_path])

    assert len(tasks) == 1
    assert tasks[0].id == "task_single"
    assert tasks[0].expected_tool_sequence[0].acceptable_sequences == [
        ["sql_query"],
        ["sql_query", "summarize"],
    ]


def test_rejects_invalid_category(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
id: task_bad_category
category: experimental
description: Invalid category.
reference_answer: This should fail validation.
expected_tool_sequence:
  - acceptable_sequences:
      - ["sql_query"]
adversarial_type: null
notes: Invalid category test.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="category"):
        load_task_file(task_path)


@pytest.mark.parametrize(
    ("category", "adversarial_type"),
    [
        ("normal", "prompt_injection"),
        ("ambiguous", "goal_hijacking"),
        ("adversarial", None),
    ],
)
def test_validates_adversarial_type_by_category(
    tmp_path: Path, category: str, adversarial_type: str | None
) -> None:
    task_path = tmp_path / "task.yaml"
    type_value = "null" if adversarial_type is None else adversarial_type
    task_path.write_text(
        f"""
id: task_bad_adversarial_type
category: {category}
description: Invalid adversarial type relationship.
reference_answer: This should fail validation.
expected_tool_sequence:
  - acceptable_sequences:
      - ["sql_query"]
adversarial_type: {type_value}
notes: Invalid adversarial type test.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="adversarial_type"):
        load_task_file(task_path)


@pytest.mark.parametrize(
    "expected_tool_sequence",
    [
        "[]",
        "- acceptable_sequences: []",
        "- acceptable_sequences:\n    - []",
    ],
)
def test_rejects_invalid_expected_tool_sequence(
    tmp_path: Path, expected_tool_sequence: str
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""
id: task_bad_sequence
category: normal
description: Invalid expected sequence.
reference_answer: This should fail validation.
expected_tool_sequence:
  {expected_tool_sequence}
adversarial_type: null
notes: Invalid expected sequence test.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="expected_tool_sequence|acceptable_sequences"):
        load_task_file(task_path)


def test_rejects_unknown_expected_tool_name(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
id: task_unknown_tool
category: normal
description: Invalid expected tool name.
reference_answer: This should fail validation.
expected_tool_sequence:
  - acceptable_sequences:
      - ["sql_quer"]
adversarial_type: null
notes: Unknown expected tool name test.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown expected tool name"):
        load_task_file(task_path)
