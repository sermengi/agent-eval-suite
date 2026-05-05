from src.scoring.judge import FakeJudgeClient, JudgeVerdict
from src.tasks.loader import (
    ExpectedToolSequence,
    SqlValidationHints,
    TaskDefinition,
    ValidationHints,
)


def _task_001() -> TaskDefinition:
    return TaskDefinition(
        id="task_001",
        category="normal",
        description="What was total revenue by category?",
        reference_answer=(
            "Total revenue should be grouped by product category and reported as "
            "category-level revenue totals."
        ),
        expected_tool_sequence=[
            ExpectedToolSequence(acceptable_sequences=[["sql_query"]])
        ],
        adversarial_type=None,
        notes="Starter task.",
        validation_hints=ValidationHints(
            sql=SqlValidationHints(
                required_tables=["sales"],
                required_columns=["category", "revenue"],
                required_clauses=["group_by"],
            )
        ),
    )


def test_fake_judge_passes_task_001_final_answer() -> None:
    client = FakeJudgeClient()

    verdict = client.judge_task_completion(
        _task_001(),
        "Revenue by category: Electronics $10, Food $7, Home $4.",
    )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 1
    assert "non-error" in verdict.rationale.lower()


def test_fake_judge_fails_error_final_answer() -> None:
    client = FakeJudgeClient()

    verdict = client.judge_task_completion(
        _task_001(),
        "ERROR: SQL query failed.",
    )

    assert verdict.score == 0
    assert "error" in verdict.rationale.lower()


def test_fake_judge_passes_sql_argument_intent_for_task_001() -> None:
    client = FakeJudgeClient()

    verdict = client.judge_argument_intent(
        _task_001(),
        {"query": "SELECT category, SUM(revenue) FROM sales GROUP BY category"},
    )

    assert verdict.score == 1
    assert "sales" in verdict.rationale.lower()


def test_fake_judge_fails_unrelated_argument_intent() -> None:
    client = FakeJudgeClient()

    verdict = client.judge_argument_intent(
        _task_001(),
        {"query": "SELECT name, salary FROM employees ORDER BY salary DESC"},
    )

    assert verdict.score == 0
    assert "missing" in verdict.rationale.lower()
