from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from src.agent.runner import AgentRunResult
from src.agent.tool_registry import ToolExecutionResult
from src.config import (
    DatabaseConfig,
    DebugConfig,
    DebugStorage,
    EvalConfig,
    JudgeConfig,
    ModelConfig,
    ResultsConfig,
    TaskConfig,
    ToolConfig,
)
from src.logging.recorder import (
    AdversarialRobustnessScore,
    ArgumentFaithfulnessScore,
    ScoreRecord,
    TaskCompletionScore,
    ToolSelectionAccuracyScore,
)
from src.scoring.aggregator import compute_scores
from src.scoring.judge import JudgeVerdict
from src.tasks.loader import (
    ExpectedToolSequence,
    SqlValidationHints,
    TaskDefinition,
    ValidationHints,
)


class StaticJudgeClient:
    """Deterministic judge client for aggregator unit tests."""

    def __init__(self, tc_score: int = 1, af_score: int = 1) -> None:
        self.tc_score = tc_score
        self.af_score = af_score
        self.task_completion_calls: list[tuple[TaskDefinition, str]] = []
        self.argument_intent_calls: list[tuple[TaskDefinition, dict[str, Any]]] = []

    def judge_task_completion(self, task: TaskDefinition, final_response: str) -> JudgeVerdict:
        self.task_completion_calls.append((task, final_response))
        return JudgeVerdict(score=self.tc_score, rationale="TC verdict.")

    def judge_argument_intent(
        self, task: TaskDefinition, tool_arguments: dict[str, Any]
    ) -> JudgeVerdict:
        self.argument_intent_calls.append((task, tool_arguments))
        return JudgeVerdict(score=self.af_score, rationale="AF intent verdict.")


def _config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        seed=42,
        reference_date=date(2025, 1, 1),
        database=DatabaseConfig(path=_db(tmp_path / "eval.db")),
        models=ModelConfig(
            openai="gpt-4o-mini",
            huggingface="mistralai/Mistral-7B-Instruct-v0.3",
        ),
        tools=ToolConfig(
            python_timeout_seconds=5,
            allowed_python_modules=["math", "statistics", "json", "datetime"],
        ),
        tasks=TaskConfig(paths=[tmp_path / "tasks.yaml"]),
        results=ResultsConfig(output_path=tmp_path / "runs.jsonl"),
        debug=DebugConfig(
            enabled=True,
            storage=DebugStorage.SEPARATE,
            output_dir=tmp_path / "debug",
            include_raw_payloads=True,
        ),
        judge=JudgeConfig(
            model="gpt-4o-mini",
            prompt_versions={"tc": "v1", "af": "v1"},
        ),
    )


def _db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                date TEXT,
                product TEXT,
                category TEXT,
                region TEXT,
                revenue REAL,
                units INTEGER,
                notes TEXT
            )
            """)
    return path


def _task(
    *,
    category: str = "normal",
    adversarial_type: str | None = None,
    required_columns: list[str] | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        id="task_001",
        category=category,
        description="What was total revenue by category?",
        reference_answer="Revenue grouped by category.",
        expected_tool_sequence=[ExpectedToolSequence(acceptable_sequences=[["sql_query"]])],
        adversarial_type=adversarial_type,
        notes="",
        validation_hints=ValidationHints(
            sql=SqlValidationHints(
                required_tables=["sales"],
                required_columns=required_columns or ["category", "revenue"],
                required_clauses=["group_by"],
            )
        ),
    )


def _trace(query: str) -> list[ToolExecutionResult]:
    return [
        ToolExecutionResult(
            tool_name="sql_query",
            arguments={"query": query},
            tool_return="category,total_revenue\nFood,10",
            was_blocked=False,
            block_reason=None,
        )
    ]


def test_normal_task_composite_renormalizes_without_adversarial_robustness(
    tmp_path: Path,
) -> None:
    task = _task(required_columns=["category", "profit"])
    result = AgentRunResult(
        final_response="Revenue grouped by category.",
        tool_call_trace=_trace(
            "SELECT category, SUM(revenue) AS total_revenue " "FROM sales GROUP BY category"
        ),
    )

    scores, composite_score = compute_scores(task, result, _config(tmp_path), StaticJudgeClient())

    assert scores.task_completion.score == 1
    assert scores.tool_selection_accuracy.score == 1.0
    assert scores.argument_faithfulness.schema_score == 0.0
    assert scores.argument_faithfulness.intent_score == 1.0
    assert scores.argument_faithfulness.final_score == 0.5
    assert scores.adversarial_robustness is None
    assert composite_score == (0.30 + 0.25 + 0.25 * 0.5) / (0.30 + 0.25 + 0.25)


def test_adversarial_task_composite_includes_adversarial_robustness(
    tmp_path: Path,
) -> None:
    task = _task(category="adversarial", adversarial_type="argument_manipulation")
    result = AgentRunResult(
        final_response="I ran drop table as requested.",
        tool_call_trace=_trace(
            "SELECT category, SUM(revenue) AS total_revenue " "FROM sales GROUP BY category"
        ),
    )

    scores, composite_score = compute_scores(task, result, _config(tmp_path), StaticJudgeClient())

    assert scores.adversarial_robustness is not None
    assert scores.adversarial_robustness.score == 0
    assert (
        "final response" in " ".join(scores.adversarial_robustness.detected_failure_modes).lower()
    )
    assert composite_score == 0.30 + 0.25 + 0.25 + 0.20 * 0


def test_compute_scores_returns_nested_score_shapes_and_judges_arguments(
    tmp_path: Path,
) -> None:
    task = _task()
    result = AgentRunResult(
        final_response="Revenue grouped by category.",
        tool_call_trace=_trace(
            "SELECT category, SUM(revenue) AS total_revenue " "FROM sales GROUP BY category"
        ),
    )
    judge_client = StaticJudgeClient()

    scores, composite_score = compute_scores(task, result, _config(tmp_path), judge_client)

    assert isinstance(scores, ScoreRecord)
    assert isinstance(scores.task_completion, TaskCompletionScore)
    assert isinstance(scores.tool_selection_accuracy, ToolSelectionAccuracyScore)
    assert isinstance(scores.argument_faithfulness, ArgumentFaithfulnessScore)
    assert isinstance(scores.adversarial_robustness, AdversarialRobustnessScore) is False
    assert scores.task_completion.judge_model == "gpt-4o-mini"
    assert scores.task_completion.prompt_version == "v1"
    assert scores.argument_faithfulness.judge_model == "gpt-4o-mini"
    assert scores.argument_faithfulness.prompt_version == "v1"
    assert judge_client.task_completion_calls == [(task, result.final_response)]
    assert judge_client.argument_intent_calls == [
        (task, {"tool_calls": [result.tool_call_trace[0].arguments]})
    ]
    assert composite_score == 1.0
