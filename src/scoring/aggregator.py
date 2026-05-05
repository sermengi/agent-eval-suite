from __future__ import annotations

from typing import Any

from src.agent.runner import AgentRunResult
from src.config import EvalConfig
from src.logging.recorder import (
    ArgumentFaithfulnessScore,
    ScoreRecord,
    TaskCompletionScore,
)
from src.scoring.judge import JudgeClient
from src.scoring.rule_based import (
    score_adversarial_robustness,
    score_tool_selection,
    validate_argument_schema,
)
from src.tasks.loader import TaskDefinition

COMPOSITE_WEIGHTS = {
    "task_completion": 0.30,
    "tool_selection_accuracy": 0.25,
    "argument_faithfulness": 0.25,
    "adversarial_robustness": 0.20,
}


def compute_scores(
    task: TaskDefinition,
    result: AgentRunResult,
    config: EvalConfig,
    judge_client: JudgeClient,
) -> tuple[ScoreRecord, float]:
    """Compute nested dimension scores and the weighted composite score."""

    task_completion_verdict = judge_client.judge_task_completion(
        task, result.final_response
    )
    tool_selection_score = score_tool_selection(task, result.tool_call_trace)
    schema_score, schema_rationale = validate_argument_schema(
        task,
        result.tool_call_trace,
        config.database.path,
        config.tools.allowed_python_modules,
    )
    intent_verdict = judge_client.judge_argument_intent(
        task, _tool_arguments_payload(result)
    )
    argument_faithfulness = ArgumentFaithfulnessScore(
        schema_score=schema_score,
        schema_rationale=schema_rationale,
        intent_score=float(intent_verdict.score),
        intent_rationale=intent_verdict.rationale,
        final_score=(schema_score + float(intent_verdict.score)) / 2,
        judge_model=config.judge.model,
        prompt_version=config.judge.prompt_versions["af"],
    )
    adversarial_robustness = score_adversarial_robustness(
        task,
        result.tool_call_trace,
        final_response=result.final_response,
    )
    scores = ScoreRecord(
        task_completion=TaskCompletionScore(
            score=task_completion_verdict.score,
            rationale=task_completion_verdict.rationale,
            judge_model=config.judge.model,
            prompt_version=config.judge.prompt_versions["tc"],
        ),
        tool_selection_accuracy=tool_selection_score,
        argument_faithfulness=argument_faithfulness,
        adversarial_robustness=adversarial_robustness,
    )
    return scores, _composite_score(scores)


def _tool_arguments_payload(result: AgentRunResult) -> dict[str, Any]:
    return {"tool_calls": [tool_call.arguments for tool_call in result.tool_call_trace]}


def _composite_score(scores: ScoreRecord) -> float:
    available_scores = {
        "task_completion": float(scores.task_completion.score),
        "tool_selection_accuracy": scores.tool_selection_accuracy.score,
        "argument_faithfulness": scores.argument_faithfulness.final_score,
    }
    if scores.adversarial_robustness is not None:
        available_scores["adversarial_robustness"] = float(
            scores.adversarial_robustness.score
        )

    total_weight = sum(COMPOSITE_WEIGHTS[name] for name in available_scores)
    weighted_score = sum(
        available_scores[name] * COMPOSITE_WEIGHTS[name]
        for name in available_scores
    )
    return weighted_score / total_weight
