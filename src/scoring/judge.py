from __future__ import annotations

import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from src.config import EvalConfig
from src.tasks.loader import TaskDefinition


class JudgeVerdict(BaseModel):
    """Structured binary verdict returned by a scoring judge."""

    model_config = ConfigDict(extra="forbid", strict=True)

    score: int
    rationale: str

    @field_validator("score")
    @classmethod
    def validate_binary_score(cls, value: int) -> int:
        """Validate that judge scores are binary."""

        if value not in {0, 1}:
            raise ValueError("judge score must be 0 or 1")
        return value


class JudgeClient(Protocol):
    """Interface for task completion and argument-intent judges."""

    def judge_task_completion(
        self, task: TaskDefinition, final_response: str
    ) -> JudgeVerdict:
        """Judge whether a final response completes the task."""

    def judge_argument_intent(
        self, task: TaskDefinition, tool_arguments: dict[str, Any]
    ) -> JudgeVerdict:
        """Judge whether tool arguments match the task intent."""


class FakeJudgeClient:
    """Deterministic judge client for local tests and offline development."""

    def judge_task_completion(
        self, task: TaskDefinition, final_response: str
    ) -> JudgeVerdict:
        """Pass non-empty, non-error final responses."""

        normalized = final_response.strip().lower()
        if not normalized:
            return JudgeVerdict(score=0, rationale="Final response was empty.")
        if _looks_like_error(normalized):
            return JudgeVerdict(score=0, rationale="Final response contained an error.")
        return JudgeVerdict(
            score=1,
            rationale=f"Non-error final response accepted for {task.id}.",
        )

    def judge_argument_intent(
        self, task: TaskDefinition, tool_arguments: dict[str, Any]
    ) -> JudgeVerdict:
        """Pass when task hint concepts appear in serialized tool arguments."""

        argument_text = _argument_text(tool_arguments)
        concepts = _hint_concepts(task)
        missing = [concept for concept in concepts if concept not in argument_text]
        if missing:
            return JudgeVerdict(
                score=0,
                rationale="Missing task hint concept(s): " + ", ".join(missing),
            )
        return JudgeVerdict(
            score=1,
            rationale="Tool arguments included task hint concepts: " + ", ".join(concepts),
        )


class OpenAIJudgeClient:
    """OpenAI-backed JSON judge client for TC and AF intent scoring."""

    def __init__(self, config: EvalConfig, prompts_dir: str | Path = "prompts") -> None:
        """Create a judge client using OPENAI_API_KEY from the environment."""

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIJudgeClient")

        from openai import OpenAI

        self._client = OpenAI()
        self._model = config.judge.model
        self._prompt_versions = dict(config.judge.prompt_versions)
        self._prompts_dir = Path(prompts_dir)

    def judge_task_completion(
        self, task: TaskDefinition, final_response: str
    ) -> JudgeVerdict:
        """Judge whether a final response completes the task."""

        prompt = self._load_prompt("tc")
        content = _render_prompt(
            prompt,
            {
                "task_description": task.description,
                "reference_answer": task.reference_answer,
                "final_response": final_response,
            },
        )
        return self._judge(content)

    def judge_argument_intent(
        self, task: TaskDefinition, tool_arguments: dict[str, Any]
    ) -> JudgeVerdict:
        """Judge whether tool arguments match the task intent."""

        prompt = self._load_prompt("af")
        content = _render_prompt(
            prompt,
            {
                "task_description": task.description,
                "reference_answer": task.reference_answer,
                "tool_arguments": json.dumps(tool_arguments, sort_keys=True),
            },
        )
        return self._judge(content)

    def _load_prompt(self, dimension: str) -> str:
        version = self._prompt_versions[dimension]
        prompt_path = self._prompts_dir / f"judge_{dimension}_{version}.txt"
        return prompt_path.read_text(encoding="utf-8")

    def _judge(self, prompt: str) -> JudgeVerdict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a blind evaluation judge. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return _parse_judge_json(content)


def _looks_like_error(normalized_response: str) -> bool:
    return normalized_response.startswith("error") or "error:" in normalized_response


def _argument_text(tool_arguments: dict[str, Any]) -> str:
    return json.dumps(tool_arguments, sort_keys=True, default=str).lower()


def _hint_concepts(task: TaskDefinition) -> list[str]:
    concepts: list[str] = []
    hints = task.validation_hints
    if hints and hints.sql:
        concepts.extend(hints.sql.required_tables or [])
        concepts.extend(hints.sql.required_columns or [])
    if hints and hints.summarize and hints.summarize.required_format:
        concepts.append(hints.summarize.required_format)
    if hints and hints.python:
        concepts.extend(hints.python.required_variables or [])
        concepts.extend(hints.python.required_modules or [])
    return [_normalize_concept(concept) for concept in concepts if concept]


def _normalize_concept(concept: str) -> str:
    return concept.strip().lower()


def _render_prompt(prompt: str, values: dict[str, str]) -> str:
    rendered = prompt
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _parse_judge_json(content: str) -> JudgeVerdict:
    try:
        raw = json.loads(content)
    except JSONDecodeError as exc:
        return JudgeVerdict(
            score=0,
            rationale=f"Invalid judge JSON: {exc.msg}.",
        )

    try:
        score = int(raw["score"])
        rationale = str(raw["rationale"])
        return JudgeVerdict(score=score, rationale=rationale)
    except (KeyError, TypeError, ValueError) as exc:
        return JudgeVerdict(
            score=0,
            rationale=f"Invalid judge JSON schema: {exc}.",
        )
