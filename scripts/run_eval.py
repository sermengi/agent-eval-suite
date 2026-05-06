from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_db import generate_database  # noqa: E402
from src.agent.runner import AgentRunner  # noqa: E402
from src.config import DebugStorage, EvalConfig, JudgeConfig, load_config  # noqa: E402
from src.debugging.tracer import build_debug_tracer  # noqa: E402
from src.inference.fake_client import FakeModelClient  # noqa: E402
from src.inference.openai_client import OpenAIModelClient  # noqa: E402
from src.logging.recorder import (  # noqa: E402
    EvaluationRecord,
    ResultRecorder,
    ToolCallTraceRecord,
)
from src.scoring.aggregator import compute_scores  # noqa: E402
from src.scoring.judge import FakeJudgeClient, OpenAIJudgeClient  # noqa: E402
from src.tasks.loader import load_tasks  # noqa: E402


def _config_hash(config_path: Path) -> str:
    """Return the SHA-256 hash of the raw config file bytes."""

    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _run_id(timestamp: datetime) -> str:
    """Build a compact run identifier from a UTC date and random suffix."""

    return f"{timestamp:%Y%m%d}-{uuid4().hex[:6]}"


def _scoring_config(config: EvalConfig, judge_client_name: str) -> EvalConfig:
    """Return a config copy with explicit judge provenance for score records."""

    if judge_client_name != "fake":
        return config
    return config.model_copy(
        update={
            "judge": JudgeConfig(
                model="fake",
                prompt_versions=config.judge.prompt_versions,
            )
        }
    )


def main() -> None:
    """Run one configured task and record nested Week 3 scores."""

    parser = argparse.ArgumentParser(description="Run one configured evaluation task.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--client", choices=["fake", "openai"], default="fake")
    parser.add_argument("--judge-client", choices=["fake", "openai"], default="fake")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if not config.database.path.exists():
        generate_database(config)

    client = (
        FakeModelClient()
        if args.client == "fake"
        else OpenAIModelClient(model=config.models.openai)
    )
    judge_client = FakeJudgeClient() if args.judge_client == "fake" else OpenAIJudgeClient(config)
    scoring_config = _scoring_config(config, args.judge_client)
    model_name = "fake" if args.client == "fake" else config.models.openai
    tasks = load_tasks(config.tasks.paths)
    if not tasks:
        raise RuntimeError("No tasks configured for evaluation.")
    task = tasks[0]

    timestamp = datetime.now(timezone.utc)
    run_id = _run_id(timestamp)
    debug_tracer = build_debug_tracer(
        config=config,
        run_id=run_id,
        task_id=task.id,
        model_name=model_name,
        user_message=task.description,
    )
    result = AgentRunner(
        client,
        config,
        debug_tracer=debug_tracer,
        run_id=run_id,
        task_id=task.id,
        task_category=task.category,
        adversarial_type=task.adversarial_type,
    ).run(task.description)
    scores, composite_score = compute_scores(task, result, scoring_config, judge_client)
    record = EvaluationRecord(
        run_id=run_id,
        timestamp=timestamp,
        model_name=model_name,
        task_id=task.id,
        task_category=task.category,
        adversarial_type=task.adversarial_type,
        user_message=task.description,
        tool_call_trace=[
            ToolCallTraceRecord(
                step=step,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                tool_return=tool_call.tool_return,
                was_blocked=tool_call.was_blocked,
                block_reason=tool_call.block_reason,
            )
            for step, tool_call in enumerate(result.tool_call_trace, start=1)
        ],
        final_response=result.final_response,
        scores=scores,
        composite_score=composite_score,
        judge_model=scoring_config.judge.model,
        judge_prompt_versions=scoring_config.judge.prompt_versions,
        config_hash=_config_hash(config_path),
        debug_trace=result.debug_trace if config.debug.storage == DebugStorage.EMBEDDED else None,
    )
    output_path = ResultRecorder(config.results.output_path).record(record)
    print(f"Logged {task.id} to {output_path}")


if __name__ == "__main__":
    main()
