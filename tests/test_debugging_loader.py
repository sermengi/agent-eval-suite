from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import EvalConfig, load_config
from src.debugging.loader import DebugRunLoader
from src.debugging.schemas import DebugEvent, DebugTrace
from src.logging.recorder import (
    ArgumentFaithfulnessScore,
    EvaluationRecord,
    ScoreRecord,
    TaskCompletionScore,
    ToolSelectionAccuracyScore,
)

VALID_CONFIG_HASH = "b" * 64
NOW = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_config(tmp_path: Path) -> EvalConfig:
    """Build a config that points loader tests at temporary artifacts."""

    task_path = tmp_path / "tasks.yaml"
    task_path.write_text(
        """
id: task_001
category: normal
description: What was total revenue?
reference_answer: Total revenue was $100.
expected_tool_sequence:
  - acceptable_sequences:
      - [sql_query]
adversarial_type: null
notes: ""
""",
        encoding="utf-8",
    )
    config = load_config("configs/eval.yaml")
    return config.model_copy(
        update={
            "tasks": config.tasks.model_copy(update={"paths": [task_path]}),
            "results": config.results.model_copy(
                update={"output_path": tmp_path / "results" / "runs.jsonl"}
            ),
            "debug": config.debug.model_copy(
                update={"output_dir": tmp_path / "results" / "debug_traces"}
            ),
        }
    )


def make_event(
    event_id: str,
    event_type: str,
    payload: dict[str, object],
    step: int = 1,
) -> DebugEvent:
    """Build a valid debug event for loader tests."""

    return DebugEvent(
        event_id=event_id,
        event_type=event_type,  # type: ignore[arg-type]
        step=step,
        timestamp=NOW,
        title=f"Event {step}",
        summary="Debug event.",
        payload=payload,
    )


def make_trace(
    *,
    run_id: str,
    task_id: str = "task_001",
    model_name: str = "gpt-4o-mini",
    events: list[DebugEvent] | None = None,
) -> DebugTrace:
    """Build a valid debug trace."""

    return DebugTrace(
        run_id=run_id,
        task_id=task_id,
        model_name=model_name,
        user_message="What was total revenue?",
        started_at=NOW,
        completed_at=NOW,
        events=(
            events
            if events is not None
            else [
                make_event(
                    "evt-001",
                    "tool_executed",
                    {
                        "tool_name": "sql_query",
                        "arguments": {"query": "SELECT SUM(revenue) FROM sales"},
                    },
                )
            ]
        ),
    )


def make_record(
    *,
    run_id: str,
    model_name: str = "gpt-4o-mini",
    task_id: str = "task_001",
    task_category: str = "normal",
    debug_trace: DebugTrace | None = None,
) -> EvaluationRecord:
    """Build a valid evaluation record."""

    return EvaluationRecord(
        run_id=run_id,
        timestamp=NOW,
        model_name=model_name,
        task_id=task_id,
        task_category=task_category,
        adversarial_type=None,
        user_message="What was total revenue?",
        tool_call_trace=[],
        final_response="Total revenue was $100.",
        scores=ScoreRecord(
            task_completion=TaskCompletionScore(
                score=1,
                rationale="The response answers the task.",
                judge_model="gpt-4o-mini",
                prompt_version="v1",
            ),
            tool_selection_accuracy=ToolSelectionAccuracyScore(
                score=1.0,
                rationale="The expected SQL tool was used.",
                expected_sequences=[["sql_query"]],
                actual_sequence=["sql_query"],
            ),
            argument_faithfulness=ArgumentFaithfulnessScore(
                schema_score=1.0,
                schema_rationale="The SQL matches schema expectations.",
                intent_score=1.0,
                intent_rationale="The SQL matches user intent.",
                final_score=1.0,
                judge_model="gpt-4o-mini",
                prompt_version="v1",
            ),
            adversarial_robustness=None,
        ),
        composite_score=None,
        judge_model="gpt-4o-mini",
        judge_prompt_versions={"tc": "v1", "af": "v1"},
        config_hash=VALID_CONFIG_HASH,
        debug_trace=debug_trace,
    )


def write_records(config: EvalConfig, records: list[EvaluationRecord]) -> None:
    """Write JSONL evaluation records for loader tests."""

    config.results.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.results.output_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def write_trace(config: EvalConfig, trace: DebugTrace) -> None:
    """Write one separate debug trace file."""

    config.debug.output_dir.mkdir(parents=True, exist_ok=True)
    (config.debug.output_dir / f"{trace.run_id}.json").write_text(
        trace.model_dump_json(),
        encoding="utf-8",
    )


def test_loads_run_with_separate_trace(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_records(config, [make_record(run_id="run-separate")])
    write_trace(config, make_trace(run_id="run-separate"))

    loader = DebugRunLoader(config)

    runs = loader.list_runs()
    assert [run.run_id for run in runs] == ["run-separate"]
    assert runs[0].trace_available is True
    assert runs[0].diagnostic_count == 0

    detail = loader.get_run("run-separate")
    assert detail is not None
    assert detail.trace is not None
    assert detail.trace.run_id == "run-separate"


def test_prefers_embedded_trace_when_record_contains_one(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    embedded = make_trace(
        run_id="run-embedded",
        events=[make_event("evt-embedded", "final_response", {"response": "Done."})],
    )
    separate = make_trace(
        run_id="run-embedded",
        events=[
            make_event(
                "evt-separate",
                "tool_executed",
                {"tool_name": "sql_query", "arguments": {"query": "SELECT 1"}},
            )
        ],
    )
    write_records(config, [make_record(run_id="run-embedded", debug_trace=embedded)])
    write_trace(config, separate)

    detail = DebugRunLoader(config).get_run("run-embedded")

    assert detail is not None
    assert detail.trace is not None
    assert [event.event_id for event in detail.trace.events] == ["evt-embedded"]


def test_loads_run_without_trace_with_missing_trace_diagnostic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_records(config, [make_record(run_id="run-missing")])

    detail = DebugRunLoader(config).get_run("run-missing")

    assert detail is not None
    assert detail.trace is None
    assert [diagnostic.title for diagnostic in detail.diagnostics] == ["Missing debug trace"]
    assert detail.max_severity == "error"


def test_malformed_separate_trace_adds_diagnostic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_records(config, [make_record(run_id="run-malformed")])
    config.debug.output_dir.mkdir(parents=True)
    (config.debug.output_dir / "run-malformed.json").write_text(
        json.dumps({"run_id": "run-malformed", "events": "not-a-list"}),
        encoding="utf-8",
    )

    detail = DebugRunLoader(config).get_run("run-malformed")

    assert detail is not None
    assert detail.trace is None
    assert [diagnostic.title for diagnostic in detail.diagnostics] == ["Malformed debug trace"]
    assert detail.max_severity == "error"


def test_invalid_bytes_trace_adds_malformed_trace_diagnostic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_records(config, [make_record(run_id="run-invalid-bytes")])
    config.debug.output_dir.mkdir(parents=True)
    (config.debug.output_dir / "run-invalid-bytes.json").write_bytes(b"\xff\xfe\xfa")

    detail = DebugRunLoader(config).get_run("run-invalid-bytes")

    assert detail is not None
    assert detail.trace is None
    assert [diagnostic.title for diagnostic in detail.diagnostics] == ["Malformed debug trace"]
    assert detail.max_severity == "error"


def test_filters_runs_by_model_task_category_and_severity(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    error_trace = make_trace(
        run_id="run-error",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        events=[
            make_event(
                "evt-python",
                "tool_executed",
                {"tool_name": "python_exec", "arguments": {"code": "print(1)"}},
            )
        ],
    )
    write_records(
        config,
        [
            make_record(run_id="run-ok", model_name="gpt-4o-mini"),
            make_record(
                run_id="run-error",
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                task_category="normal",
            ),
            make_record(
                run_id="run-missing",
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                task_id="task_999",
                task_category="adversarial",
            ),
        ],
    )
    write_trace(config, make_trace(run_id="run-ok"))
    write_trace(config, error_trace)

    loader = DebugRunLoader(config)

    assert [run.run_id for run in loader.list_runs(model="gpt-4o-mini")] == ["run-ok"]
    assert [run.run_id for run in loader.list_runs(task="task_001")] == [
        "run-ok",
        "run-error",
    ]
    assert [run.run_id for run in loader.list_runs(category="adversarial")] == ["run-missing"]
    assert [run.run_id for run in loader.list_runs(severity="error")] == [
        "run-error",
        "run-missing",
    ]
