from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import DebugStorage, load_config
from src.debugging.schemas import DebugEvent, DebugTrace
from src.debugging.tracer import (
    FileDebugTracer,
    InMemoryDebugTracer,
    NoOpDebugTracer,
    build_debug_tracer,
)


def make_event(event_id: str, step: int) -> DebugEvent:
    """Build a valid debug event for tracer tests."""

    return DebugEvent(
        event_id=event_id,
        event_type="llm_request",
        step=step,
        timestamp=datetime(2025, 1, 1, 12, 0, step, tzinfo=timezone.utc),
        title=f"Event {step}",
        summary=f"Debug event {step}.",
        payload={"step": step},
    )


def test_no_op_tracer_ignores_events_and_finish_returns_none() -> None:
    tracer = NoOpDebugTracer()

    tracer.record(make_event("evt-001", 1))
    trace = tracer.finish(datetime(2025, 1, 1, 12, 1, tzinfo=timezone.utc))

    assert trace is None


def test_in_memory_tracer_preserves_event_order() -> None:
    tracer = InMemoryDebugTracer(
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
        started_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    tracer.record(make_event("evt-001", 1))
    tracer.record(make_event("evt-002", 2))
    trace = tracer.finish(datetime(2025, 1, 1, 12, 2, tzinfo=timezone.utc))

    assert isinstance(trace, DebugTrace)
    assert [event.event_id for event in trace.events] == ["evt-001", "evt-002"]
    assert trace.completed_at == datetime(2025, 1, 1, 12, 2, tzinfo=timezone.utc)


def test_file_tracer_writes_trace_json_to_run_id_path(tmp_path: Path) -> None:
    tracer = FileDebugTracer(
        output_dir=tmp_path / "debug_traces",
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
        started_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    tracer.record(make_event("evt-001", 1))
    tracer.record(make_event("evt-002", 2))
    trace = tracer.finish(datetime(2025, 1, 1, 12, 2, tzinfo=timezone.utc))

    output_path = tmp_path / "debug_traces" / "run-001.json"
    assert output_path.exists()
    assert isinstance(trace, DebugTrace)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    restored = DebugTrace.model_validate(payload)
    assert restored.run_id == "run-001"
    assert restored.task_id == "task_001"
    assert [event.event_id for event in restored.events] == ["evt-001", "evt-002"]


def test_build_debug_tracer_returns_no_op_when_disabled() -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(update={"debug": config.debug.model_copy(update={"enabled": False})})

    tracer = build_debug_tracer(
        config=config,
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
    )

    assert isinstance(tracer, NoOpDebugTracer)


def test_build_debug_tracer_returns_no_op_for_disabled_storage() -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"debug": config.debug.model_copy(update={"storage": DebugStorage.DISABLED})}
    )

    tracer = build_debug_tracer(
        config=config,
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
    )

    assert isinstance(tracer, NoOpDebugTracer)


def test_build_debug_tracer_returns_file_tracer_for_separate_storage(tmp_path: Path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"debug": config.debug.model_copy(update={"output_dir": tmp_path})}
    )

    tracer = build_debug_tracer(
        config=config,
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
    )

    assert isinstance(tracer, FileDebugTracer)


def test_build_debug_tracer_uses_in_memory_for_embedded_storage(tmp_path: Path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={
            "debug": config.debug.model_copy(
                update={"storage": DebugStorage.EMBEDDED, "output_dir": tmp_path}
            )
        }
    )
    tracer = build_debug_tracer(
        config=config,
        run_id="run-001",
        task_id="task_001",
        model_name="gpt-4o-mini",
        user_message="What was total revenue by category?",
    )

    tracer.record(make_event("evt-001", 1))
    trace = tracer.finish(datetime(2025, 1, 1, 12, 1, tzinfo=timezone.utc))

    assert isinstance(tracer, InMemoryDebugTracer)
    assert isinstance(trace, DebugTrace)
    assert not (tmp_path / "run-001.json").exists()
