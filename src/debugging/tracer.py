from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.config import DebugStorage, EvalConfig
from src.debugging.schemas import DebugEvent, DebugTrace


class DebugTracer(Protocol):
    """Interface for collecting debug events during an agent run."""

    def record(self, event: DebugEvent) -> None:
        """Record a debug event."""

    def finish(self, completed_at: datetime) -> DebugTrace | None:
        """Finalize the trace and return it when storage keeps one."""


class NoOpDebugTracer:
    """Debug tracer used when trace collection is disabled."""

    def record(self, event: DebugEvent) -> None:
        """Ignore a debug event."""

    def finish(self, completed_at: datetime) -> None:
        """Finish without producing a debug trace."""

        return None


class InMemoryDebugTracer:
    """Collect debug events in memory for later embedding or inspection."""

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        model_name: str,
        user_message: str,
        started_at: datetime | None = None,
    ) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.model_name = model_name
        self.user_message = user_message
        self.started_at = started_at or datetime.now(timezone.utc)
        self._events: list[DebugEvent] = []

    def record(self, event: DebugEvent) -> None:
        """Append a debug event while preserving insertion order."""

        self._events.append(event)

    def finish(self, completed_at: datetime) -> DebugTrace:
        """Return the completed in-memory debug trace."""

        return self._build_trace(completed_at)

    def _build_trace(self, completed_at: datetime) -> DebugTrace:
        return DebugTrace(
            run_id=self.run_id,
            task_id=self.task_id,
            model_name=self.model_name,
            user_message=self.user_message,
            started_at=self.started_at,
            completed_at=completed_at,
            events=list(self._events),
        )


class FileDebugTracer(InMemoryDebugTracer):
    """Collect debug events and write one JSON trace file on completion."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        task_id: str,
        model_name: str,
        user_message: str,
        started_at: datetime | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            task_id=task_id,
            model_name=model_name,
            user_message=user_message,
            started_at=started_at,
        )
        self.output_dir = output_dir

    def finish(self, completed_at: datetime) -> DebugTrace:
        """Write the completed debug trace to ``<output_dir>/<run_id>.json``."""

        trace = self._build_trace(completed_at)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.run_id}.json"
        output_path.write_text(trace.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return trace


def build_debug_tracer(
    config: EvalConfig,
    run_id: str,
    task_id: str,
    model_name: str,
    user_message: str,
) -> DebugTracer:
    """Build the configured debug tracer for one agent run."""

    if not config.debug.enabled or config.debug.storage == DebugStorage.DISABLED:
        return NoOpDebugTracer()
    if config.debug.storage == DebugStorage.SEPARATE:
        return FileDebugTracer(
            output_dir=config.debug.output_dir,
            run_id=run_id,
            task_id=task_id,
            model_name=model_name,
            user_message=user_message,
        )
    return InMemoryDebugTracer(
        run_id=run_id,
        task_id=task_id,
        model_name=model_name,
        user_message=user_message,
    )
