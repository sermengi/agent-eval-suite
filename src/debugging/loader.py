from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from src.config import EvalConfig
from src.debugging.diagnostics import (
    DebugDiagnostic,
    DiagnosticSeverity,
    diagnose_missing_trace,
    diagnose_trace,
)
from src.debugging.schemas import DebugTrace
from src.logging.recorder import EvaluationRecord
from src.tasks.loader import ExpectedToolSequence, load_tasks

SeverityRank = Literal["none", "info", "warning", "error"]


class DebugRunSummary(BaseModel):
    """Compact debug run metadata for list views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    timestamp: datetime
    model_name: str
    task_id: str
    task_category: str
    adversarial_type: str | None
    trace_available: bool
    diagnostic_count: int
    max_severity: SeverityRank
    diagnostics: list[DebugDiagnostic]


class DebugRunDetail(DebugRunSummary):
    """Full debug run record for drill-down views."""

    record: EvaluationRecord
    trace: DebugTrace | None
    trace_path: Path | None


class DebugRunLoader:
    """Load evaluation records and their debug traces from result artifacts."""

    def __init__(self, config: EvalConfig) -> None:
        """Initialize the loader with an evaluation config."""

        self.config = config
        self._expected_sequences_by_task = self._load_expected_sequences_by_task()

    def list_runs(
        self,
        *,
        model: str | None = None,
        task: str | None = None,
        category: str | None = None,
        severity: DiagnosticSeverity | None = None,
    ) -> list[DebugRunSummary]:
        """Return run summaries filtered by model, task, category, or diagnostic severity."""

        details = [
            detail
            for detail in self._load_details()
            if self._matches_filters(detail, model, task, category, severity)
        ]
        return [self._to_summary(detail) for detail in details]

    def get_run(self, run_id: str) -> DebugRunDetail | None:
        """Return a full run detail by run ID, or None when it is absent."""

        for detail in self._load_details():
            if detail.run_id == run_id:
                return detail
        return None

    def _load_details(self) -> list[DebugRunDetail]:
        details: list[DebugRunDetail] = []
        for record in self._load_records():
            trace, trace_path, trace_diagnostics = self._load_trace(record)
            if trace is None:
                diagnostics = trace_diagnostics or diagnose_missing_trace(record.run_id)
            else:
                diagnostics = diagnose_trace(
                    trace,
                    expected_tool_sequences=self._expected_sequences_by_task.get(record.task_id),
                )
            details.append(
                DebugRunDetail(
                    run_id=record.run_id,
                    timestamp=record.timestamp,
                    model_name=record.model_name,
                    task_id=record.task_id,
                    task_category=record.task_category,
                    adversarial_type=record.adversarial_type,
                    trace_available=trace is not None,
                    diagnostic_count=len(diagnostics),
                    max_severity=_max_severity(diagnostics),
                    diagnostics=diagnostics,
                    record=record,
                    trace=trace,
                    trace_path=trace_path,
                )
            )
        return details

    def _load_records(self) -> list[EvaluationRecord]:
        results_path = self.config.results.output_path
        if not results_path.exists():
            return []

        records: list[EvaluationRecord] = []
        with results_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    records.append(EvaluationRecord.model_validate(payload))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise ValueError(
                        f"Malformed evaluation record at {results_path}:{line_number}: {exc}"
                    ) from exc
        return records

    def _load_trace(
        self,
        record: EvaluationRecord,
    ) -> tuple[DebugTrace | None, Path | None, list[DebugDiagnostic]]:
        if record.debug_trace is not None:
            return record.debug_trace, None, []

        trace_path = self.config.debug.output_dir / f"{record.run_id}.json"
        if not trace_path.exists():
            return None, trace_path, []

        try:
            trace_text = trace_path.read_text(encoding="utf-8")
            payload = json.loads(trace_text)
            return DebugTrace.model_validate(payload), trace_path, []
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            return (
                None,
                trace_path,
                [
                    DebugDiagnostic(
                        severity="error",
                        title="Malformed debug trace",
                        reason=f"Could not load debug trace from {trace_path}: {exc}",
                        event_id=None,
                    )
                ],
            )

    def _load_expected_sequences_by_task(self) -> dict[str, list[ExpectedToolSequence]]:
        try:
            tasks = load_tasks(self.config.tasks.paths)
        except FileNotFoundError:
            return {}
        return {task.id: task.expected_tool_sequence for task in tasks}

    @staticmethod
    def _matches_filters(
        detail: DebugRunDetail,
        model: str | None,
        task: str | None,
        category: str | None,
        severity: DiagnosticSeverity | None,
    ) -> bool:
        if model is not None and detail.model_name != model:
            return False
        if task is not None and detail.task_id != task:
            return False
        if category is not None and detail.task_category != category:
            return False
        if severity is not None and not any(
            diagnostic.severity == severity for diagnostic in detail.diagnostics
        ):
            return False
        return True

    @staticmethod
    def _to_summary(detail: DebugRunDetail) -> DebugRunSummary:
        return DebugRunSummary(
            run_id=detail.run_id,
            timestamp=detail.timestamp,
            model_name=detail.model_name,
            task_id=detail.task_id,
            task_category=detail.task_category,
            adversarial_type=detail.adversarial_type,
            trace_available=detail.trace_available,
            diagnostic_count=detail.diagnostic_count,
            max_severity=detail.max_severity,
            diagnostics=detail.diagnostics,
        )


def _max_severity(diagnostics: list[DebugDiagnostic]) -> SeverityRank:
    rank: dict[DiagnosticSeverity, int] = {"info": 1, "warning": 2, "error": 3}
    if not diagnostics:
        return "none"
    return max(diagnostics, key=lambda diagnostic: rank[diagnostic.severity]).severity
