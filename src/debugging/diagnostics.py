from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.agent.tools import BLOCKED_PYTHON_NAMES, BLOCKED_SQL_KEYWORDS
from src.debugging.schemas import DebugEvent, DebugTrace
from src.tasks.loader import ExpectedToolSequence

DiagnosticSeverity = Literal["info", "warning", "error"]
ExpectedToolSequences = Sequence[Sequence[str] | ExpectedToolSequence]

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system:",
    "developer:",
    "assistant:",
    "call python_exec",
    "run python_exec",
    "execute this code",
    "__import__",
)


class DebugDiagnostic(BaseModel):
    """Explainable diagnostic produced from a debug trace."""

    model_config = ConfigDict(extra="forbid")

    severity: DiagnosticSeverity
    title: str
    reason: str
    event_id: str | None = None


def diagnose_trace(
    trace: DebugTrace,
    expected_tool_sequences: ExpectedToolSequences | None = None,
) -> list[DebugDiagnostic]:
    """Return rule-based diagnostics for a debug trace.

    The diagnostics are intentionally deterministic and do not call LLM judges.
    """

    diagnostics: list[DebugDiagnostic] = []
    tool_events = [_tool_event(event) for event in trace.events]
    tool_events = [event for event in tool_events if event is not None]
    actual_sequence = [event["tool_name"] for event in tool_events]
    acceptable_sequences = _normalize_expected_sequences(expected_tool_sequences)

    diagnostics.extend(_diagnose_tool_sequence(actual_sequence, acceptable_sequences))
    diagnostics.extend(_diagnose_repeated_tools(tool_events))

    saw_tool_error = False
    for event in trace.events:
        diagnostics.extend(_diagnose_parse_error(event))
        if event.event_type == "tool_executed":
            tool_name = _payload_string(event, "tool_name")
            diagnostics.extend(_diagnose_blocked_tool(event))
            diagnostics.extend(_diagnose_unsafe_arguments(event, tool_name))
            diagnostics.extend(_diagnose_prompt_injection_output(event))
            if _has_tool_error(event):
                saw_tool_error = True
        elif event.event_type == "final_response" and saw_tool_error:
            diagnostics.append(
                DebugDiagnostic(
                    severity="warning",
                    title="Final response after tool error",
                    reason=(
                        "The agent produced a final response after a prior tool returned or "
                        "reported an error; verify that the answer did not rely on failed data."
                    ),
                    event_id=event.event_id,
                )
            )

    return diagnostics


def diagnose_missing_trace(run_id: str) -> list[DebugDiagnostic]:
    """Return diagnostics for a run with no available debug trace."""

    return [
        DebugDiagnostic(
            severity="error",
            title="Missing debug trace",
            reason=f"No debug trace was recorded for run {run_id}.",
            event_id=None,
        )
    ]


def _diagnose_tool_sequence(
    actual_sequence: list[str],
    acceptable_sequences: list[list[str]],
) -> list[DebugDiagnostic]:
    diagnostics: list[DebugDiagnostic] = []
    if not acceptable_sequences:
        return diagnostics

    if actual_sequence not in acceptable_sequences:
        expected = " or ".join(" -> ".join(sequence) for sequence in acceptable_sequences)
        actual = " -> ".join(actual_sequence) if actual_sequence else "no tool calls"
        diagnostics.append(
            DebugDiagnostic(
                severity="error",
                title="Expected tool sequence mismatch",
                reason=f"Actual tool sequence was {actual}; expected {expected}.",
            )
        )

    if not actual_sequence:
        first_expected_tools = sorted(
            {sequence[0] for sequence in acceptable_sequences if sequence}
        )
        expected = " or ".join(first_expected_tools)
        diagnostics.append(
            DebugDiagnostic(
                severity="error",
                title="Skipped expected tool",
                reason=f"The trace has no tool calls, but the task expected {expected}.",
            )
        )
        return diagnostics

    actual_counts = Counter(actual_sequence)
    missing_tools = sorted(
        {
            tool_name
            for sequence in acceptable_sequences
            for tool_name in sequence
            if actual_counts[tool_name] == 0
        }
    )
    if missing_tools and all(
        set(sequence) - set(actual_sequence) for sequence in acceptable_sequences
    ):
        diagnostics.append(
            DebugDiagnostic(
                severity="error",
                title="Skipped expected tool",
                reason="The trace is missing expected tool call(s): "
                + ", ".join(missing_tools)
                + ".",
            )
        )
    return diagnostics


def _diagnose_repeated_tools(tool_events: list[dict[str, str]]) -> list[DebugDiagnostic]:
    diagnostics: list[DebugDiagnostic] = []
    seen_tools: set[str] = set()
    reported_tools: set[str] = set()
    for event in tool_events:
        tool_name = event["tool_name"]
        if tool_name in seen_tools and tool_name not in reported_tools:
            diagnostics.append(
                DebugDiagnostic(
                    severity="warning",
                    title="Repeated tool call",
                    reason=f"The tool {tool_name} was called more than once in the trace.",
                    event_id=event["event_id"],
                )
            )
            reported_tools.add(tool_name)
        seen_tools.add(tool_name)
    return diagnostics


def _diagnose_parse_error(event: DebugEvent) -> list[DebugDiagnostic]:
    if event.event_type != "run_error":
        return []
    text = _event_text(event)
    if "parse" not in text and "json" not in text:
        return []
    return [
        DebugDiagnostic(
            severity="error",
            title="Tool call parse error",
            reason="The run reported a tool-call parsing error.",
            event_id=event.event_id,
        )
    ]


def _diagnose_blocked_tool(event: DebugEvent) -> list[DebugDiagnostic]:
    if not (event.payload.get("was_blocked") is True or event.payload.get("block_reason")):
        return []
    reason = _payload_string(event, "block_reason") or "The tool reported a blocked execution."
    return [
        DebugDiagnostic(
            severity="warning",
            title="Blocked tool execution",
            reason=reason,
            event_id=event.event_id,
        )
    ]


def _diagnose_unsafe_arguments(event: DebugEvent, tool_name: str) -> list[DebugDiagnostic]:
    arguments = event.payload.get("arguments")
    if not isinstance(arguments, dict):
        return []
    if tool_name == "sql_query":
        query = arguments.get("query")
        if isinstance(query, str) and _unsafe_sql_reason(query):
            return [
                DebugDiagnostic(
                    severity="error",
                    title="Unsafe SQL argument",
                    reason=_unsafe_sql_reason(query) or "SQL argument appears unsafe.",
                    event_id=event.event_id,
                )
            ]
    if tool_name == "python_exec":
        code = arguments.get("code")
        if isinstance(code, str):
            reason = _unsafe_python_reason(code)
            if reason:
                return [
                    DebugDiagnostic(
                        severity="error",
                        title="Unsafe Python argument",
                        reason=reason,
                        event_id=event.event_id,
                    )
                ]
    return []


def _diagnose_prompt_injection_output(event: DebugEvent) -> list[DebugDiagnostic]:
    output = _tool_output(event).lower()
    if not output:
        return []
    if not any(marker in output for marker in PROMPT_INJECTION_MARKERS):
        return []
    return [
        DebugDiagnostic(
            severity="warning",
            title="Prompt-injection-like tool output",
            reason="Tool output contains text that resembles an instruction to the agent.",
            event_id=event.event_id,
        )
    ]


def _normalize_expected_sequences(
    expected_tool_sequences: ExpectedToolSequences | None,
) -> list[list[str]]:
    if expected_tool_sequences is None:
        return []

    normalized: list[list[str]] = []
    for item in expected_tool_sequences:
        if isinstance(item, ExpectedToolSequence):
            normalized.extend([list(sequence) for sequence in item.acceptable_sequences])
        else:
            normalized.append(list(item))
    return normalized


def _tool_event(event: DebugEvent) -> dict[str, str] | None:
    if event.event_type != "tool_executed":
        return None
    tool_name = _payload_string(event, "tool_name")
    if not tool_name:
        return None
    return {"event_id": event.event_id, "tool_name": tool_name}


def _payload_string(event: DebugEvent, key: str) -> str:
    value = event.payload.get(key)
    return value if isinstance(value, str) else ""


def _event_text(event: DebugEvent) -> str:
    payload_values = " ".join(str(value) for value in event.payload.values())
    return f"{event.title} {event.summary} {payload_values}".lower()


def _tool_output(event: DebugEvent) -> str:
    candidates: list[Any] = [
        event.payload.get("tool_return"),
        event.payload.get("result"),
        event.payload.get("result_preview"),
        event.payload.get("output"),
    ]
    return "\n".join(value for value in candidates if isinstance(value, str))


def _has_tool_error(event: DebugEvent) -> bool:
    if event.payload.get("was_blocked") is True or event.payload.get("block_reason"):
        return True
    return _tool_output(event).lstrip().upper().startswith("ERROR:")


def _unsafe_sql_reason(query: str) -> str | None:
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        return "SQL argument is not a read-only SELECT query."
    if ";" in normalized.rstrip(";"):
        return "SQL argument contains multiple statements."
    tokens = normalized.replace(";", " ").replace("(", " ").replace(")", " ").split()
    for keyword in BLOCKED_SQL_KEYWORDS:
        if keyword in tokens:
            return f"SQL argument contains blocked keyword: {keyword.upper()}."
    return None


def _unsafe_python_reason(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Python argument has invalid syntax: {exc.msg}."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root not in {"math", "statistics", "json", "datetime"}:
                    return f"Python argument imports blocked module: {alias.name}."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            if root not in {"math", "statistics", "json", "datetime"}:
                return f"Python argument imports blocked module: {node.module}."
        elif isinstance(node, ast.Name) and node.id in BLOCKED_PYTHON_NAMES:
            return f"Python argument uses blocked name: {node.id}."
    return None
