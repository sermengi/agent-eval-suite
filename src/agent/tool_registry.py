from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.tools import python_exec, sql_query, summarize
from src.config import EvalConfig


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result for one tool execution."""

    tool_name: str
    arguments: dict[str, Any]
    tool_return: str
    was_blocked: bool
    block_reason: str | None


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool schemas for the fixed tool set."""

    return [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "Execute a read-only SQL query against the local SQLite database.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "python_exec",
                "description": "Execute safe Python for calculations and transformations.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Summarize data as a table, bullets, or narrative.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "format": {"type": "string", "enum": ["table", "bullets", "narrative"]},
                    },
                    "required": ["data", "format"],
                },
            },
        },
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    db_path: str | Path,
    config: EvalConfig,
) -> ToolExecutionResult:
    """Execute a registered tool and convert expected failures into trace metadata."""

    validation_error = _validate_arguments(name, arguments)
    if validation_error:
        output = f"ERROR: {validation_error}"
    elif name == "sql_query":
        output = sql_query(str(arguments.get("query", "")), db_path)
    elif name == "python_exec":
        output = python_exec(
            str(arguments.get("code", "")),
            timeout_seconds=config.tools.python_timeout_seconds,
            allowed_modules=config.tools.allowed_python_modules,
        )
    elif name == "summarize":
        output = summarize(str(arguments.get("data", "")), str(arguments.get("format", "")))
    else:
        output = f"ERROR: Unknown tool: {name}"

    was_blocked = output.startswith("ERROR:")
    return ToolExecutionResult(
        tool_name=name,
        arguments=arguments,
        tool_return=output,
        was_blocked=was_blocked,
        block_reason=output if was_blocked else None,
    )


def _validate_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    required_by_tool = {
        "sql_query": {"query": str},
        "python_exec": {"code": str},
        "summarize": {"data": str, "format": str},
    }
    required = required_by_tool.get(name)
    if required is None:
        return None
    for key, expected_type in required.items():
        if key not in arguments:
            return f"Missing required argument for {name}: {key}"
        if not isinstance(arguments[key], expected_type):
            return f"Invalid argument type for {name}.{key}: expected {expected_type.__name__}"
    return None
