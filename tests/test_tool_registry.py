import sqlite3
from pathlib import Path

from src.agent.tool_registry import execute_tool, get_tool_schemas
from src.config import load_config


def test_registry_exposes_exactly_three_tools() -> None:
    names = {schema["function"]["name"] for schema in get_tool_schemas()}

    assert names == {"sql_query", "python_exec", "summarize"}


def test_execute_unknown_tool_returns_error(tmp_path: Path) -> None:
    result = execute_tool("missing", {}, tmp_path / "x.db", load_config("configs/eval.yaml"))

    assert result.was_blocked is True
    assert result.tool_return.startswith("ERROR:")


def test_execute_blocked_tool_marks_blocked(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY)")

    result = execute_tool(
        "sql_query",
        {"query": "DROP TABLE sales"},
        db_path,
        load_config("configs/eval.yaml"),
    )

    assert result.was_blocked is True
    assert result.block_reason is not None


def test_execute_tool_validates_required_arguments(tmp_path: Path) -> None:
    result = execute_tool("python_exec", {}, tmp_path / "x.db", load_config("configs/eval.yaml"))

    assert result.was_blocked is True
    assert result.tool_return.startswith("ERROR:")
