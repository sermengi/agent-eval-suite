import sqlite3
from pathlib import Path

from src.agent.tools import python_exec, sql_query, summarize


def _db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, revenue REAL)")
        conn.executemany(
            "INSERT INTO sales (region, revenue) VALUES (?, ?)",
            [("North", 10.0), ("South", 20.0)],
        )
    return path


def test_sql_query_allows_select(tmp_path: Path) -> None:
    result = sql_query(
        "SELECT region, revenue FROM sales ORDER BY revenue DESC", _db(tmp_path / "t.db")
    )

    assert "South" in result
    assert "20.0" in result


def test_sql_query_blocks_writes_and_multi_statement(tmp_path: Path) -> None:
    db_path = _db(tmp_path / "t.db")

    assert sql_query("DELETE FROM sales", db_path).startswith("ERROR:")
    assert sql_query("SELECT * FROM sales; DROP TABLE sales", db_path).startswith("ERROR:")


def test_python_exec_allows_safe_code() -> None:
    result = python_exec("import math\nprint(math.sqrt(81))")

    assert result.strip() == "9.0"


def test_python_exec_blocks_unsafe_code_and_timeouts() -> None:
    assert python_exec("import os\nprint(os.getcwd())").startswith("ERROR:")
    assert python_exec("open('/tmp/x', 'w')").startswith("ERROR:")
    assert python_exec("print(__builtins__.__dict__['__import__']('os').getcwd())").startswith(
        "ERROR:"
    )
    assert python_exec("print(().__class__.__mro__[1].__subclasses__())").startswith("ERROR:")
    assert python_exec("while True:\n    pass", timeout_seconds=1).startswith("ERROR:")


def test_summarize_formats_and_rejects_unknown() -> None:
    assert "- alpha" in summarize("alpha\nbeta", "bullets")
    assert "| data |" in summarize("alpha", "table")
    assert summarize("alpha", "narrative").startswith("Summary:")
    assert summarize("alpha", "json").startswith("ERROR:")
