from __future__ import annotations

import ast
import base64
import sqlite3
import subprocess
import sys
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Sequence

DEFAULT_ALLOWED_MODULES = ("math", "statistics", "json", "datetime")
BLOCKED_SQL_KEYWORDS = ("insert", "update", "delete", "drop", "alter", "create", "truncate")
BLOCKED_PYTHON_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "__builtins__",
}
BLOCKED_ATTRIBUTE_ROOTS = {"os", "pathlib", "shutil", "subprocess", "sys"}


def sql_query(query: str, db_path: str | Path) -> str:
    """Execute a read-only SQL query against the local SQLite database."""

    blocked_reason = _sql_block_reason(query)
    if blocked_reason:
        return f"ERROR: {blocked_reason}"

    try:
        with sqlite3.connect(Path(db_path)) as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            headers = [description[0] for description in cursor.description or []]
    except SQLiteError as exc:
        return f"ERROR: SQL execution failed: {exc}"

    if not rows:
        return "No rows returned."
    return _format_rows(headers, rows)


def python_exec(
    code: str,
    timeout_seconds: int = 5,
    allowed_modules: Sequence[str] = DEFAULT_ALLOWED_MODULES,
) -> str:
    """Run Python code in a subprocess after AST-based safety validation."""

    blocked_reason = _python_block_reason(code, allowed_modules)
    if blocked_reason:
        return f"ERROR: {blocked_reason}"

    try:
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        encoded_modules = base64.b64encode("\n".join(allowed_modules).encode("utf-8")).decode(
            "ascii"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _sandbox_wrapper(encoded_code, encoded_modules)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Python execution timed out after {timeout_seconds} seconds"

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown runtime error"
        return f"ERROR: Python execution failed: {stderr}"
    return completed.stdout


def _sandbox_wrapper(encoded_code: str, encoded_modules: str) -> str:
    return f"""
import base64
import builtins

code = base64.b64decode({encoded_code!r}).decode("utf-8")
allowed_modules = set(base64.b64decode({encoded_modules!r}).decode("utf-8").splitlines())

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in allowed_modules:
        raise ImportError(f"Import not allowed: {{name}}")
    return builtins.__import__(name, globals, locals, fromlist, level)

safe_builtins = {{
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}}

exec(compile(code, "<agent-python-exec>", "exec"), {{"__builtins__": safe_builtins}}, {{}})
"""


def summarize(data: str, format: str) -> str:
    """Summarize data in a constrained output format."""

    if format not in {"table", "bullets", "narrative"}:
        return "ERROR: Unsupported summary format"
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    if format == "table":
        body = "\n".join(f"| {line} |" for line in lines) if lines else "| No data |"
        return "| data |\n| --- |\n" + body
    if format == "bullets":
        return "\n".join(f"- {line}" for line in lines) if lines else "- No data"
    joined = " ".join(lines) if lines else "No data."
    return f"Summary: {joined}"


def _sql_block_reason(query: str) -> str | None:
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        return "Only SELECT queries are allowed"
    if ";" in normalized.rstrip(";"):
        return "Multiple SQL statements are not allowed"
    tokens = normalized.replace(";", " ").replace("(", " ").replace(")", " ").split()
    for keyword in BLOCKED_SQL_KEYWORDS:
        if keyword in tokens:
            return f"Blocked unsafe SQL keyword: {keyword.upper()}"
    return None


def _format_rows(headers: list[str], rows: list[tuple[object, ...]]) -> str:
    if not headers:
        return "\n".join(str(row) for row in rows)
    lines = [" | ".join(headers)]
    lines.extend(" | ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


def _python_block_reason(code: str, allowed_modules: Sequence[str]) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Invalid Python syntax: {exc.msg}"

    allowed = set(allowed_modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root not in allowed:
                    return f"Import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            if root not in allowed:
                return f"Import not allowed: {node.module}"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_PYTHON_NAMES:
            return f"Use of blocked Python name: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"Use of blocked Python dunder attribute: {node.attr}"
            root = _attribute_root(node)
            if root in BLOCKED_ATTRIBUTE_ROOTS:
                return f"Use of blocked Python module or attribute: {root}"
    return None


def _attribute_root(node: ast.Attribute) -> str | None:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None
