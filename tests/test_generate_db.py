import sqlite3
from pathlib import Path

from scripts.generate_db import generate_database
from src.config import EvalConfig, load_config


def _with_db_path(config: EvalConfig, db_path: Path) -> EvalConfig:
    return config.model_copy(
        update={"database": config.database.model_copy(update={"path": db_path})}
    )


def test_generate_database_creates_required_tables_and_counts(tmp_path: Path) -> None:
    config = _with_db_path(load_config("configs/eval.yaml"), tmp_path / "agent_eval.sqlite")

    generate_database(config)

    with sqlite3.connect(config.database.path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {"sales", "employees", "products"}
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 500
        assert conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 50
        assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 30
        assert (
            conn.execute("SELECT COUNT(*) FROM sales WHERE notes LIKE '%System:%'").fetchone()[0]
            == 3
        )


def test_generate_database_is_deterministic(tmp_path: Path) -> None:
    config = _with_db_path(load_config("configs/eval.yaml"), tmp_path / "agent_eval.sqlite")

    generate_database(config)
    with sqlite3.connect(config.database.path) as conn:
        first_rows = conn.execute(_representative_sales_query()).fetchall()

    generate_database(config)
    with sqlite3.connect(config.database.path) as conn:
        second_rows = conn.execute(_representative_sales_query()).fetchall()

    assert second_rows == first_rows


def _representative_sales_query() -> str:
    return """
        SELECT date, product, category, region, revenue, units, notes
        FROM sales
        ORDER BY id
        LIMIT 10
    """
