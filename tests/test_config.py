from pathlib import Path

import pytest

from src.config import load_config


def test_loads_week1_config_values() -> None:
    config = load_config("configs/eval.yaml")

    assert config.seed == 42
    assert config.reference_date.isoformat() == "2025-01-01"
    assert config.database.path == (Path.cwd() / "data/agent_eval.sqlite").resolve()
    assert config.tasks.paths == [(Path.cwd() / "tasks/normal/tasks_001_020.yaml").resolve()]
    assert config.results.output_path == (Path.cwd() / "results/runs.jsonl").resolve()
    assert config.judge.model == "gpt-4o-mini"
    assert config.judge.prompt_versions == {"tc": "v1", "af": "v1"}
    assert config.models.openai == "gpt-4o-mini"
    assert config.models.huggingface == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config.tools.python_timeout_seconds == 5
    assert config.tools.allowed_python_modules == ["math", "statistics", "json", "datetime"]


def test_load_config_resolves_relative_paths_from_custom_config_directory(
    tmp_path: Path,
) -> None:
    task_file = tmp_path / "tasks.yaml"
    task_file.write_text("[]", encoding="utf-8")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
seed: 42
reference_date: "2025-01-01"
database:
  path: "data/db.sqlite"
tasks:
  paths:
    - "tasks.yaml"
results:
  output_path: "results/runs.jsonl"
judge:
  model: "gpt-4o-mini"
  prompt_versions:
    tc: "v1"
    af: "v1"
models:
  openai: "gpt-4o-mini"
  huggingface: "mistralai/Mistral-7B-Instruct-v0.3"
tools:
  python_timeout_seconds: 5
  allowed_python_modules:
    - math
    - statistics
    - json
    - datetime
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.database.path == (tmp_path / "data/db.sqlite").resolve()
    assert config.tasks.paths == [task_file.resolve()]
    assert config.results.output_path == (tmp_path / "results/runs.jsonl").resolve()


def test_missing_config_file_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)
