from pathlib import Path

import pytest

from src.config import load_config


def test_loads_week1_config_values() -> None:
    config = load_config("configs/eval.yaml")

    assert config.seed == 42
    assert config.reference_date.isoformat() == "2025-01-01"
    assert config.database.path == (Path.cwd() / "data/agent_eval.sqlite").resolve()
    assert config.models.openai == "gpt-4o-mini"
    assert config.models.huggingface == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config.tools.python_timeout_seconds == 5
    assert config.tools.allowed_python_modules == ["math", "statistics", "json", "datetime"]


def test_missing_config_file_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)
