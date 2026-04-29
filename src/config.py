from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatabaseConfig(BaseModel):
    """Database-related runtime configuration."""

    path: Path


class ModelConfig(BaseModel):
    """Model names used by the evaluation suite."""

    openai: str
    huggingface: str


class ToolConfig(BaseModel):
    """Safety and runtime settings for agent tools."""

    python_timeout_seconds: int = Field(gt=0)
    allowed_python_modules: list[str]


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int
    reference_date: date
    database: DatabaseConfig
    models: ModelConfig
    tools: ToolConfig


def load_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    """Load and validate an evaluation config from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = EvalConfig.model_validate(raw)
    if not config.database.path.is_absolute():
        base_dir = (
            config_path.parent.parent
            if config_path.parent.name == "configs"
            else config_path.parent
        )
        database = config.database.model_copy(
            update={"path": (base_dir / config.database.path).resolve()}
        )
        config = config.model_copy(update={"database": database})
    return config


def allowed_modules(config: EvalConfig) -> Sequence[str]:
    """Return the configured Python sandbox module allowlist."""

    return tuple(config.tools.allowed_python_modules)
