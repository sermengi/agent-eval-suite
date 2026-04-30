from __future__ import annotations

from datetime import date
from enum import StrEnum
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


class TaskConfig(BaseModel):
    """Task file configuration for evaluation runs."""

    paths: list[Path] = Field(min_length=1)


class ResultsConfig(BaseModel):
    """Result output configuration for evaluation runs."""

    output_path: Path


class DebugStorage(StrEnum):
    """Supported debug trace storage modes."""

    SEPARATE = "separate"
    EMBEDDED = "embedded"
    DISABLED = "disabled"


class DebugConfig(BaseModel):
    """Debug trace capture configuration."""

    enabled: bool
    storage: DebugStorage
    output_dir: Path
    include_raw_payloads: bool


class JudgeConfig(BaseModel):
    """LLM-as-judge configuration for evaluation scoring."""

    model: str
    prompt_versions: dict[str, str]


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int
    reference_date: date
    database: DatabaseConfig
    models: ModelConfig
    tools: ToolConfig
    tasks: TaskConfig
    results: ResultsConfig
    debug: DebugConfig
    judge: JudgeConfig


def load_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    """Load and validate an evaluation config from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = EvalConfig.model_validate(raw)
    base_dir = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    if not config.database.path.is_absolute():
        database = config.database.model_copy(
            update={"path": (base_dir / config.database.path).resolve()}
        )
        config = config.model_copy(update={"database": database})
    if any(not task_path.is_absolute() for task_path in config.tasks.paths):
        tasks = config.tasks.model_copy(
            update={
                "paths": [
                    task_path if task_path.is_absolute() else (base_dir / task_path).resolve()
                    for task_path in config.tasks.paths
                ]
            }
        )
        config = config.model_copy(update={"tasks": tasks})
    if not config.results.output_path.is_absolute():
        results = config.results.model_copy(
            update={"output_path": (base_dir / config.results.output_path).resolve()}
        )
        config = config.model_copy(update={"results": results})
    if not config.debug.output_dir.is_absolute():
        debug = config.debug.model_copy(
            update={"output_dir": (base_dir / config.debug.output_dir).resolve()}
        )
        config = config.model_copy(update={"debug": debug})
    return config


def allowed_modules(config: EvalConfig) -> Sequence[str]:
    """Return the configured Python sandbox module allowlist."""

    return tuple(config.tools.allowed_python_modules)
