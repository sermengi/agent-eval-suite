import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.generate_db import generate_database
from src.agent.runner import AgentRunner
from src.config import load_config
from src.inference.fake_client import FakeModelClient


def test_runner_completes_fake_end_to_end_task(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    generate_database(config)

    result = AgentRunner(FakeModelClient(), config).run("What was total revenue by category?")

    assert result.final_response
    assert len(result.tool_call_trace) == 1
    assert result.tool_call_trace[0].tool_name == "sql_query"
    assert result.tool_call_trace[0].was_blocked is False


def test_run_eval_cli_fake_client(tmp_path) -> None:
    config_path = tmp_path / "eval.yaml"
    raw_config = yaml.safe_load(open("configs/eval.yaml", encoding="utf-8"))
    raw_config["database"]["path"] = str(tmp_path / "cli.sqlite")
    raw_config["tasks"] = {"paths": [str(Path.cwd() / "tasks/normal/tasks_001_020.yaml")]}
    raw_config["results"] = {"output_path": str(tmp_path / "runs.jsonl")}
    raw_config["judge"] = {
        "model": "gpt-4o-mini",
        "prompt_versions": {"tc": "v1", "af": "v1"},
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            str(config_path),
            "--client",
            "fake",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Logged task_001" in completed.stdout
    assert str(tmp_path / "runs.jsonl") in completed.stdout

    jsonl_path = tmp_path / "runs.jsonl"
    assert jsonl_path.exists()
    record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["task_id"] == "task_001"
    assert record["model_name"] == "fake"
    assert record["tool_call_trace"][0]["step"] == 1
    assert record["tool_call_trace"][0]["tool_name"] == "sql_query"
    assert record["scores"]["task_completion"] is None
    assert record["scores"]["task_completion_rationale"] is None
    assert record["scores"]["tool_selection_accuracy"] is None
    assert record["scores"]["argument_faithfulness_schema"] is None
    assert record["scores"]["argument_faithfulness_intent"] is None
    assert record["scores"]["argument_faithfulness_final"] is None
    assert record["scores"]["adversarial_robustness"] is None
    assert record["composite_score"] is None
    assert len(record["config_hash"]) == 64
