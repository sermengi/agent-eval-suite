import subprocess
import sys

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
    assert "Total revenue by category" in completed.stdout
