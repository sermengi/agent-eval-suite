from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from src.config import EvalConfig
from src.ui.app import create_debug_ui_app
from tests.test_debugging_loader import (
    make_config,
    make_event,
    make_record,
    make_trace,
    write_records,
    write_trace,
)


def make_ui_config(tmp_path: Path) -> EvalConfig:
    """Create fixture artifacts for debug UI route tests."""

    config = make_config(tmp_path)
    write_records(
        config,
        [
            make_record(run_id="run-complete"),
            make_record(
                run_id="run-missing",
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                task_id="task_999",
                task_category="adversarial",
            ),
        ],
    )
    write_trace(
        config,
        make_trace(
            run_id="run-complete",
            events=[
                make_event(
                    "evt-started",
                    "run_started",
                    {"config_hash": "b" * 64},
                    step=1,
                ),
                make_event(
                    "evt-tool",
                    "tool_executed",
                    {
                        "tool_name": "sql_query",
                        "arguments": {"query": "SELECT SUM(revenue) FROM sales"},
                        "result_preview": "100.0",
                    },
                    step=2,
                ),
                make_event(
                    "evt-final",
                    "final_response",
                    {"response": "Total revenue was $100."},
                    step=3,
                ),
            ],
        ),
    )
    return config


def make_client(tmp_path: Path) -> TestClient:
    """Build a TestClient backed by temporary debug artifacts."""

    return TestClient(create_debug_ui_app(make_ui_config(tmp_path)))


def test_root_redirects_to_runs(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/runs"


def test_runs_page_renders_run_list_and_missing_trace_state(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/runs")

    assert response.status_code == 200
    assert "run-complete" in response.text
    assert "run-missing" in response.text
    assert "gpt-4o-mini" in response.text
    assert "Missing debug trace" in response.text
    assert 'class="badge severity-error"' in response.text
    assert "Trace" in response.text


def test_run_detail_page_renders_summary_timeline_and_raw_json(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/runs/run-complete")

    assert response.status_code == 200
    assert "run-complete" in response.text
    assert "Total revenue was $100." in response.text
    assert "sql_query" in response.text
    assert "Event 2" in response.text
    assert "Raw event JSON" in response.text
    assert response.text.index("Event 1") < response.text.index("Event 2")


def test_run_detail_page_returns_404_for_unknown_run(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/runs/not-a-run")

    assert response.status_code == 404


def test_api_runs_returns_serializable_summaries(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/api/runs", params={"severity": "error"})

    assert response.status_code == 200
    payload = response.json()
    assert [run["run_id"] for run in payload] == ["run-missing"]
    assert payload[0]["timestamp"] == "2025-01-01T12:00:00Z"
    assert payload[0]["diagnostics"][0]["title"] == "Missing debug trace"


def test_api_run_detail_returns_record_trace_and_trace_path(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/api/runs/run-complete")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-complete"
    assert payload["record"]["final_response"] == "Total revenue was $100."
    assert payload["trace"]["events"][1]["payload"]["tool_name"] == "sql_query"
    assert payload["trace_path"].endswith("run-complete.json")


def test_api_run_detail_returns_404_for_unknown_run(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/api/runs/not-a-run")

    assert response.status_code == 404


def test_debug_ui_entrypoint_runs_uvicorn_with_cli_arguments(monkeypatch) -> None:
    module = importlib.import_module("scripts.run_debug_ui")
    captured: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(module.uvicorn, "run", fake_run)
    module.main(["--config", "configs/eval.yaml", "--host", "0.0.0.0", "--port", "9001"])

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001
    assert captured["app"].title == "Agent Debug UI"
