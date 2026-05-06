import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.generate_db import generate_database
from src.agent.runner import AgentRunner
from src.config import load_config
from src.debugging.schemas import DebugTrace
from src.debugging.tracer import InMemoryDebugTracer
from src.inference.client import ModelMessage, ToolCall
from src.inference.fake_client import FakeModelClient


class FinalOnlyClient:
    """Fake client that answers without requesting tools."""

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return a direct final answer."""

        return ModelMessage(content="Direct answer", raw_response={"id": "final-only"})


class ParseErrorThenFinalClient:
    """Fake client that emits one malformed tool call before answering."""

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return a parse-error tool call, then a final answer."""

        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            return ModelMessage(
                tool_calls=[
                    ToolCall(
                        id="bad-call-1",
                        name="sql_query",
                        arguments={},
                        parse_error="Invalid JSON arguments",
                    )
                ],
                raw_response={"id": "parse-error"},
            )
        return ModelMessage(content="Handled parse error")


class EndlessToolClient:
    """Fake client that always requests a tool call."""

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return one harmless summarize tool call on every step."""

        return ModelMessage(
            tool_calls=[
                ToolCall(
                    id=f"endless-call-{len(messages)}",
                    name="summarize",
                    arguments={"data": "alpha", "format": "bullets"},
                )
            ],
            raw_response={"id": f"endless-{len(messages)}"},
        )


class RecordingDebugTracer(InMemoryDebugTracer):
    """In-memory tracer that remembers whether the runner finished it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.finish_count = 0
        self.completed_trace: DebugTrace | None = None

    def finish(self, completed_at: datetime) -> DebugTrace:
        """Record finish calls and return the completed trace."""

        self.finish_count += 1
        self.completed_trace = super().finish(completed_at)
        return self.completed_trace


def _memory_tracer(user_message: str = "debug task") -> RecordingDebugTracer:
    return RecordingDebugTracer(
        run_id="run-debug-test",
        task_id="task_debug",
        model_name="fake",
        user_message=user_message,
    )


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


def test_runner_emits_debug_events_for_normal_fake_run(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    generate_database(config)
    tracer = _memory_tracer("What was total revenue by category?")

    result = AgentRunner(FakeModelClient(), config, debug_tracer=tracer).run(
        "What was total revenue by category?"
    )

    assert result.final_response
    assert tracer.finish_count == 1
    assert tracer.completed_trace is not None
    trace = tracer.completed_trace
    assert [event.event_type for event in trace.events] == [
        "run_started",
        "llm_request",
        "llm_response",
        "tool_call_parsed",
        "tool_executed",
        "llm_request",
        "llm_response",
        "final_response",
    ]
    assert trace.events[2].payload["content"] is None
    assert trace.events[2].payload["tool_calls"] == [
        {"id": "fake-call-1", "name": "sql_query", "parse_error": None}
    ]
    assert trace.events[2].payload["raw_response"]["id"] == "fake-response-1"
    assert trace.events[4].payload["tool_name"] == "sql_query"
    assert trace.events[4].payload["was_blocked"] is False


def test_runner_snapshots_llm_request_messages_before_conversation_mutates(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    generate_database(config)
    tracer = _memory_tracer("What was total revenue by category?")

    AgentRunner(FakeModelClient(), config, debug_tracer=tracer).run(
        "What was total revenue by category?"
    )

    assert tracer.completed_trace is not None
    first_request = tracer.completed_trace.events[1]
    second_request = tracer.completed_trace.events[5]
    assert first_request.payload["messages"] == [
        {"role": "user", "content": "What was total revenue by category?"}
    ]
    assert first_request.payload["message_count"] == 1
    assert second_request.payload["message_count"] == 3
    assert first_request.payload["messages"] is not second_request.payload["messages"]


def test_runner_omits_raw_response_when_debug_config_excludes_raw_payloads(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={
            "database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"}),
            "debug": config.debug.model_copy(update={"include_raw_payloads": False}),
        }
    )
    generate_database(config)
    tracer = _memory_tracer("What was total revenue by category?")

    AgentRunner(FakeModelClient(), config, debug_tracer=tracer).run(
        "What was total revenue by category?"
    )

    assert tracer.completed_trace is not None
    llm_response_events = [
        event for event in tracer.completed_trace.events if event.event_type == "llm_response"
    ]
    assert llm_response_events
    assert all("raw_response" not in event.payload for event in llm_response_events)


def test_runner_emits_debug_events_for_parse_error_flow(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    tracer = _memory_tracer()

    result = AgentRunner(ParseErrorThenFinalClient(), config, debug_tracer=tracer).run("bad args")

    assert result.final_response == "Handled parse error"
    assert tracer.finish_count == 1
    assert tracer.completed_trace is not None
    trace = tracer.completed_trace
    assert [event.event_type for event in trace.events] == [
        "run_started",
        "llm_request",
        "llm_response",
        "tool_call_parsed",
        "tool_executed",
        "llm_request",
        "llm_response",
        "final_response",
    ]
    assert trace.events[3].payload["parse_error"] == "Invalid JSON arguments"
    assert trace.events[4].payload["tool_return"] == "ERROR: Invalid JSON arguments"
    assert trace.events[4].payload["was_blocked"] is True


def test_runner_emits_debug_events_for_final_answer_without_tool(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    tracer = _memory_tracer()

    result = AgentRunner(FinalOnlyClient(), config, debug_tracer=tracer).run("answer directly")

    assert result.final_response == "Direct answer"
    assert result.tool_call_trace == []
    assert tracer.finish_count == 1
    assert tracer.completed_trace is not None
    trace = tracer.completed_trace
    assert [event.event_type for event in trace.events] == [
        "run_started",
        "llm_request",
        "llm_response",
        "final_response",
    ]
    assert trace.events[-1].payload == {"final_response": "Direct answer"}


def test_runner_emits_run_error_for_max_step_termination(tmp_path) -> None:
    config = load_config("configs/eval.yaml")
    config = config.model_copy(
        update={"database": config.database.model_copy(update={"path": tmp_path / "db.sqlite"})}
    )
    tracer = _memory_tracer()

    result = AgentRunner(EndlessToolClient(), config, max_steps=1, debug_tracer=tracer).run(
        "never finish"
    )

    assert result.final_response == "ERROR: Agent exceeded max step limit of 1"
    assert tracer.finish_count == 1
    assert tracer.completed_trace is not None
    trace = tracer.completed_trace
    assert [event.event_type for event in trace.events] == [
        "run_started",
        "llm_request",
        "llm_response",
        "tool_call_parsed",
        "tool_executed",
        "run_error",
    ]
    assert trace.events[-1].payload == {
        "error": "Agent exceeded max step limit of 1",
        "max_steps": 1,
    }


def test_run_eval_cli_fake_client(tmp_path) -> None:
    config_path = tmp_path / "eval.yaml"
    raw_config = yaml.safe_load(open("configs/eval.yaml", encoding="utf-8"))
    raw_config["database"]["path"] = str(tmp_path / "cli.sqlite")
    raw_config["tasks"] = {"paths": [str(Path.cwd() / "tasks/normal/tasks_001_020.yaml")]}
    raw_config["results"] = {"output_path": str(tmp_path / "runs.jsonl")}
    raw_config["debug"]["output_dir"] = str(tmp_path / "debug_traces")
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
            "--judge-client",
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
    assert record["scores"]["task_completion"]["score"] == 1
    assert record["scores"]["tool_selection_accuracy"]["score"] == 1.0
    assert record["scores"]["argument_faithfulness"]["final_score"] is not None
    assert record["scores"]["adversarial_robustness"] is None
    assert record["composite_score"] is not None
    assert len(record["config_hash"]) == 64

    debug_files = list((tmp_path / "debug_traces").glob("*.json"))
    assert len(debug_files) == 1
    debug_trace = json.loads(debug_files[0].read_text(encoding="utf-8"))
    assert debug_trace["run_id"] == record["run_id"]
    assert debug_trace["task_id"] == record["task_id"]
    assert debug_trace["model_name"] == record["model_name"]
    assert debug_trace["user_message"] == record["user_message"]


def test_run_eval_cli_fake_client_skips_debug_trace_when_disabled(tmp_path) -> None:
    config_path = tmp_path / "eval.yaml"
    raw_config = yaml.safe_load(open("configs/eval.yaml", encoding="utf-8"))
    raw_config["database"]["path"] = str(tmp_path / "cli.sqlite")
    raw_config["tasks"] = {"paths": [str(Path.cwd() / "tasks/normal/tasks_001_020.yaml")]}
    raw_config["results"] = {"output_path": str(tmp_path / "runs.jsonl")}
    raw_config["debug"]["enabled"] = False
    raw_config["debug"]["storage"] = "disabled"
    raw_config["debug"]["output_dir"] = str(tmp_path / "debug_traces")
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
    assert (tmp_path / "runs.jsonl").exists()
    assert not (tmp_path / "debug_traces").exists()


def test_run_eval_cli_fake_client_embeds_debug_trace_when_configured(tmp_path) -> None:
    config_path = tmp_path / "eval.yaml"
    raw_config = yaml.safe_load(open("configs/eval.yaml", encoding="utf-8"))
    raw_config["database"]["path"] = str(tmp_path / "cli.sqlite")
    raw_config["tasks"] = {"paths": [str(Path.cwd() / "tasks/normal/tasks_001_020.yaml")]}
    raw_config["results"] = {"output_path": str(tmp_path / "runs.jsonl")}
    raw_config["debug"]["storage"] = "embedded"
    raw_config["debug"]["output_dir"] = str(tmp_path / "debug_traces")
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
    assert not (tmp_path / "debug_traces").exists()

    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["debug_trace"]["run_id"] == record["run_id"]
    assert record["debug_trace"]["task_id"] == record["task_id"]
    assert record["debug_trace"]["model_name"] == record["model_name"]
    assert record["debug_trace"]["user_message"] == record["user_message"]
    assert [event["event_type"] for event in record["debug_trace"]["events"]]
