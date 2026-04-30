from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.agent.tool_registry import ToolExecutionResult, execute_tool, get_tool_schemas
from src.config import EvalConfig
from src.debugging.schemas import DebugEvent, DebugEventType, DebugTrace
from src.debugging.tracer import DebugTracer, NoOpDebugTracer
from src.inference.client import ModelClient, ModelMessage, ToolCall


@dataclass(frozen=True)
class AgentRunResult:
    """Final response and full tool trace for one agent run."""

    final_response: str
    tool_call_trace: list[ToolExecutionResult]
    debug_trace: DebugTrace | None = None


class AgentRunner:
    """Simple function-calling agent loop for Week 1."""

    def __init__(
        self,
        model_client: ModelClient,
        config: EvalConfig,
        max_steps: int = 5,
        debug_tracer: DebugTracer | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        task_category: str | None = None,
        adversarial_type: str | None = None,
    ) -> None:
        """Create a runner with a model client, config, and optional debug tracing."""

        self._model_client = model_client
        self._config = config
        self._max_steps = max_steps
        self._debug_tracer = debug_tracer or NoOpDebugTracer()
        self._run_id = run_id
        self._task_id = task_id
        self._task_category = task_category
        self._adversarial_type = adversarial_type

    def run(self, user_message: str) -> AgentRunResult:
        """Run the agent until a final response is produced or max steps is exceeded."""

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        trace: list[ToolExecutionResult] = []
        tools = get_tool_schemas()
        current_step = 0

        self._record_event(
            event_type="run_started",
            step=0,
            title="Run started",
            summary="Agent run started.",
            payload={
                "run_id": self._run_id,
                "task_id": self._task_id,
                "task_category": self._task_category,
                "adversarial_type": self._adversarial_type,
                "user_message": user_message,
                "max_steps": self._max_steps,
            },
        )

        try:
            for step in range(1, self._max_steps + 1):
                current_step = step
                self._record_event(
                    event_type="llm_request",
                    step=step,
                    title="LLM request",
                    summary="Requested the next model message.",
                    payload={
                        "message_count": len(messages),
                        "messages": self._snapshot_messages(messages),
                        "tool_names": [
                            str(tool_schema["function"]["name"]) for tool_schema in tools
                        ],
                    },
                )
                response = self._model_client.next_message(messages, tools)
                self._record_event(
                    event_type="llm_response",
                    step=step,
                    title="LLM response",
                    summary="Received the next model message.",
                    payload=self._llm_response_payload(response),
                )
                if response.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": json.dumps(tool_call.arguments),
                                    },
                                }
                                for tool_call in response.tool_calls
                            ],
                        }
                    )
                    for tool_call in response.tool_calls:
                        self._record_tool_call_parsed(step, tool_call)
                        if tool_call.parse_error:
                            result = ToolExecutionResult(
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments,
                                tool_return=f"ERROR: {tool_call.parse_error}",
                                was_blocked=True,
                                block_reason=tool_call.parse_error,
                            )
                        else:
                            result = execute_tool(
                                tool_call.name,
                                tool_call.arguments,
                                self._config.database.path,
                                self._config,
                            )
                        trace.append(result)
                        self._record_tool_executed(step, result)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result.tool_return,
                            }
                        )
                    continue
                if response.content:
                    self._record_event(
                        event_type="final_response",
                        step=step,
                        title="Final response",
                        summary="Agent produced a final response.",
                        payload={"final_response": response.content},
                    )
                    debug_trace = self._finish_trace()
                    return AgentRunResult(
                        final_response=response.content,
                        tool_call_trace=trace,
                        debug_trace=debug_trace,
                    )

            error = f"Agent exceeded max step limit of {self._max_steps}"
            self._record_event(
                event_type="run_error",
                step=current_step,
                title="Run error",
                summary=error,
                payload={"error": error, "max_steps": self._max_steps},
            )
            debug_trace = self._finish_trace()
            return AgentRunResult(
                final_response=f"ERROR: {error}",
                tool_call_trace=trace,
                debug_trace=debug_trace,
            )
        except Exception as exc:
            self._record_event(
                event_type="run_error",
                step=current_step,
                title="Run error",
                summary=str(exc),
                payload={"error": str(exc), "error_type": exc.__class__.__name__},
            )
            self._finish_trace()
            raise

    def _record_tool_call_parsed(self, step: int, tool_call: ToolCall) -> None:
        self._record_event(
            event_type="tool_call_parsed",
            step=step,
            title="Tool call parsed",
            summary=f"Parsed tool call for {tool_call.name}.",
            payload={
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "parse_error": tool_call.parse_error,
            },
        )

    def _record_tool_executed(self, step: int, result: ToolExecutionResult) -> None:
        self._record_event(
            event_type="tool_executed",
            step=step,
            title="Tool executed",
            summary=f"Executed tool {result.tool_name}.",
            payload={
                "tool_name": result.tool_name,
                "arguments": result.arguments,
                "tool_return": result.tool_return,
                "was_blocked": result.was_blocked,
                "block_reason": result.block_reason,
            },
        )

    def _record_event(
        self,
        *,
        event_type: DebugEventType,
        step: int,
        title: str,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        self._debug_tracer.record(
            DebugEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                step=step,
                timestamp=datetime.now(timezone.utc),
                title=title,
                summary=summary,
                payload=payload,
            )
        )

    def _finish_trace(self) -> DebugTrace | None:
        return self._debug_tracer.finish(datetime.now(timezone.utc))

    def _llm_response_payload(self, response: ModelMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": response.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "parse_error": tool_call.parse_error,
                }
                for tool_call in response.tool_calls or []
            ],
        }
        if self._config.debug.include_raw_payloads and response.raw_response is not None:
            payload["raw_response"] = response.raw_response
        return payload

    @staticmethod
    def _snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return json.loads(json.dumps(messages, default=str))
