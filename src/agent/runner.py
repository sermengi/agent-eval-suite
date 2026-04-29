from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.tool_registry import ToolExecutionResult, execute_tool, get_tool_schemas
from src.config import EvalConfig
from src.inference.client import ModelClient


@dataclass(frozen=True)
class AgentRunResult:
    """Final response and full tool trace for one agent run."""

    final_response: str
    tool_call_trace: list[ToolExecutionResult]


class AgentRunner:
    """Simple function-calling agent loop for Week 1."""

    def __init__(self, model_client: ModelClient, config: EvalConfig, max_steps: int = 5) -> None:
        """Create a runner with a model client, config, and step cap."""

        self._model_client = model_client
        self._config = config
        self._max_steps = max_steps

    def run(self, user_message: str) -> AgentRunResult:
        """Run the agent until a final response is produced or max steps is exceeded."""

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        trace: list[ToolExecutionResult] = []
        tools = get_tool_schemas()

        for _ in range(self._max_steps):
            response = self._model_client.next_message(messages, tools)
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
                return AgentRunResult(final_response=response.content, tool_call_trace=trace)

        return AgentRunResult(
            final_response=f"ERROR: Agent exceeded max step limit of {self._max_steps}",
            tool_call_trace=trace,
        )
