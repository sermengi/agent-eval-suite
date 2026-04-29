from __future__ import annotations

from typing import Any

from src.inference.client import ModelMessage, ToolCall


class FakeModelClient:
    """Deterministic offline model client for Week 1 tests and demo runs."""

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return a SQL tool call first, then a final answer from the tool result."""

        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            return ModelMessage(
                tool_calls=[
                    ToolCall(
                        id="fake-call-1",
                        name="sql_query",
                        arguments={
                            "query": (
                                "SELECT category, ROUND(SUM(revenue), 2) AS total_revenue "
                                "FROM sales GROUP BY category ORDER BY total_revenue DESC"
                            )
                        },
                    )
                ]
            )
        latest_tool_output = str(tool_messages[-1].get("content", ""))
        return ModelMessage(
            content="Total revenue by category, based on the database query:\n" + latest_tool_output
        )
