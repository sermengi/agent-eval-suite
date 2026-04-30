from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any

from src.inference.client import ModelMessage, ToolCall


class OpenAIModelClient:
    """OpenAI function-calling client for manual Week 1 runs."""

    def __init__(self, model: str) -> None:
        """Create an OpenAI client using OPENAI_API_KEY from the environment."""

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIModelClient")
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def next_message(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelMessage:
        """Return the next normalized model response."""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            raw_arguments = call.function.arguments or "{}"
            parse_error = None
            try:
                arguments = json.loads(raw_arguments)
            except JSONDecodeError as exc:
                arguments = {"_raw_arguments": raw_arguments}
                parse_error = f"Invalid JSON tool arguments: {exc.msg}"
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return ModelMessage(
            content=message.content,
            tool_calls=tool_calls or None,
            raw_response=response.model_dump(mode="json"),
        )
