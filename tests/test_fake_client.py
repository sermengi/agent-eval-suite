from src.inference.fake_client import FakeModelClient


def test_fake_client_emits_tool_call_then_final_answer() -> None:
    client = FakeModelClient()

    first = client.next_message([{"role": "user", "content": "Revenue by category?"}], [])
    assert first.tool_calls
    assert first.tool_calls[0].name == "sql_query"

    second = client.next_message(
        [
            {"role": "user", "content": "Revenue by category?"},
            {"role": "tool", "name": "sql_query", "content": "category | total_revenue\nFood | 10"},
        ],
        [],
    )
    assert second.content is not None
    assert "Food" in second.content


def test_fake_client_includes_deterministic_raw_response_metadata() -> None:
    client = FakeModelClient()

    first = client.next_message([{"role": "user", "content": "Revenue by category?"}], [])
    assert first.raw_response == {
        "id": "fake-response-1",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fake-call-1",
                            "type": "function",
                            "function": {
                                "name": "sql_query",
                                "arguments": {
                                    "query": (
                                        "SELECT category, ROUND(SUM(revenue), 2) AS total_revenue "
                                        "FROM sales GROUP BY category ORDER BY total_revenue DESC"
                                    )
                                },
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    assert first.tool_calls
    assert first.tool_calls[0].name == "sql_query"

    second = client.next_message(
        [
            {"role": "user", "content": "Revenue by category?"},
            {"role": "tool", "name": "sql_query", "content": "category | total_revenue\nFood | 10"},
        ],
        [],
    )
    assert second.raw_response == {
        "id": "fake-response-2",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Total revenue by category, based on the database query:\n"
                    "category | total_revenue\nFood | 10",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }
    assert second.content is not None
    assert "Food" in second.content
