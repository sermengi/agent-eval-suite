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
