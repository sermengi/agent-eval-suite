from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_db import generate_database  # noqa: E402
from src.agent.runner import AgentRunner  # noqa: E402
from src.config import load_config  # noqa: E402
from src.inference.fake_client import FakeModelClient  # noqa: E402
from src.inference.openai_client import OpenAIModelClient  # noqa: E402

DEMO_TASK = "What was total revenue by category?"


def main() -> None:
    """Run the Week 1 one-task demo."""

    parser = argparse.ArgumentParser(description="Run the Week 1 agent demo.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--client", choices=["fake", "openai"], default="fake")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.database.path.exists():
        generate_database(config)

    client = (
        FakeModelClient()
        if args.client == "fake"
        else OpenAIModelClient(model=config.models.openai)
    )
    result = AgentRunner(client, config).run(DEMO_TASK)
    print(result.final_response)


if __name__ == "__main__":
    main()
