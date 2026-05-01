from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.app import create_debug_ui_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the debug UI server."""

    parser = argparse.ArgumentParser(description="Serve the read-only Agent Debug UI.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Serve the read-only FastAPI debug UI with Uvicorn."""

    args = build_parser().parse_args(argv)
    app = create_debug_ui_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
