# agent-eval-suite

`agent-eval-suite` is a reproducible evaluation framework for a simple function-calling
data analyst agent. The project measures task completion, tool selection accuracy,
argument faithfulness, and adversarial robustness separately so tool-use failures are
not hidden behind correct-looking final answers.

## Week 1 Scope

This first slice creates the project scaffold, generates a deterministic local SQLite
database, implements the three fixed tools, and runs one fake-client end-to-end agent
task. Full YAML task loading, scoring, structured JSON logging, MLflow, Docker, Mistral
runs, and the full 50-task set are later milestones.

## Setup

```bash
uv sync --dev
```

## Generate The Database

```bash
uv run python scripts/generate_db.py --config configs/eval.yaml
```

## Run The Week 1 Demo

```bash
uv run python scripts/run_eval.py --config configs/eval.yaml --client fake
```

Use `--client openai` only when `OPENAI_API_KEY` is set.

## Test And Check

```bash
uv run pytest
uv run black --check .
uv run isort --check-only .
uv run flake8 .
```

