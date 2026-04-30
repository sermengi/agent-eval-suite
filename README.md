# agent-eval-suite

`agent-eval-suite` is a reproducible evaluation framework for a simple function-calling
data analyst agent. The project measures task completion, tool selection accuracy,
argument faithfulness, and adversarial robustness separately so tool-use failures are
not hidden behind correct-looking final answers.

## Current Scope

The current implementation covers the Week 1 and Week 2 slices:

- deterministic local SQLite database generation
- the three fixed tools: `sql_query`, `python_exec`, and `summarize`
- a simple function-calling agent runner with fake and OpenAI clients
- YAML task loading from `tasks/`
- one configured task run end to end
- structured JSONL result logging with the final evaluation record schema

Scoring, MLflow, Docker, Mistral runs, and the full 50-task set are later milestones.

## Setup

```bash
uv sync --dev
```

## Generate The Database

```bash
uv run python scripts/generate_db.py --config configs/eval.yaml
```

## Run One Configured Task

```bash
uv run python scripts/run_eval.py --config configs/eval.yaml --client fake
```

Use `--client openai` only when `OPENAI_API_KEY` is set.

The CLI loads the first configured YAML task, runs the agent, and appends one structured
record to `results/runs.jsonl`. Generated result files are ignored by git.

## Tasks

Tasks are defined in YAML under `tasks/` and are configured through
`configs/eval.yaml`.

The starter task is `task_001` in `tasks/normal/tasks_001_020.yaml`. Each task includes:

- `id`
- `category`
- `description`
- `reference_answer`
- `expected_tool_sequence`
- `adversarial_type`
- `notes`

The loader validates task categories, adversarial type rules, and expected tool names
against the fixed tool set.

## Result Logs

Run records are written as JSONL using the final schema shape from `CLAUDE.md`.
Week 3 scoring fields are present but currently logged as `null` placeholders.

Each record includes:

- run and timestamp metadata
- model and task metadata
- full tool call trace
- final agent response
- nullable dimension scores
- judge model and prompt versions
- SHA-256 config hash

## Test And Check

```bash
uv run pytest
uv run black --check .
uv run isort --check-only .
uv run flake8 .
```
