# agent-eval-suite

`agent-eval-suite` is a reproducible evaluation framework for a simple function-calling
data analyst agent. The project measures task completion, tool selection accuracy,
argument faithfulness, and adversarial robustness separately so tool-use failures are
not hidden behind correct-looking final answers.

## Current Scope

The current implementation covers the Week 1, Week 2, and Week 3 scoring slices:

- deterministic local SQLite database generation
- the three fixed tools: `sql_query`, `python_exec`, and `summarize`
- a simple function-calling agent runner with fake and OpenAI clients
- YAML task loading from `tasks/`
- one configured task run end to end
- structured JSONL result logging with the final evaluation record schema
- nested scoring for task completion, tool selection accuracy, argument faithfulness,
  adversarial robustness, and composite score

MLflow, Docker, Mistral runs, and the full 50-task set are later milestones.

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
uv run python scripts/run_eval.py --config configs/eval.yaml --client fake --judge-client fake
```

Use `--client openai` to evaluate the OpenAI-backed agent and `--judge-client openai`
to score with the configured OpenAI judge. Either OpenAI option requires
`OPENAI_API_KEY` to be set. The two flags are independent so offline development can
use the deterministic fake agent, deterministic fake judge, or a mix of fake and OpenAI
components.

The CLI loads the first configured YAML task, runs the agent, and appends one structured
record to `results/runs.jsonl`. Generated result files are ignored by git.

## Debug UI

Generate local run artifacts, then start the read-only debug UI:

```bash
python scripts/run_eval.py --config configs/eval.yaml --client fake
python scripts/run_debug_ui.py --config configs/eval.yaml
```

The UI lets you inspect existing run records and debug traces. It does not start
evaluations or modify result files.

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
Week 3 scoring is logged as nested per-dimension objects rather than flat placeholder
fields.

Each record includes:

- run and timestamp metadata
- model and task metadata
- full tool call trace
- final agent response
- `scores.task_completion` with binary judge score and rationale
- `scores.tool_selection_accuracy` with rule-based sequence score
- `scores.argument_faithfulness` with schema, intent, and final scores
- `scores.adversarial_robustness`, which is `null` for normal and ambiguous tasks
- weighted `composite_score`
- judge model and prompt versions
- SHA-256 config hash

The default fake judge is deterministic and offline: it passes non-error final answers
and checks task validation hints against serialized tool arguments. The OpenAI judge
uses the configured judge model with versioned prompts from `prompts/` for task
completion and argument-intent scoring.

## Test And Check

```bash
uv run pytest
uv run black --check .
uv run isort --check-only .
uv run flake8 .
```
