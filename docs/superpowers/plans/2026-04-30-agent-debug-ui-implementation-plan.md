# Agent Debug UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local FastAPI/Jinja debugging UI for completed agent runs, backed by structured debug traces emitted by the agent loop.

**Architecture:** Instrument `AgentRunner` with an optional tracer interface that records structured debug events without coupling runner code to the UI. Store debug traces separately by default as `results/debug_traces/<run_id>.json`, then build a read-only timeline-first UI that loads evaluation JSONL records plus linked traces.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, Jinja2, HTMX via CDN or minimal progressive-enhancement attributes, pytest.

---

## Key Changes

- Add config support for:
  - `debug.enabled: bool`
  - `debug.storage: "separate" | "embedded" | "disabled"`
  - `debug.output_dir: Path`
  - `debug.include_raw_payloads: bool`
- Add dependencies:
  - runtime: `fastapi`, `uvicorn`, `jinja2`
  - dev/test if needed: `httpx` for FastAPI route tests
- Add `src/debugging/`:
  - Pydantic schemas for `DebugTrace`, `DebugEvent`, diagnostics, and run summaries
  - tracer interface plus no-op and file-backed tracer implementations
  - loader that joins evaluation records with debug traces
  - diagnostics module for rule-based warnings
- Extend inference clients:
  - keep `ModelClient.next_message(...) -> ModelMessage`
  - add optional raw/debug metadata to `ModelMessage`, preferably `raw_response: dict[str, Any] | None = None`
  - `OpenAIModelClient` should include sanitized `response.model_dump(mode="json")` when raw payload capture is enabled by the tracer path
  - `FakeModelClient` should provide deterministic raw-like payloads for tests
- Extend `AgentRunner`:
  - accept optional `run_id`, task metadata, and `DebugTracer`
  - emit `run_started`, `llm_request`, `llm_response`, `tool_call_parsed`, `tool_executed`, `final_response`, and `run_error`
  - preserve current `AgentRunResult` behavior and existing tool call trace behavior
- Update `scripts/run_eval.py`:
  - create `run_id` before invoking `AgentRunner`
  - initialize tracer from config
  - pass task/model metadata into the runner
  - write evaluation record as before
  - write debug trace only when config enables it
- Add `src/ui/`:
  - FastAPI app factory
  - routes for run list, run detail, and JSON endpoints
  - Jinja templates for timeline-first run detail
  - static CSS for compact debugging UI
  - CLI entrypoint script, e.g. `scripts/run_debug_ui.py --config configs/eval.yaml`

## Implementation Tasks

### Task 1: Config And Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/eval.yaml`
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add tests that assert default debug values load from `configs/eval.yaml`, relative `debug.output_dir` resolves from a custom config directory, and invalid storage values fail validation.

- [ ] **Step 2: Run config tests to verify failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because `EvalConfig` has no `debug` field yet.

- [ ] **Step 3: Add debug dependencies and config models**

Add `fastapi`, `uvicorn`, and `jinja2` to runtime dependencies. Add `httpx` to dev dependencies if route tests require it. Add `DebugConfig` and `DebugStorage` to `src/config.py`, and resolve `debug.output_dir` the same way `results.output_path` is resolved.

- [ ] **Step 4: Add default debug config**

Add this to `configs/eval.yaml`:

```yaml
debug:
  enabled: true
  storage: separate
  output_dir: "results/debug_traces"
  include_raw_payloads: true
```

- [ ] **Step 5: Run config tests**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml configs/eval.yaml src/config.py tests/test_config.py
git commit -m "feat: add debug trace configuration"
```

### Task 2: Debug Trace Schemas

**Files:**
- Create: `src/debugging/__init__.py`
- Create: `src/debugging/schemas.py`
- Create: `tests/test_debugging_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Cover valid trace serialization/deserialization, unknown event type rejection, and missing run metadata rejection.

- [ ] **Step 2: Run schema tests to verify failure**

Run: `pytest tests/test_debugging_schemas.py -v`

Expected: FAIL because `src.debugging.schemas` does not exist.

- [ ] **Step 3: Implement schemas**

Define event type literals for `run_started`, `llm_request`, `llm_response`, `tool_call_parsed`, `tool_executed`, `final_response`, and `run_error`. Define `DebugEvent` with `event_id`, `event_type`, `step`, `timestamp`, `title`, `summary`, and `payload`. Define `DebugTrace` with `run_id`, `task_id`, `model_name`, `user_message`, `started_at`, `completed_at`, and `events`.

- [ ] **Step 4: Run schema tests**

Run: `pytest tests/test_debugging_schemas.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/debugging/__init__.py src/debugging/schemas.py tests/test_debugging_schemas.py
git commit -m "feat: add debug trace schemas"
```

### Task 3: Tracer Implementations

**Files:**
- Create: `src/debugging/tracer.py`
- Create: `tests/test_debugging_tracer.py`

- [ ] **Step 1: Write failing tracer tests**

Cover no-op behavior, ordered in-memory events, and file tracer output at `<output_dir>/<run_id>.json`.

- [ ] **Step 2: Run tracer tests to verify failure**

Run: `pytest tests/test_debugging_tracer.py -v`

Expected: FAIL because tracer implementations do not exist.

- [ ] **Step 3: Implement tracers**

Add a `DebugTracer` protocol with `record(event)` and `finish(completed_at)`. Add `NoOpDebugTracer`, `InMemoryDebugTracer`, and `FileDebugTracer`. Add `build_debug_tracer(config, run_id, task_id, model_name, user_message)` that returns the correct tracer for `enabled` and `storage`.

- [ ] **Step 4: Run tracer tests**

Run: `pytest tests/test_debugging_tracer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/debugging/tracer.py tests/test_debugging_tracer.py
git commit -m "feat: add debug trace writers"
```

### Task 4: Model Response Raw Metadata

**Files:**
- Modify: `src/inference/client.py`
- Modify: `src/inference/fake_client.py`
- Modify: `src/inference/openai_client.py`
- Modify or create: `tests/test_fake_client.py`

- [ ] **Step 1: Write failing client tests**

Assert fake responses include deterministic `raw_response` metadata and existing tool-call behavior is unchanged.

- [ ] **Step 2: Run client tests to verify failure**

Run: `pytest tests/test_fake_client.py -v`

Expected: FAIL because `ModelMessage.raw_response` does not exist.

- [ ] **Step 3: Add raw response metadata**

Add `raw_response: dict[str, Any] | None = None` to `ModelMessage`. Update `FakeModelClient` to include deterministic raw-like payloads. Update `OpenAIModelClient` to attach sanitized `response.model_dump(mode="json")`.

- [ ] **Step 4: Run client tests**

Run: `pytest tests/test_fake_client.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference/client.py src/inference/fake_client.py src/inference/openai_client.py tests/test_fake_client.py
git commit -m "feat: expose raw model response metadata"
```

### Task 5: Agent Runner Instrumentation

**Files:**
- Modify: `src/agent/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing runner trace tests**

Cover normal fake run event order, parse-error flow, final-answer-without-tool flow, and max-step termination event.

- [ ] **Step 2: Run runner tests to verify failure**

Run: `pytest tests/test_runner.py -v`

Expected: FAIL because `AgentRunner` does not emit debug events.

- [ ] **Step 3: Instrument runner**

Add optional `debug_tracer` to `AgentRunner.__init__`. Emit the agreed event types around model calls, parsed tool calls, tool execution, final response, and max-step termination. Keep `AgentRunResult` unchanged.

- [ ] **Step 4: Run runner tests**

Run: `pytest tests/test_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/runner.py tests/test_runner.py
git commit -m "feat: emit agent debug trace events"
```

### Task 6: Run Eval Integration

**Files:**
- Modify: `scripts/run_eval.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing CLI integration tests**

Extend the fake CLI test to assert one debug file is written with the same `run_id` as the JSONL evaluation record. Add a disabled-debug config case that writes no debug file.

- [ ] **Step 2: Run CLI tests to verify failure**

Run: `pytest tests/test_runner.py::test_run_eval_cli_fake_client -v`

Expected: FAIL because no debug trace file is created.

- [ ] **Step 3: Wire tracer into CLI**

Generate `run_id` before `AgentRunner.run(...)`. Build the debug tracer from config and pass it to `AgentRunner`. Use the same `run_id` in the evaluation record and trace artifact. Preserve existing stdout behavior.

- [ ] **Step 4: Run CLI tests**

Run: `pytest tests/test_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py tests/test_runner.py
git commit -m "feat: write debug traces during eval runs"
```

### Task 7: Diagnostics

**Files:**
- Create: `src/debugging/diagnostics.py`
- Create: `tests/test_debugging_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics tests**

Use synthetic traces to cover expected tool mismatch, skipped expected tool, repeated tool, parse error, blocked tool, unsafe SQL/Python argument, prompt-injection-like tool output, final response after tool error, and missing trace.

- [ ] **Step 2: Run diagnostics tests to verify failure**

Run: `pytest tests/test_debugging_diagnostics.py -v`

Expected: FAIL because diagnostics do not exist.

- [ ] **Step 3: Implement diagnostics**

Add diagnostic models or dictionaries with `severity`, `title`, `reason`, and optional `event_id`. Implement rule-based checks without calling an LLM.

- [ ] **Step 4: Run diagnostics tests**

Run: `pytest tests/test_debugging_diagnostics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/debugging/diagnostics.py tests/test_debugging_diagnostics.py
git commit -m "feat: add explainable debug diagnostics"
```

### Task 8: Trace Loader

**Files:**
- Create: `src/debugging/loader.py`
- Create: `tests/test_debugging_loader.py`

- [ ] **Step 1: Write failing loader tests**

Cover loading a run with trace, loading a run without trace, malformed trace handling, and filtering by model/task/category/severity.

- [ ] **Step 2: Run loader tests to verify failure**

Run: `pytest tests/test_debugging_loader.py -v`

Expected: FAIL because the loader does not exist.

- [ ] **Step 3: Implement loader**

Load evaluation records from the configured JSONL path, load linked debug traces by `run_id`, attach diagnostics, and return run summaries plus full run detail objects.

- [ ] **Step 4: Run loader tests**

Run: `pytest tests/test_debugging_loader.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/debugging/loader.py tests/test_debugging_loader.py
git commit -m "feat: load debug runs from result artifacts"
```

### Task 9: FastAPI UI

**Files:**
- Create: `src/ui/__init__.py`
- Create: `src/ui/app.py`
- Create: `src/ui/templates/base.html`
- Create: `src/ui/templates/runs.html`
- Create: `src/ui/templates/run_detail.html`
- Create: `src/ui/templates/_event_card.html`
- Create: `src/ui/static/debug.css`
- Create: `tests/test_debug_ui.py`

- [ ] **Step 1: Write failing UI route tests**

Cover `/`, `/runs`, `/runs/{run_id}`, `/api/runs`, and `/api/runs/{run_id}` with fixture artifacts.

- [ ] **Step 2: Run UI tests to verify failure**

Run: `pytest tests/test_debug_ui.py -v`

Expected: FAIL because the UI app does not exist.

- [ ] **Step 3: Implement app and routes**

Create a FastAPI app factory that accepts config path or loaded config. Add read-only routes for run list, run detail, and JSON APIs.

- [ ] **Step 4: Implement templates and CSS**

Render a compact run table, severity badges, summary strip, sequence map, timeline cards, expandable raw JSON, and graceful missing-data states.

- [ ] **Step 5: Run UI tests**

Run: `pytest tests/test_debug_ui.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ui tests/test_debug_ui.py
git commit -m "feat: add local debug trace UI"
```

### Task 10: Developer Entry Point And Docs

**Files:**
- Create: `scripts/run_debug_ui.py`
- Modify: `README.md`
- Create or modify: `tests/test_debug_ui.py`

- [ ] **Step 1: Write failing entrypoint smoke test**

Assert the app factory imports cleanly and the debug UI script exposes expected CLI arguments without starting a server during tests.

- [ ] **Step 2: Run smoke test to verify failure**

Run: `pytest tests/test_debug_ui.py -v`

Expected: FAIL because the CLI script does not exist.

- [ ] **Step 3: Add debug UI script**

Add `scripts/run_debug_ui.py --config configs/eval.yaml --host 127.0.0.1 --port 8000` using Uvicorn to serve the app.

- [ ] **Step 4: Update README**

Add a short Debug UI section explaining:

```bash
python scripts/run_eval.py --config configs/eval.yaml --client fake
python scripts/run_debug_ui.py --config configs/eval.yaml
```

State that the UI is read-only and does not start evaluations.

- [ ] **Step 5: Run smoke tests**

Run: `pytest tests/test_debug_ui.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_debug_ui.py README.md tests/test_debug_ui.py
git commit -m "docs: document local debug UI"
```

## Test Plan

- Run focused tests after each task, then full suite:
  - `pytest tests/test_config.py -v`
  - `pytest tests/test_runner.py -v`
  - `pytest tests/test_debugging_schemas.py tests/test_debugging_tracer.py -v`
  - `pytest tests/test_debugging_diagnostics.py tests/test_debugging_loader.py -v`
  - `pytest tests/test_debug_ui.py -v`
  - `pytest -v`
- Manual verification:
  - `python scripts/run_eval.py --config configs/eval.yaml --client fake`
  - confirm one JSONL record in configured results path
  - confirm one debug trace file in configured debug output dir
  - `python scripts/run_debug_ui.py --config configs/eval.yaml`
  - open `http://127.0.0.1:8000`
  - verify timeline shows LLM request, response, parsed tool call, tool execution, and final response
  - verify raw JSON expanders render or show `raw payload not captured`

## Assumptions And Defaults

- Default debug storage is `separate`.
- First implementation writes one debug JSON file per run.
- UI is read-only and never starts or edits evaluation runs.
- Raw payload capture is best-effort and sanitized to exclude secrets, environment variables, and headers.
- Server-rendered FastAPI/Jinja is the first frontend; React/Vite remains a future replacement path.
