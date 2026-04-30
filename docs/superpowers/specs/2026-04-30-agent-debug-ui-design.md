# Agent Debug UI Design

Date: 2026-04-30
Project: agent-eval-suite

## Purpose

Build a local debugging interface for understanding how one completed agent run moved
through the agent loop. The UI is a debugging microscope, not a workflow launcher and
not an evaluation dashboard.

The main question it should answer is:

> What exactly did the agent send to the LLM, what did the LLM return, how did we parse
> it, and where did the workflow go wrong?

## Goals

- Inspect completed agent runs from disk.
- Show the prompt and model interaction flow behind one run.
- Preserve full raw LLM request and response details when configured.
- Highlight likely workflow problems with explainable diagnostics.
- Keep evaluation execution separate from UI inspection.
- Design the debug trace format so live streaming can be added later.

## Non-Goals

- Starting or managing evaluation runs from the UI.
- Editing task YAML, model config, or credentials from the UI.
- Replacing the CLI evaluation workflow.
- Building a multi-run analytics dashboard.
- Implementing live run observation in the first version.

## Recommended Approach

Use a two-part design:

1. Add structured debug instrumentation around the existing agent loop.
2. Build a FastAPI + Jinja + HTMX-style local UI that reads completed evaluation records
   and linked debug traces from disk.

The agent runner should not depend on FastAPI or UI code. It should emit structured
debug events through a small tracer interface. This keeps the core agent testable and
allows a later live tracer implementation without changing the event model.

## Data Model

Each completed run can have two linked artifacts.

### Evaluation Record

The existing JSONL evaluation record remains focused on reproducible evaluation data:

- `run_id`
- task metadata
- model name
- user message
- tool call trace
- final response
- scores
- judge metadata
- config hash

This record should stay compact and publishable.

### Debug Trace

The debug trace stores heavier local debugging details:

- `run_id`
- `task_id`
- `model_name`
- timestamp metadata
- original user message
- ordered debug events
- request messages sent to the model
- tool schemas sent to the model
- raw model response, when available
- normalized parsed `ModelMessage`
- parse errors
- tool execution results
- final response event

Debug traces are structured data, not console output.

### Event Types

Use a small set of event types so the UI can render a stable timeline:

- `run_started`
- `llm_request`
- `llm_response`
- `tool_call_parsed`
- `tool_executed`
- `final_response`
- `run_error`

Raw provider payload capture should be best-effort. The trace should include raw request
and response fields when they are available from the model client, but it must not capture
API keys, environment variables, or transport-level headers.

## Configuration

Add a debug section to `configs/eval.yaml`:

```yaml
debug:
  enabled: true
  storage: separate
  output_dir: results/debug_traces
  include_raw_payloads: true
```

Supported storage modes:

- `separate`: write debug traces as separate artifacts linked by `run_id`; default.
- `embedded`: include debug data in evaluation records for small local runs.
- `disabled`: do not capture debug traces.

The first implementation should default to `separate` to keep official result logs
clean while preserving full local debugging detail.

For `separate` mode, write one JSON debug trace file per run:

```text
results/debug_traces/<run_id>.json
```

The UI should read the evaluation output path and debug trace directory from config.
It should not hardcode `results/` paths outside the default config values.

## UI Architecture

Use FastAPI for local routes and data loading. Use Jinja templates with light HTMX-style
interactions for expandable panels and filtering. Keep backend JSON endpoints clean so
a React/Vite frontend could replace the templates later if needed.

Suggested module layout:

```text
src/debugging/
  schemas.py
  tracer.py
  diagnostics.py
  loader.py

src/ui/
  app.py
  routes.py
  templates/
  static/
```

Exact names can be adjusted to fit existing project conventions during planning.

## Screens

### Run List

Show completed runs with filters:

- model
- task id
- task category
- adversarial type
- score status
- diagnostic severity
- debug trace present or missing

The run list should be compact and work as an entry point into one-run inspection.

### Run Detail

Use a timeline-first layout.

Top-level sections:

- summary strip with task id, model, category, final response status, and scores when
  available
- compact sequence map showing User -> LLM -> Tool -> LLM -> Final
- diagnostics panel with warnings linked to timeline events
- main timeline with one event card per agent action
- expandable raw JSON sections for request, response, parsed message, and tool result

The timeline is the primary debugging surface. The sequence map is supporting context,
not the main navigation model.

### Event Detail

Each timeline event card should show curated fields first and raw JSON second.

Examples:

- LLM request event:
  - step
  - message count
  - tool schemas sent
  - raw request payload, if captured
- LLM response event:
  - raw response, if captured
  - parsed content
  - parsed tool calls
  - parse errors
- Tool execution event:
  - tool name
  - arguments
  - tool return
  - blocked status
  - block reason
- Final response event:
  - final response text
  - whether max step limit was hit

## Diagnostics

Diagnostics should be rule-based and explainable in the first version. Each diagnostic
should include:

- severity
- short title
- reason
- linked event id or step
- optional suggested area to inspect

Initial diagnostics:

- expected tool sequence mismatch
- no tool call when one was expected
- extra or repeated tool call
- tool call parse error
- tool execution blocked by safety layer
- unsafe-looking SQL or Python arguments
- prompt-injection-looking text in tool output
- final response after a tool error without acknowledging the error
- missing debug trace for a logged evaluation record
- malformed debug trace artifact

Diagnostics should help locate suspicious events without pretending to be a definitive
judge of correctness.

## Error Handling

- If a run has no debug trace, show the run and add a missing-trace diagnostic.
- If a trace file is malformed, show the load error for that run without breaking the
  entire run list.
- If raw payload capture is disabled, show curated fields and a clear "raw payload not
  captured" message.
- If scores are unavailable, keep the trace debugger functional.
- Tool violations and parsing problems should be rendered as events, not hidden as
  backend exceptions.

## Testing Strategy

- Unit tests for debug trace schemas and serialization.
- Agent runner tests that verify emitted events for:
  - normal tool call flow
  - parse error flow
  - blocked tool flow
  - final response without tool calls
  - max-step termination
- Diagnostics tests with synthetic traces.
- Loader tests for present, missing, and malformed debug traces.
- FastAPI route/template smoke tests for run list and run detail pages.
- One fixture run that renders a useful timeline in development.

## Future Extensions

- Live streaming by adding a streaming tracer implementation.
- Side-by-side run comparison.
- Richer provider metadata such as token usage, latency, and finish reasons.
- Frontend replacement with React/Vite if the UI outgrows server-rendered templates.
- Deeper diagnostics for argument faithfulness and prompt injection propagation.

## Implementation Planning Notes

- Exact Pydantic field names can be chosen during implementation, but they should map
  directly to the event types and data described above.
- The initial trace writer should write one JSON file per run in `separate` mode.
- The UI should use configurable result and debug paths from `configs/eval.yaml`.
- Raw provider response capture should be best-effort and sanitized to exclude secrets
  or environment details.
