from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import EvalConfig, load_config
from src.debugging.diagnostics import DiagnosticSeverity
from src.debugging.loader import DebugRunDetail, DebugRunLoader, DebugRunSummary

UI_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"


def create_debug_ui_app(config: str | Path | EvalConfig = "configs/eval.yaml") -> FastAPI:
    """Create the read-only FastAPI app for browsing debug run artifacts."""

    eval_config = config if isinstance(config, EvalConfig) else load_config(config)
    loader = DebugRunLoader(eval_config)
    templates = _build_templates()

    app = FastAPI(title="Agent Debug UI")
    app.state.config = eval_config
    app.state.debug_loader = loader
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Redirect the root path to the run list."""

        return RedirectResponse(url="/runs")

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(
        request: Request,
        model: str | None = None,
        task: str | None = None,
        category: str | None = None,
        severity: DiagnosticSeverity | None = None,
    ) -> HTMLResponse:
        """Render the debug run list."""

        runs = loader.list_runs(model=model, task=task, category=category, severity=severity)
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": runs,
                "filters": {
                    "model": model,
                    "task": task,
                    "category": category,
                    "severity": severity,
                },
                "summary": _summarize_runs(runs),
            },
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail_page(request: Request, run_id: str) -> HTMLResponse:
        """Render one run detail page."""

        detail = _require_run(loader, run_id)
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "run": detail,
                "events": detail.trace.events if detail.trace is not None else [],
                "tool_sequence": _tool_sequence(detail),
            },
        )

    @app.get("/api/runs")
    def api_runs(
        model: str | None = None,
        task: str | None = None,
        category: str | None = None,
        severity: DiagnosticSeverity | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        """Return JSON run summaries."""

        runs = loader.list_runs(model=model, task=task, category=category, severity=severity)
        return [_dump_model(run) for run in runs]

    @app.get("/api/runs/{run_id}")
    def api_run_detail(run_id: str) -> dict[str, Any]:
        """Return JSON detail for one run."""

        return _dump_model(_require_run(loader, run_id))

    return app


def _build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=TEMPLATE_DIR)
    templates.env.filters["timestamp"] = _format_timestamp
    templates.env.filters["json"] = _format_json
    return templates


def _require_run(loader: DebugRunLoader, run_id: str) -> DebugRunDetail:
    detail = loader.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


def _dump_model(model: DebugRunSummary | DebugRunDetail) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _summarize_runs(runs: list[DebugRunSummary]) -> dict[str, int]:
    return {
        "total": len(runs),
        "with_trace": sum(1 for run in runs if run.trace_available),
        "diagnostics": sum(run.diagnostic_count for run in runs),
        "errors": sum(1 for run in runs if run.max_severity == "error"),
    }


def _tool_sequence(detail: DebugRunDetail) -> list[str]:
    if detail.trace is None:
        return []
    sequence: list[str] = []
    for event in detail.trace.events:
        tool_name = event.payload.get("tool_name")
        if event.event_type == "tool_executed" and isinstance(tool_name, str):
            sequence.append(tool_name)
    return sequence
