from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .integrations import diagnose_tool


class DiagnosticToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    task: str | None = None
    inputs: dict[str, Any]
    policy_ref: str | None = None
    model_refs: dict[str, str] = Field(default_factory=dict)
    run_context: dict[str, Any] | None = None


app = FastAPI(
    title="Time-Series Diagnostic Agent API",
    version="1.1.0",
    description="Host-neutral HTTP boundary for evidence-backed industrial diagnostics.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "time-series-diagnostic-agent", "version": "1.1.0"}


@app.post("/v1/diagnose")
def diagnose(request: DiagnosticToolRequest) -> dict[str, Any]:
    try:
        return diagnose_tool(request.model_dump(mode="python"))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
