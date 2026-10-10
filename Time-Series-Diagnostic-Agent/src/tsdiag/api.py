from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .integrations import diagnose_tool
from .input_contract import catalogue, validate_payload


class DiagnosticToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    contract_version: str
    task: str | None = None
    inputs: dict[str, Any]
    policy_ref: str | None = None
    model_refs: dict[str, str] = Field(default_factory=dict)
    run_context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_input_contract(self):
        validate_payload(self.model_dump(mode="python"))
        return self


app = FastAPI(
    title="Time-Series Diagnostic Agent API",
    version="1.1.0",
    description="Host-neutral HTTP boundary for evidence-backed industrial diagnostics.",
)


@app.get("/v1/input-contracts")
def input_contracts() -> dict[str, Any]:
    return catalogue()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "time-series-diagnostic-agent", "version": "1.1.0"}


@app.post("/v1/diagnose")
def diagnose(request: DiagnosticToolRequest) -> dict[str, Any]:
    try:
        return diagnose_tool(request.model_dump(mode="python"))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
