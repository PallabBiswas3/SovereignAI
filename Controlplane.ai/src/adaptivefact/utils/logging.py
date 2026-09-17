"""
Structured per-request logging, per §57 of the master project prompt.

Every request should log:
    request ID, query, response, risk score, chosen route, claim count,
    claims checked, retrieval count, LLM call count, tool call count,
    verification result, latency breakdown, token usage, cost, errors

This is written as JSONL (one JSON object per line) so it's trivially
appendable and greppable/loadable with pandas.read_json(..., lines=True)
for later analysis, without needing a database for early experiments.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptivefact.data.schema import ResponseRecord


class RequestLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: dict[str, Any]) -> None:
        entry = {"logged_at_utc": datetime.now(timezone.utc).isoformat(), **entry}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_from_record(self, record: ResponseRecord, errors: list[str] | None = None) -> None:
        """Convenience method: extracts the §57 fields from a fully
        populated ResponseRecord (i.e. one that has been through risk
        estimation and verification) and logs them."""
        risk = record.risk_metadata
        verification = record.verification_metadata
        timing = record.timing_metadata

        self.log({
            "request_id": record.id,
            "query": record.query,
            "response": record.generated_response,
            "risk_score": risk.calibrated_risk_score if risk else None,
            "chosen_route": verification.route.value if verification else None,
            "claim_count": len(record.atomic_claims),
            "claims_checked": verification.claims_checked if verification else 0,
            "retrieval_count": verification.retrieval_call_count if verification else 0,
            "llm_call_count": verification.llm_call_count if verification else 0,
            "tool_call_count": verification.agent_iterations if verification else 0,
            "verification_result": verification.final_decision.value
            if verification and verification.final_decision else None,
            "latency_breakdown": timing.stage_breakdown_ms if timing else {},
            "total_latency_ms": timing.total_latency_ms if timing else None,
            "cost": sum(c.cost or 0.0 for c in record.atomic_claims),
            "errors": errors or [],
        })
