from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.integrations.models import DiagnosticInvocation
from app.tools.file_tools import SafeWorkspace


class IntegrationRoutePlan(BaseModel):
    use_graph: bool = False
    diagnostic_domain: str | None = None
    diagnostic: DiagnosticInvocation | None = None
    missing_diagnostic_input: bool = False
    services: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "The request can use the standard SovereignAI workflow."

    @property
    def handles_request(self) -> bool:
        return self.use_graph or self.diagnostic_domain is not None


class IntegrationRoutePlanner:
    """Select cross-system services deterministically without sending data to an LLM."""

    GRAPH_PHRASES = (
        "knowledge graph", "graph rag", "graph-rag", "cross-document",
        "internal knowledge", "internal documents", "company documents",
        "document evidence", "source evidence", "cite the sources",
        "according to our", "according to the manual", "according to the sop",
        "find the procedure", "find the policy", "find the standard",
        "relationship between", "what do our documents",
    )
    DIAGNOSTIC_PHRASES = (
        "diagnose", "diagnostic", "fault detection", "fault diagnosis",
        "anomaly", "abnormal", "root cause", "remaining useful life", "rul",
        "condition monitoring", "predict failure",
    )
    DOMAIN_TERMS = {
        "bearing": ("bearing", "vibration", "bpfo", "bpfi", "ball pass"),
        "process": ("process", "pressure", "flow", "level", "process sensor"),
        "wind_scada": ("wind turbine", "wind scada", "scada"),
        "battery": ("battery", "cell voltage", "cell temperature", "state of health", "soh"),
        "turbofan": ("turbofan", "jet engine", "engine cycle"),
        "transformer": ("transformer", "winding", "partial discharge"),
    }

    def plan(self, request: str, attachments: list[str], workspace_root: Path) -> IntegrationRoutePlan:
        normalized = " ".join(request.lower().split())
        diagnostic = self._load_diagnostic_payload(attachments, workspace_root)
        domain = diagnostic.domain if diagnostic else self._detect_domain(normalized)
        diagnostic_intent = diagnostic is not None or (
            domain is not None and any(phrase in normalized for phrase in self.DIAGNOSTIC_PHRASES)
        )
        use_graph = any(phrase in normalized for phrase in self.GRAPH_PHRASES)

        services: list[str] = []
        if use_graph:
            services.append("graph-rag")
        if diagnostic_intent:
            services.append("time-series-diagnostic-agent")
        if services:
            services.append("controlplane")

        if diagnostic_intent and diagnostic is None:
            reason = f"The request needs the {domain} diagnostic service, but no validated diagnostic JSON input was attached."
            confidence = 0.9
        elif diagnostic and use_graph:
            reason = "The request combines time-series diagnosis with internal document evidence."
            confidence = 0.98
        elif diagnostic:
            reason = f"A validated {diagnostic.domain} diagnostic payload was attached."
            confidence = 0.99
        elif use_graph:
            reason = "The question explicitly asks for internal, cited, or cross-document evidence."
            confidence = 0.92
        else:
            reason = "No cross-system evidence or diagnostic intent was detected."
            confidence = 0.86

        return IntegrationRoutePlan(
            use_graph=use_graph,
            diagnostic_domain=domain if diagnostic_intent else None,
            diagnostic=diagnostic,
            missing_diagnostic_input=diagnostic_intent and diagnostic is None,
            services=services,
            confidence=confidence,
            reason=reason,
        )

    def _load_diagnostic_payload(
        self, attachments: list[str], workspace_root: Path,
    ) -> DiagnosticInvocation | None:
        workspace = SafeWorkspace(workspace_root)
        for attachment in attachments:
            if Path(attachment).suffix.lower() != ".json":
                continue
            path = workspace.resolve(attachment, must_exist=True)
            try:
                decoded: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(decoded, dict) and isinstance(decoded.get("diagnostic"), dict):
                decoded = decoded["diagnostic"]
            try:
                return DiagnosticInvocation.model_validate(decoded)
            except (ValidationError, TypeError):
                continue
        return None

    def _detect_domain(self, normalized: str) -> str | None:
        matches = [
            domain for domain, terms in self.DOMAIN_TERMS.items()
            if any(term in normalized for term in terms)
        ]
        return matches[0] if len(matches) == 1 else None
