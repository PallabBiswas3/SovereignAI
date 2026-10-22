"""Format a released industrial assessment without another inference call."""
from __future__ import annotations

import json
import re
from time import monotonic
from uuid import uuid4

from app.agent.checkpoints import checkpoint
from app.agent.state import RunStatus
from app.agent.tool_agent import BoundedToolAgent
from app.artifacts.service import ArtifactService
from app.core.database import HumanApprovalRecord
from app.governance.action_guard import ActionGuard
from app.identity import ContentIdentityService
from app.identity.models import Permission
from app.tools.registry import ToolRegistry
from app.tools.application_tools import GenerateDocxTool


def report_requested(request: str) -> bool:
    return bool(re.search(r"\b(?:generate|create|draft|produce|prepare)\b.{0,80}\bmaintenance report\b", request, re.I))


async def create_maintenance_report(state, settings, db, principal, scope):
    """Use the ordinary tool boundary, including permission, policy and receipt.

    Only the released answer is presented as an assessment. Source extracts are
    labelled as evidence; neither their presence nor report creation grants a
    permit to perform maintenance.
    """
    if not report_requested(state.request) or not state.context_metrics.get("integration_route", {}).get("diagnostic"):
        return
    if state.status != RunStatus.completed or state.context_metrics.get("controlplane_status") != "released" or state.governance.get("decision") not in {"ALLOW", "ALLOW_WITH_WARNING"}:
        return
    if any(record.get("tool") == "generate_docx" for record in state.tool_records):
        return  # Existing actions are reconciled by resume, never proposed twice.
    if not principal.has_permission(Permission.artifact_create):
        state.status = RunStatus.failed
        state.warnings.append("Maintenance report was not created: artifact.create permission is required.")
        return

    route = state.context_metrics["integration_route"]
    invocation = route["diagnostic"]
    metadata = (invocation.get("run_context") or {}).get("metadata") or {}
    asset = str(metadata.get("asset_id") or "Industrial asset")
    requested_assets = {value.casefold() for value in re.findall(r"(?<![\w-])[A-Za-z]+-\d+\b", state.request)}
    if requested_assets and requested_assets != {asset.casefold()}:
        state.status = RunStatus.failed
        state.warnings.append("Maintenance report was not created: diagnostic asset identity does not match the requested asset.")
        return
    if re.search(r"\bSOP\b", state.request, re.I) and not any(
        (item.get("metadata") or {}).get("document_status") == "current"
        and (item.get("metadata") or {}).get("document_type") in {"sop", "maintenance_sop"}
        and re.search(rf"\b{re.escape(asset)}\b", str(item.get("content") or ""), re.I)
        for item in state.evidence_records
    ):
        state.status = RunStatus.failed
        state.warnings.append("Maintenance report was not created: the requested current SOP was not retrieved.")
        return
    synthetic = metadata.get("provenance") == "synthetic" or metadata.get("synthetic") is True
    title = f"{'SYNTHETIC TEST - ' if synthetic else ''}{asset} Maintenance Report"
    units = (invocation.get("inputs") or {}).get("channel_units") or []
    provenance = (
        ("SYNTHETIC TEST DATA. This report is an exercise, not a plant measurement or work order.\n" if synthetic else "")
        + f"Task: {state.id}\nAsset: {asset}\nDiagnostic domain: {invocation['domain']}\n"
        + f"Declared observation time: {metadata.get('observation_timestamp_utc', 'not supplied')}\n"
        + f"Declared channel units: {', '.join(map(str, units)) or 'see input contract'}\n"
        + "Document/time linkage is a consistency check, not physical measurement attestation."
    )
    procedures = []
    for source in state.evidence_records:
        info = source.get("metadata") or {}
        if info.get("document_status") == "current" and info.get("document_type") in {"maintenance_sop", "sop", "equipment_manual", "operating_limits"}:
            identifier = source.get("id") or source.get("chunk_id")
            procedures.append(f"[{identifier}] Revision {info.get('revision', 'not supplied')}: {source.get('content', '')}")
    comparison_note = (
        "Normalized diagnostic channels cannot be compared numerically with SOP limits expressed in engineering units. "
        "Logged document readings and the independent synthetic signal are separate evidence streams."
        if "normalized" in units else
        "Apply only limits for the same asset, metric, units and operating conditions; missing evidence requires review."
    )
    args = {
        "filename": f"maintenance_{state.id}.docx", "title": title,
        "sections": [
            {"heading": "Provenance and scope", "content": provenance},
            {"heading": "ControlPlane-released assessment", "content": state.final_response},
            {"heading": "SOP and manual comparison boundaries", "content": comparison_note},
            {"heading": "Retrieved current procedure evidence (verbatim)", "content": "\n\n".join(procedures) or "No current SOP/manual was retrieved; procedure comparison remains unverified."},
            {"heading": "Maintenance authorization", "content": "Advisory report only. No plant command or work order was executed. Physical intervention requires the applicable human authorization, permit and isolation procedure."},
        ],
        "sources": [{"file": str(s.get("file") or "Evidence"), "source_id": str(s.get("chunk_id") or ""),
                     "page": s.get("page"), "section": f"{s.get('section') or 'Evidence'} [{s.get('chunk_id') or 'unidentified'}]"} for s in state.sources],
    }
    # This adapter invokes one registered formatting tool; it never asks a model
    # to choose, rewrite or expand the verified assessment.
    tools = ToolRegistry()
    tools.register(GenerateDocxTool(ArtifactService(db, settings.workspace_root / "artifacts"), scope))
    executor = BoundedToolAgent(None, "format-only", tools,
                                ActionGuard(settings.tools_config), principal=principal)
    result = await executor._invoke_tool(state, "generate_docx", args, len(state.tool_records) + 1,
                                         "Format the released assessment and identified evidence.", monotonic() + 60)
    waiting = next((r for r in state.tool_records if r.get("waiting_for_approval") and not r.get("approval_id")), None)
    if waiting:
        approval = HumanApprovalRecord(id=str(uuid4()), run_id=state.id, tool="generate_docx",
            args_json=json.dumps(waiting["arguments"]), risk=str(waiting["risk"]), status="pending",
            requester_id=principal.user_id, organization_id=scope.organization_id, workspace_id=scope.workspace_id,
            action_hash=ContentIdentityService().hash_json({"tool": "generate_docx", "arguments": waiting["arguments"]}))
        db.add(approval)
        db.commit()
        waiting["approval_id"] = approval.id
    elif not result or not result.success:
        state.status = RunStatus.failed
        state.warnings.append("The assessment was released, but maintenance report generation did not succeed.")
    await checkpoint(state, "maintenance_report_saved")
