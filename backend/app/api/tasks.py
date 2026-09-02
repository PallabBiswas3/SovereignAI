from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import json
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_agent import BoundedToolAgent
from app.agent.state import AgentPlan, AgentRunState, AgentStep, RunStatus, StepStatus
from app.core.config import get_settings
from app.core.database import AgentRunRecord, ArtifactRecord, HumanApprovalRecord, SessionLocal, TaskAccessRecord, TaskEventRecord, get_db
from app.core.events import task_event_broker
from app.audit.logger import AuditLogger
from app.governance.injection import PromptInjectionScanner
from app.governance.pii import PIIDetector
from app.governance.policy_engine import PolicyEngine
from app.governance.policy_engine import GovernanceDecision
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.artifacts.service import ArtifactService
from app.artifacts.xlsx_generator import XlsxGenerator
from app.artifacts.pptx_generator import PptxGenerator
from app.documents.evidence import MultiFileEvidenceProcessor
from app.multimodal.vision import OllamaVisionProvider
from app.rag.embeddings import configured_embedding_provider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.factory import configured_hybrid_retriever
from app.sandbox.executor import DockerSandboxExecutor
from app.tools.file_tools import SafeWorkspace
from app.workflows.coding import CodingWorkflow
from app.workflows.inspection import InspectionWorkflow
from pathlib import Path
from app.llm.ollama_provider import OllamaProvider
from app.llm.base import ModelGenerationCancelled
from app.orchestration.execution_mode import ExecutionMode, ExecutionModeSelector
from app.governance.action_guard import ActionGuard
from app.governance.grounding import GroundingChecker
from app.tools.registry import create_agent_registry
from app.resources.cache import get_cache_backend
from app.multimodal.ocr import DocumentTextExtractor, LocalOCRService
from app.workcells.defaults import configured_workcell_registry, create_workcell_handler_registry
from app.workcells.executor import WorkcellExecutor
from app.workcells.handlers import WorkcellHandlerContext
from app.workcells.models import WorkcellDefinition
from app.identity.authorization import AuthorizationService
from app.identity.dependencies import require_permission
from app.identity.models import ClearanceLevel, DocumentACL, Permission, Principal, ResourceScope
from app.identity.provider import LocalIdentityProvider
from app.identity import ContentIdentityService


router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field(min_length=1, max_length=50_000)
    model_override: str | None = None
    use_case: str = "internal_assistant"
    attachments: list[str] = Field(default_factory=list, max_length=10)
    execution_mode: ExecutionMode = ExecutionMode.automatic
    workcell_id: str | None = Field(default=None, max_length=100)


class StartTaskResponse(BaseModel):
    task_id: str
    status: str = "accepted"


EventCallback = Callable[[str, dict[str, object]], Awaitable[None]]


def _mode_selection(payload: CreateTaskRequest, routing):
    return ExecutionModeSelector().select(
        payload.execution_mode, payload.request, routing.task_profile, len(payload.attachments)
    )


def _completed_step(step_id: int, action: str, title: str, observation: str) -> AgentStep:
    return AgentStep(id=step_id, action=action, title=title, status=StepStatus.completed, observation=observation, verification="Completed by deterministic workflow service.")


async def _run_inspection_task_implementation(
    payload: CreateTaskRequest, settings, db: Session, principal: Principal, *, workcell_identity: str | None = None
) -> AgentRunState:
    run_id = str(uuid4())
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
    selection = _mode_selection(payload, routing)
    workspace = SafeWorkspace(settings.workspace_root)
    attachment_paths = [workspace.resolve(item, must_exist=True) for item in payload.attachments]
    uploaded_sops = [path for path in attachment_paths if "sop" in path.stem.lower()]
    candidates = [path for path in attachment_paths if path not in uploaded_sops and path.suffix.lower() in {".pdf", ".md", ".txt", ".docx"}]
    if not candidates:
        raise ValueError("Inspection workflow requires a PDF, Word, Markdown, or text inspection report")
    inspection = next((path for path in candidates if "inspection" in path.stem.lower()), candidates[0])
    if uploaded_sops:
        sop = uploaded_sops[0]
    else:
        sop_name = "Maintenance_SOP.pdf" if (settings.knowledge_root / "Maintenance_SOP.pdf").exists() else "Maintenance_SOP.md"
        sop = SafeWorkspace(settings.knowledge_root).resolve(sop_name, must_exist=True)
    embeddings = configured_embedding_provider()
    cache = get_cache_backend() if settings.cache_enabled else None
    scope = AuthorizationService.owned_scope(principal)
    secure_acl = DocumentACL(**scope.model_dump(exclude={"resource_id"})) if settings.auth_mode.lower() == "local" else None
    document = KnowledgeIngestionService(db, embeddings).ingest(
        sop, {"department": "maintenance", "classification": "internal"},
        acl=secure_acl, require_acl=secure_acl is not None,
    )
    package_requested = any(phrase in payload.request.lower() for phrase in ("management package", "management pack", "docx xlsx pptx"))
    artifact_root = settings.workspace_root / "artifacts"
    output = (artifact_root / run_id / "approval_note.docx") if package_requested else (artifact_root / f"Approval_Note_{run_id[:8]}.docx")
    selected_model = registry.get(routing.model_id)
    analysis = InspectionWorkflow(
        configured_hybrid_retriever(
            db, embeddings=embeddings, cache=cache, settings=settings,
            execution_mode=selection.selected.value,
            principal=principal if settings.auth_mode.lower() == "local" else None,
        ),
        DocumentTextExtractor(LocalOCRService(cache=cache)),
        cache,
        workcell_identity=workcell_identity,
    ).analyze(
        inspection, output, payload.request,
        selected_model=selected_model.model_tag,
        context_window=selected_model.context_length,
        execution_mode=selection.selected.value,
    )
    structured = analysis.evidence_bundle
    artifact_service = ArtifactService(db, artifact_root)
    artifact = artifact_service.register(output, run_id, scope=scope)
    artifact_records = [artifact]
    extra_paths = [path for path in attachment_paths if path not in {inspection, sop}]
    evidence = [{
        "file": inspection.name, "media_type": "inspection/report", "processor": "deterministic-inspection",
        "summary": f"Extracted {len(analysis.findings)} supported measurements.",
        "provenance": {"file": inspection.name, "path": str(inspection)},
        "metadata": {"findings": analysis.findings},
    }, {
        "file": sop.name, "media_type": "knowledge/sop", "processor": "semantic-ingestion",
        "summary": f"Indexed {document.chunk_count} provenance chunks.",
        "provenance": {"file": sop.name, "path": str(sop)},
        "metadata": {"document_id": document.id},
    }]
    if extra_paths:
        vision = registry.get("vision")
        processed = await MultiFileEvidenceProcessor(
            OllamaVisionProvider(vision.endpoint, vision.model_tag, cache=cache)
        ).process(extra_paths, payload.request)
        evidence.extend(item.model_dump(mode="json") for item in processed)
    if package_requested:
        package_root = artifact_root / run_id
        xlsx = XlsxGenerator().generate(package_root / "inspection_analysis.xlsx", "Inspection Analysis", analysis.findings)
        pptx = PptxGenerator().generate(package_root / "management_briefing.pptx", f"{analysis.equipment} Management Briefing", [
            {"title": "Evidence reviewed", "bullets": [record["file"] for record in evidence]},
            {"title": "Inspection findings", "bullets": [f"{item['parameter']}: {item['observed']} ({item['status']})" for item in analysis.findings]},
            {"title": "Recommendation", "bullets": [analysis.recommendation, "Human engineering authorization remains required."]},
        ])
        artifact_records.extend([
            artifact_service.register(xlsx, run_id, scope=scope),
            artifact_service.register(pptx, run_id, scope=scope),
        ])
    observations = [
        f"Task classified as {routing.task_profile.task_type}.",
        f"Selected {routing.model_id}: {routing.reason}",
        f"Inspected {inspection.name}; scanned={bool(analysis.ocr)}.",
        (f"Local OCR completed at {analysis.ocr['mean_confidence']:.0%} confidence." if analysis.ocr else "Digital text extracted; OCR was not required."),
        f"Extracted {len(analysis.findings)} supported measurements.",
        f"Ingested {document.filename} into {document.chunk_count} provenance chunks.",
        f"Retrieved {len(analysis.sources)} applicable SOP sections.",
        "Compared observed values against retrieved limits.",
        "Attached file, page, and section citations.",
        "Verified each reported finding has an evidence record.",
        analysis.recommendation,
        f"Created {len(artifact_records)} registered artifact(s).",
        "Applied engineering governance and retained human authorization warning.",
    ]
    actions = [
        "classify_task", "route_model", "inspect_document", "ocr_document", "extract_measurements",
        "ingest_sop", "retrieve_sources", "compare_readings", "attach_citations", "verify_claims",
        "recommend_disposition", "create_docx", "governance_check",
    ]
    titles = [
        "Classify industrial task", "Select local model", "Inspect uploaded report", "Run local OCR",
        "Extract equipment measurements", "Index maintenance SOP", "Retrieve applicable SOP sections",
        "Compare readings with allowed limits", "Attach source evidence", "Verify factual claims",
        "Generate recommendation", "Create approval note", "Apply governance controls",
    ]
    warnings = []
    if analysis.ocr and analysis.ocr.get("warning"):
        warnings.append(str(analysis.ocr["warning"]))
    steps = [_completed_step(index + 1, action, title, observations[index]) for index, (action, title) in enumerate(zip(actions, titles))]
    if package_requested:
        steps.append(_completed_step(14, "create_management_package", "Create management workbook and briefing", "Generated and validated DOCX, XLSX, and PPTX outputs."))
    return AgentRunState(
        id=run_id, request=payload.request, status=RunStatus.completed, routing=routing,
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
        plan=AgentPlan(goal=payload.request, steps=steps),
        final_response=analysis.recommendation, warnings=warnings, sources=analysis.sources,
        evidence_records=evidence,
        measurements=list(structured.get("measurements", [])),
        rules=list(structured.get("rules", [])),
        calculations=list(structured.get("calculations", [])),
        claims=list(structured.get("claims", [])),
        conflicts=list(structured.get("conflicts", [])),
        context_metrics=dict(analysis.compiled_context.get("metrics", {})),
        retrieval_metrics=analysis.retrieval_metrics,
        artifacts=[{"id": item.id, "name": item.name, "url": f"/api/artifacts/{item.id}"} for item in artifact_records],
    )


async def _run_inspection_workcell(
    payload: CreateTaskRequest,
    settings,
    db: Session,
    definition: WorkcellDefinition,
    principal: Principal,
    event_callback: EventCallback | None = None,
) -> AgentRunState:
    async def runner() -> AgentRunState:
        return await _run_inspection_task_implementation(
            payload, settings, db, principal,
            workcell_identity=f"{definition.manifest.id}:{definition.manifest.version}",
        )

    context = WorkcellHandlerContext(
        task_id="pending",
        request=payload.request,
        definition=definition,
        inputs={
            "request": payload.request,
            "attachments": payload.attachments,
            "execution_mode": payload.execution_mode.value,
        },
        services={"pump_inspection_runner": runner},
    )
    execution = await WorkcellExecutor(create_workcell_handler_registry()).execute(
        context,
        event_callback=event_callback,
    )
    state = context.accumulated.get("task_state")
    if not isinstance(state, AgentRunState):
        raise RuntimeError("WORKCELL_EXECUTION_FAILED: Pump handler returned no task state")
    state.workcell_id = definition.manifest.id
    state.workcell_version = definition.manifest.version
    state.workcell_hash = definition.content_hash
    state.workflow_version = definition.workflow.version
    state.workcell_state = execution.model_dump(mode="json")
    claim_ids = [str(item.get("id")) for item in state.claims if item.get("id")]
    for artifact in db.query(ArtifactRecord).filter(ArtifactRecord.run_id == state.id).all():
        artifact.workcell_id = state.workcell_id
        artifact.workcell_version = state.workcell_version
        artifact.lineage_json = json.dumps({"derived_from_claims": claim_ids})
    db.commit()
    return state


async def _run_coding_task(payload: CreateTaskRequest, settings, db: Session, principal: Principal) -> AgentRunState:
    run_id = str(uuid4())
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
    selection = _mode_selection(payload, routing)
    csv_path = SafeWorkspace(settings.workspace_root).resolve(payload.attachments[0], must_exist=True)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Coding workflow requires a CSV attachment")
    action = ActionGuard(settings.tools_config).evaluate("run_python")
    if action.decision != GovernanceDecision.allow:
        raise ValueError(f"Sandbox execution was not authorized: {action.reason}")
    artifact_root = settings.workspace_root / "artifacts"
    selected = registry.get(routing.model_id)
    result = await CodingWorkflow(
        DockerSandboxExecutor(settings.workspace_root / "sandbox"),
        OllamaProvider(
            selected.endpoint, settings.allow_deterministic_fallback,
            role=selected.role, memory_requirement=selected.memory_requirement,
            execution_mode=selection.selected.value, priority=selection.priority,
        ),
        selected.model_tag,
    ).run(csv_path, artifact_root, payload.request, run_id)
    service = ArtifactService(db, artifact_root)
    paths = [result.source_path, result.report_path, *result.result_paths]
    scope = AuthorizationService.owned_scope(principal)
    records = [service.register(Path(path), run_id, scope=scope) for path in paths if path]
    executed = bool(result.execution.get("executed"))
    succeeded = executed and result.execution.get("exit_code") == 0
    steps = [
        _completed_step(1, "inspect_csv", "Inspect sensor dataset", f"Validated {csv_path.name}."),
        _completed_step(2, "write_python", "Generate reusable code with local coder model", f"Created code using {result.model_used}."),
        AgentStep(id=3, action="run_python", title="Execute and repair in isolated Docker sandbox", status=StepStatus.completed if succeeded else StepStatus.failed, observation=f"Completed {len(result.attempts)} bounded attempt(s).", error=None if succeeded else str(result.execution.get("stderr"))),
        AgentStep(id=4, action="verify_output", title="Verify anomaly output", status=StepStatus.completed if succeeded else StepStatus.failed, observation=("Output and exit code verified." if succeeded else "Output verification could not complete without Docker.")),
        _completed_step(5, "generate_report", "Generate source and analysis report", f"Registered {len(records)} downloadable artifacts."),
    ]
    return AgentRunState(
        id=run_id, request=payload.request, status=RunStatus.completed if succeeded else RunStatus.failed,
        routing=routing, plan=AgentPlan(goal=payload.request, steps=steps),
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
        final_response=("Anomaly detector executed and its output was verified." if succeeded else "The reusable anomaly detector and report were created, but execution could not be verified because Docker is unavailable."),
        warnings=result.warnings,
        artifacts=[{"id": record.id, "name": record.name, "url": f"/api/artifacts/{record.id}"} for record in records],
        execution_records=[attempt.model_dump(mode="json") for attempt in result.attempts],
    )


def _requires_tool_agent(payload: CreateTaskRequest) -> bool:
    if payload.attachments:
        return True
    request = payload.request.lower()
    return any(phrase in request for phrase in (
        "read file", "list file", "search file", "knowledge base", "search knowledge",
        "internal document", "ocr", "analyze image", "create report", "generate report",
        "spreadsheet", "presentation", "powerpoint", "word document",
    ))


async def _run_tool_task(payload: CreateTaskRequest, settings, db: Session, principal: Principal) -> AgentRunState:
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
    selection = _mode_selection(payload, routing)
    selected = registry.get(routing.model_id)
    state = AgentRunState(
        id=str(uuid4()), request=payload.request, status=RunStatus.running, routing=routing,
        plan=AgentPlan(goal=payload.request, steps=[]),
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
    )
    completed = await BoundedToolAgent(
        OllamaProvider(
            selected.endpoint, settings.allow_deterministic_fallback,
            role=selected.role, memory_requirement=selected.memory_requirement,
            execution_mode=selection.selected.value, priority=selection.priority,
        ),
        selected.model_tag,
        create_agent_registry(settings, db, principal, AuthorizationService.owned_scope(principal)),
        ActionGuard(settings.tools_config),
        principal=principal,
    ).execute(state, payload.attachments)
    waiting = next((item for item in completed.tool_records if item.get("waiting_for_approval")), None)
    if waiting:
        approval = HumanApprovalRecord(
            id=str(uuid4()), run_id=completed.id, tool=str(waiting["tool"]),
            args_json=json.dumps(waiting.get("arguments", {})), risk=str(waiting.get("risk", "UNKNOWN")),
            status="pending",
            requester_id=principal.user_id, organization_id=principal.organization_id,
            workspace_id=next(iter(principal.workspace_ids), None),
            action_hash=ContentIdentityService().hash_json({"tool": waiting["tool"], "arguments": waiting.get("arguments", {})}),
        )
        db.add(approval)
        db.commit()
        waiting["approval_id"] = approval.id
        completed.final_response = f"The proposed {approval.tool} action is pending approval {approval.id}."
    return completed


async def _execute_task(
    payload: CreateTaskRequest,
    db: Session,
    principal: Principal,
    event_callback: EventCallback | None = None,
    cancellation_event: asyncio.Event | None = None,
) -> AgentRunState:
    settings = get_settings()
    policy_engine = PolicyEngine(settings.policies_config)
    policy = policy_engine.get(payload.use_case)
    pii = PIIDetector().detect(payload.request)
    injections = PromptInjectionScanner().scan(payload.request)
    input_decision = policy_engine.decide(policy, pii_count=len(pii), injection_count=len(injections))
    try:
        if input_decision == GovernanceDecision.block:
            registry = ModelRegistry(settings.models_config)
            routing = ModelRouter(registry).route(payload.request, payload.model_override)
            selection = _mode_selection(payload, routing)
            state = AgentRunState(
                id=str(uuid4()), request=payload.request, status=RunStatus.completed,
                routing=routing,
                plan=AgentPlan(goal=payload.request, steps=[AgentStep(
                    id=1, action="governance_check", title="Block unsafe instruction pattern",
                    status=StepStatus.completed, observation="Prompt-injection pattern detected; model and tools were not invoked.",
                    verification="Control-plane block applied before inference.",
                )]),
                final_response="This request was blocked because it contains an instruction pattern that conflicts with workbench safety policy.",
                warnings=["No model or tool was invoked."],
                requested_execution_mode=selection.requested.value,
                execution_mode=selection.selected.value,
                execution_mode_reason=selection.reason,
            )
        elif payload.workcell_id or (payload.attachments and any(word in payload.request.lower() for word in ("inspection", "approval note", "maintenance sop"))):
            workcells = configured_workcell_registry(settings)
            if payload.workcell_id:
                definition = workcells.get(payload.workcell_id)
            else:
                definition = workcells.resolve_for_task("engineering_inspection")
                if definition is None:
                    raise ValueError("WORKCELL_NOT_FOUND: no ready engineering inspection Workcell")
            if definition.manifest.id != "pump-inspection":
                raise ValueError(f"WORKCELL_EXECUTION_FAILED: no trusted task adapter for {definition.manifest.id}")
            if event_callback:
                await event_callback("workcell_selected", {
                    "workcell_id": definition.manifest.id,
                    "version": definition.manifest.version,
                    "hash": definition.content_hash,
                })
                await event_callback("workcell_validated", {"status": "READY"})
            access = AuthorizationService().can_execute_workcell(principal, definition.manifest.id)
            if not access.allowed:
                raise HTTPException(status_code=403, detail={"code": access.reason_code})
            state = await _run_inspection_workcell(payload, settings, db, definition, principal, event_callback)
        elif payload.attachments and payload.attachments[0].lower().endswith(".csv") and any(word in payload.request.lower() for word in ("code", "python", "anomal")):
            access = AuthorizationService().can_use_tool(principal, "run_python")
            if not access.allowed:
                raise HTTPException(status_code=403, detail={"code": access.reason_code})
            state = await _run_coding_task(payload, settings, db, principal)
        elif _requires_tool_agent(payload):
            state = await _run_tool_task(payload, settings, db, principal)
        else:
            state = await AgentOrchestrator(settings).run(
                payload.request,
                payload.model_override,
                payload.execution_mode,
                len(payload.attachments),
                event_callback,
                cancellation_event,
            )
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    grounding_score = 1.0 if state.sources else None
    claim_results = []
    unsupported_claims: list[str] = []
    final_decision = input_decision
    if state.tool_records and state.final_response:
        grounding = GroundingChecker(threshold=policy.grounding_threshold).evaluate(state.final_response, state.sources)
        grounding_score = grounding.grounding_confidence
        claim_results = [claim.model_dump(mode="json") for claim in grounding.claims]
        unsupported_claims = grounding.unsupported_material_claims
        grounding_decision = policy_engine.decide(policy, grounding_score=grounding_score)
        if grounding_decision != GovernanceDecision.allow:
            final_decision = grounding_decision
        elif unsupported_claims:
            final_decision = GovernanceDecision.rewrite
        if unsupported_claims:
            state.warnings.append("Material claims without strong source support were flagged for review.")
        if final_decision == GovernanceDecision.require_human_approval:
            state.status = RunStatus.waiting_for_approval
            state.final_response = "The draft requires human review because its material claims did not meet the engineering grounding threshold."
    state.governance = {
        "decision": final_decision.value,
        "input_decision": input_decision.value,
        "policy": policy.name,
        "privacy_risk": "medium" if pii else "low",
        "pii_findings": len(pii),
        "injection_findings": len(injections),
        "grounding_score": grounding_score,
        "claim_results": claim_results,
        "unsupported_material_claims": unsupported_claims,
    }
    scope = AuthorizationService.owned_scope(principal)
    state.principal_id = principal.user_id
    state.organization_id = principal.organization_id
    state.workspace_id = scope.workspace_id
    state.department_id = scope.department_id
    state.classification = scope.classification.name.upper()
    record = AgentRunRecord(
        id=state.id,
        request=state.request,
        status=state.status.value,
        state_json=state.model_dump_json(),
        created_at=state.created_at,
        updated_at=state.updated_at,
        organization_id=scope.organization_id, owner_id=scope.owner_id,
        workspace_id=scope.workspace_id, department_id=scope.department_id,
        classification=scope.classification.name.upper(),
    )
    db.add(record)
    db.commit()
    audit = AuditLogger(db, principal)
    audit.log(state.id, "user_request", "Task accepted by the workbench.", {"request": payload.request, "use_case": payload.use_case})
    audit.log(state.id, "model_routing", state.routing.reason, state.routing.model_dump(mode="json"))
    audit.log(state.id, "execution_mode", state.execution_mode_reason, {
        "requested": state.requested_execution_mode, "selected": state.execution_mode,
    })
    audit.log(state.id, "governance", f"Input decision: {input_decision.value}", state.governance)
    if state.workcell_id:
        audit.log(state.id, "workcell_selected", f"Selected {state.workcell_id} {state.workcell_version}", {
            "workcell_id": state.workcell_id, "version": state.workcell_version,
        })
        audit.log(state.id, "workcell_validation", f"Validated {state.workcell_id} {state.workcell_version}", {
            "workcell_id": state.workcell_id,
            "version": state.workcell_version,
            "hash": state.workcell_hash,
            "workflow_version": state.workflow_version,
        })
        for step_id in state.workcell_state.get("completed_steps", []):
            audit.log(state.id, "workcell_step_completed", f"Workcell step completed: {step_id}", {"step_id": step_id})
    for step in state.plan.steps:
        audit.log(state.id, "agent_step", f"{step.title}: {step.status.value}", {"action": step.action, "status": step.status.value, "observation": step.observation, "verification": step.verification, "error": step.error})
    for execution in state.execution_records:
        audit.log(
            state.id, "code_execution",
            f"Code attempt {execution.get('attempt')} verified={execution.get('verified')}",
            execution,
        )
    for tool_call in state.tool_records:
        audit.log(
            state.id, "tool_call",
            f"Tool {tool_call.get('tool')} success={tool_call.get('success')}",
            tool_call,
        )
    for artifact in state.artifacts:
        audit.log(state.id, "artifact_created", f"Created {artifact.get('name')}", artifact)
    for claim in state.claims:
        audit.log(state.id, "claim_verification", f"Claim {claim.get('id')}: {claim.get('support_status')}", claim)
    for calculation in state.calculations:
        audit.log(state.id, "deterministic_calculation", f"Calculation {calculation.get('id')} verified={calculation.get('verified')}", calculation)
    for conflict in state.conflicts:
        audit.log(state.id, "evidence_conflict", str(conflict.get("summary", "Evidence conflict")), conflict)
    if state.context_metrics:
        audit.log(state.id, "context_compilation", "Compiled bounded evidence context.", state.context_metrics)
    audit.log(state.id, "final_output", "Agent run completed without hidden reasoning in the audit.", {"response": state.final_response, "warnings": state.warnings})
    return state


def _task_scope(row: AgentRunRecord | TaskAccessRecord) -> ResourceScope | None:
    if not row.organization_id or not row.workspace_id:
        return None
    return ResourceScope(
        resource_id=getattr(row, "id", None) or getattr(row, "task_id", None),
        organization_id=row.organization_id, owner_id=row.owner_id,
        workspace_id=row.workspace_id, department_id=row.department_id,
        classification=ClearanceLevel.parse(row.classification),
    )


def _assert_task_access(principal: Principal, row: AgentRunRecord | TaskAccessRecord) -> None:
    if get_settings().auth_mode.lower() != "local":
        return
    scope = _task_scope(row)
    decision = AuthorizationService().authorize(principal, Permission.task_read, scope) if scope else None
    if not decision or not decision.allowed:
        raise HTTPException(status_code=404, detail="Task not found")


@router.post("", response_model=AgentRunState)
async def create_task(
    payload: CreateTaskRequest,
    principal: Principal = Depends(require_permission(Permission.task_create)),
    db: Session = Depends(get_db),
) -> AgentRunState:
    return await _execute_task(payload, db, principal)


async def _publish_event(db: Session, task_id: str, event_type: str, payload: dict | None = None) -> None:
    event = await task_event_broker.publish(task_id, event_type, payload)
    db.add(TaskEventRecord(
        id=str(uuid4()), task_id=task_id, event_type=event_type,
        payload_json=json.dumps(event["payload"], ensure_ascii=False, default=str),
        created_at=datetime.fromisoformat(event["timestamp"]),
    ))
    db.commit()


async def _run_background_task(tracking_id: str, payload: CreateTaskRequest, principal: Principal) -> None:
    db = SessionLocal()
    try:
        await _publish_event(db, tracking_id, "task_accepted", {"request": payload.request})
        policy_engine = PolicyEngine(get_settings().policies_config)
        policy = policy_engine.get(payload.use_case)
        pii = PIIDetector().detect(payload.request)
        injections = PromptInjectionScanner().scan(payload.request)
        decision = policy_engine.decide(policy, pii_count=len(pii), injection_count=len(injections))
        await _publish_event(db, tracking_id, "governance_completed", {"decision": decision.value, "policy": policy.name})
        routing = ModelRouter(ModelRegistry(get_settings().models_config)).route(payload.request, payload.model_override)
        await _publish_event(db, tracking_id, "task_classified", routing.task_profile.model_dump(mode="json"))
        await _publish_event(db, tracking_id, "model_selected", {"model_id": routing.model_id, "model": routing.selected_model, "reason": routing.reason})
        await _publish_event(db, tracking_id, "plan_created", {"workflow": routing.task_profile.task_type})
        selection = _mode_selection(payload, routing)
        await _publish_event(db, tracking_id, "execution_mode_selected", selection.model_dump(mode="json"))
        await _publish_event(db, tracking_id, "step_started", {"id": 0, "action": "execute_workflow", "title": "Execute selected bounded workflow"})
        cancellation_event = await task_event_broker.cancellation_event(tracking_id)

        async def publish_live(event_type: str, event_payload: dict[str, object]) -> None:
            await _publish_event(db, tracking_id, event_type, event_payload)

        result = await _execute_task(payload, db, principal, publish_live, cancellation_event)
        access_row = db.get(TaskAccessRecord, tracking_id)
        if access_row:
            access_row.run_id = result.id
            db.commit()
        await _publish_event(db, tracking_id, "step_completed", {"id": 0, "status": result.status.value, "observation": "Workflow returned a persisted run state."})
        for step in result.plan.steps:
            await _publish_event(db, tracking_id, "step_started", {"id": step.id, "action": step.action, "title": step.title})
            matching = [item for item in result.tool_records if item.get("tool") == step.action]
            for tool_call in matching:
                await _publish_event(db, tracking_id, "tool_proposed", {"tool": step.action, "arguments": tool_call.get("arguments", {}), "reason_summary": tool_call.get("reason_summary")})
                await _publish_event(db, tracking_id, "tool_started", {"tool": step.action, "arguments": tool_call.get("arguments", {})})
                await _publish_event(db, tracking_id, "tool_completed", {"tool": step.action, "success": tool_call.get("success"), "duration_ms": tool_call.get("duration_ms")})
            await _publish_event(db, tracking_id, "step_completed", {"id": step.id, "status": step.status.value, "observation": step.observation})
        for artifact in result.artifacts:
            await _publish_event(db, tracking_id, "artifact_created", artifact)
        for source in result.sources:
            await _publish_event(db, tracking_id, "source_retrieved", source)
        for calculation in result.calculations:
            await _publish_event(db, tracking_id, "calculation_completed", calculation)
        for claim in result.claims:
            await _publish_event(db, tracking_id, "claim_verified", claim)
        for conflict in result.conflicts:
            await _publish_event(db, tracking_id, "evidence_conflict", conflict)
        for warning in result.warnings:
            await _publish_event(db, tracking_id, "warning", {"summary": warning})
        await _publish_event(db, tracking_id, "task_completed", {"result": result.model_dump(mode="json")})
    except ModelGenerationCancelled as exc:
        await _publish_event(db, tracking_id, "task_cancelled", {"error": str(exc)})
    except Exception as exc:
        await _publish_event(db, tracking_id, "task_failed", {"error": str(exc)})
    finally:
        await task_event_broker.complete(tracking_id)
        db.close()


@router.post("/start", response_model=StartTaskResponse, status_code=202)
async def start_task(
    payload: CreateTaskRequest,
    principal: Principal = Depends(require_permission(Permission.task_create)),
    db: Session = Depends(get_db),
) -> StartTaskResponse:
    tracking_id = str(uuid4())
    scope = AuthorizationService.owned_scope(principal)
    db.add(TaskAccessRecord(
        task_id=tracking_id, organization_id=scope.organization_id,
        owner_id=principal.user_id, workspace_id=scope.workspace_id,
        department_id=scope.department_id, classification=scope.classification.name.upper(),
    ))
    db.commit()
    await task_event_broker.create(tracking_id)
    task = asyncio.create_task(_run_background_task(tracking_id, payload, principal))
    await task_event_broker.attach_task(tracking_id, task)
    return StartTaskResponse(task_id=tracking_id)


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    principal: Principal = Depends(require_permission(Permission.task_read)),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    access_row = db.get(TaskAccessRecord, task_id)
    if not access_row:
        raise HTTPException(status_code=404, detail="Task not found")
    _assert_task_access(principal, access_row)
    try:
        await task_event_broker.cancel(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    return {"task_id": task_id, "status": "cancellation_requested"}


@router.get("/{task_id}/events")
async def task_events(
    task_id: str, request: Request,
    principal: Principal = Depends(require_permission(Permission.task_read)),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    access_row = db.get(TaskAccessRecord, task_id)
    if not access_row:
        raise HTTPException(status_code=404, detail="Task not found")
    _assert_task_access(principal, access_row)
    settings = get_settings()
    token = request.cookies.get(settings.auth_cookie_name)
    async def generate():
        try:
            async for event in task_event_broker.stream(task_id):
                if settings.auth_mode.lower() == "local":
                    if not token or LocalIdentityProvider(db, settings.access_config).resolve_principal(token) is None:
                        yield f"event: access_revoked\ndata: {json.dumps({'task_id': task_id, 'type': 'access_revoked', 'payload': {'code': 'AUTHENTICATION_REQUIRED'}})}\n\n"
                        break
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except KeyError:
            yield f"event: task_failed\ndata: {json.dumps({'task_id': task_id, 'type': 'task_failed', 'payload': {'error': 'Unknown or expired task stream'}})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{task_id}", response_model=AgentRunState)
async def get_task(
    task_id: str,
    principal: Principal = Depends(require_permission(Permission.task_read)),
    db: Session = Depends(get_db),
) -> AgentRunState:
    record = db.get(AgentRunRecord, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    _assert_task_access(principal, record)
    return AgentRunState.model_validate_json(record.state_json)


@router.get("")
async def list_tasks(
    principal: Principal = Depends(require_permission(Permission.task_read)),
    db: Session = Depends(get_db),
) -> list[dict[str, str]]:
    records = db.query(AgentRunRecord).order_by(AgentRunRecord.created_at.desc()).limit(100).all()
    if get_settings().auth_mode.lower() == "local":
        records = [row for row in records if _task_scope(row) and AuthorizationService().authorize(
            principal, Permission.task_read, _task_scope(row)
        ).allowed]
    return [
        {"id": row.id, "request": row.request, "status": row.status, "updated_at": row.updated_at.isoformat()}
        for row in records
    ]
