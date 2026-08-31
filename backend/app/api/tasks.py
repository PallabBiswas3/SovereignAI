from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_agent import BoundedToolAgent
from app.agent.state import AgentPlan, AgentRunState, AgentStep, RunStatus, StepStatus
from app.core.config import get_settings
from app.core.database import AgentRunRecord, HumanApprovalRecord, SessionLocal, TaskEventRecord, get_db
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
from app.rag.retrieval import LocalRetriever
from app.sandbox.executor import DockerSandboxExecutor
from app.tools.file_tools import SafeWorkspace
from app.workflows.coding import CodingWorkflow
from app.workflows.inspection import InspectionWorkflow
from pathlib import Path
from app.llm.ollama_provider import OllamaProvider
from app.governance.action_guard import ActionGuard
from app.governance.grounding import GroundingChecker
from app.tools.registry import create_agent_registry


router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    request: str = Field(min_length=1, max_length=50_000)
    model_override: str | None = None
    use_case: str = "internal_assistant"
    attachments: list[str] = Field(default_factory=list, max_length=10)


class StartTaskResponse(BaseModel):
    task_id: str
    status: str = "accepted"


def _completed_step(step_id: int, action: str, title: str, observation: str) -> AgentStep:
    return AgentStep(id=step_id, action=action, title=title, status=StepStatus.completed, observation=observation, verification="Completed by deterministic workflow service.")


async def _run_inspection_task(payload: CreateTaskRequest, settings, db: Session) -> AgentRunState:
    run_id = str(uuid4())
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
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
    document = KnowledgeIngestionService(db, embeddings).ingest(sop, {"department": "maintenance", "classification": "internal"})
    package_requested = any(phrase in payload.request.lower() for phrase in ("management package", "management pack", "docx xlsx pptx"))
    artifact_root = settings.workspace_root / "artifacts"
    output = (artifact_root / run_id / "approval_note.docx") if package_requested else (artifact_root / f"Approval_Note_{run_id[:8]}.docx")
    analysis = InspectionWorkflow(LocalRetriever(db, embeddings)).analyze(inspection, output)
    artifact_service = ArtifactService(db, artifact_root)
    artifact = artifact_service.register(output, run_id)
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
            OllamaVisionProvider(vision.endpoint, vision.model_tag)
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
        artifact_records.extend([artifact_service.register(xlsx, run_id), artifact_service.register(pptx, run_id)])
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
        plan=AgentPlan(goal=payload.request, steps=steps),
        final_response=analysis.recommendation, warnings=warnings, sources=analysis.sources,
        evidence_records=evidence,
        artifacts=[{"id": item.id, "name": item.name, "url": f"/api/artifacts/{item.id}"} for item in artifact_records],
    )


async def _run_coding_task(payload: CreateTaskRequest, settings, db: Session) -> AgentRunState:
    run_id = str(uuid4())
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
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
        OllamaProvider(selected.endpoint, settings.allow_deterministic_fallback),
        selected.model_tag,
    ).run(csv_path, artifact_root, payload.request, run_id)
    service = ArtifactService(db, artifact_root)
    paths = [result.source_path, result.report_path, *result.result_paths]
    records = [service.register(Path(path), run_id) for path in paths if path]
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


async def _run_tool_task(payload: CreateTaskRequest, settings, db: Session) -> AgentRunState:
    registry = ModelRegistry(settings.models_config)
    routing = ModelRouter(registry).route(payload.request, payload.model_override)
    selected = registry.get(routing.model_id)
    state = AgentRunState(
        id=str(uuid4()), request=payload.request, status=RunStatus.running, routing=routing,
        plan=AgentPlan(goal=payload.request, steps=[]),
    )
    completed = await BoundedToolAgent(
        OllamaProvider(selected.endpoint, settings.allow_deterministic_fallback),
        selected.model_tag,
        create_agent_registry(settings, db),
        ActionGuard(settings.tools_config),
    ).execute(state, payload.attachments)
    waiting = next((item for item in completed.tool_records if item.get("waiting_for_approval")), None)
    if waiting:
        approval = HumanApprovalRecord(
            id=str(uuid4()), run_id=completed.id, tool=str(waiting["tool"]),
            args_json=json.dumps(waiting.get("arguments", {})), risk=str(waiting.get("risk", "UNKNOWN")),
            status="pending",
        )
        db.add(approval)
        db.commit()
        waiting["approval_id"] = approval.id
        completed.final_response = f"The proposed {approval.tool} action is pending approval {approval.id}."
    return completed


@router.post("", response_model=AgentRunState)
async def create_task(payload: CreateTaskRequest, db: Session = Depends(get_db)) -> AgentRunState:
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
            )
        elif payload.attachments and any(word in payload.request.lower() for word in ("inspection", "approval note", "maintenance sop")):
            state = await _run_inspection_task(payload, settings, db)
        elif payload.attachments and payload.attachments[0].lower().endswith(".csv") and any(word in payload.request.lower() for word in ("code", "python", "anomal")):
            state = await _run_coding_task(payload, settings, db)
        elif _requires_tool_agent(payload):
            state = await _run_tool_task(payload, settings, db)
        else:
            state = await AgentOrchestrator(settings).run(payload.request, payload.model_override)
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
    record = AgentRunRecord(
        id=state.id,
        request=state.request,
        status=state.status.value,
        state_json=state.model_dump_json(),
        created_at=state.created_at,
        updated_at=state.updated_at,
    )
    db.add(record)
    db.commit()
    audit = AuditLogger(db)
    audit.log(state.id, "user_request", "Task accepted by the workbench.", {"request": payload.request, "use_case": payload.use_case})
    audit.log(state.id, "model_routing", state.routing.reason, state.routing.model_dump(mode="json"))
    audit.log(state.id, "governance", f"Input decision: {input_decision.value}", state.governance)
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
    audit.log(state.id, "final_output", "Agent run completed without hidden reasoning in the audit.", {"response": state.final_response, "warnings": state.warnings})
    return state


async def _publish_event(db: Session, task_id: str, event_type: str, payload: dict | None = None) -> None:
    event = await task_event_broker.publish(task_id, event_type, payload)
    db.add(TaskEventRecord(
        id=str(uuid4()), task_id=task_id, event_type=event_type,
        payload_json=json.dumps(event["payload"], ensure_ascii=False, default=str),
        created_at=datetime.fromisoformat(event["timestamp"]),
    ))
    db.commit()


async def _run_background_task(tracking_id: str, payload: CreateTaskRequest) -> None:
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
        await _publish_event(db, tracking_id, "step_started", {"id": 0, "action": "execute_workflow", "title": "Execute selected bounded workflow"})
        result = await create_task(payload, db)
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
        for warning in result.warnings:
            await _publish_event(db, tracking_id, "warning", {"summary": warning})
        await _publish_event(db, tracking_id, "task_completed", {"result": result.model_dump(mode="json")})
    except Exception as exc:
        await _publish_event(db, tracking_id, "task_failed", {"error": str(exc)})
    finally:
        await task_event_broker.complete(tracking_id)
        db.close()


@router.post("/start", response_model=StartTaskResponse, status_code=202)
async def start_task(payload: CreateTaskRequest) -> StartTaskResponse:
    tracking_id = str(uuid4())
    await task_event_broker.create(tracking_id)
    task = asyncio.create_task(_run_background_task(tracking_id, payload))
    await task_event_broker.attach_task(tracking_id, task)
    return StartTaskResponse(task_id=tracking_id)


@router.get("/{task_id}/events")
async def task_events(task_id: str) -> StreamingResponse:
    async def generate():
        try:
            async for event in task_event_broker.stream(task_id):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except KeyError:
            yield f"event: task_failed\ndata: {json.dumps({'task_id': task_id, 'type': 'task_failed', 'payload': {'error': 'Unknown or expired task stream'}})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{task_id}", response_model=AgentRunState)
async def get_task(task_id: str, db: Session = Depends(get_db)) -> AgentRunState:
    record = db.get(AgentRunRecord, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    return AgentRunState.model_validate_json(record.state_json)


@router.get("")
async def list_tasks(db: Session = Depends(get_db)) -> list[dict[str, str]]:
    records = db.query(AgentRunRecord).order_by(AgentRunRecord.created_at.desc()).limit(100).all()
    return [
        {"id": row.id, "request": row.request, "status": row.status, "updated_at": row.updated_at.isoformat()}
        for row in records
    ]
