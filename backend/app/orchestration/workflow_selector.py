"""Select executable task paths without inference or permission grants."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from time import perf_counter

from pydantic import BaseModel, Field

from app.integrations.routing import IntegrationRoutePlan, IntegrationRoutePlanner
from app.orchestration.chat_mode import (
    ChatMode, ChatModeSelection, ChatModeSelector, extract_asset_references,
)


class WorkflowStrategy(str, Enum):
    direct = "DIRECT"
    retrieval = "RETRIEVAL"
    workflow = "WORKFLOW"
    planned = "PLANNED"
    adaptive = "ADAPTIVE"
    blocked = "BLOCKED"


class WorkflowHandler(str, Enum):
    direct = "direct_answer"
    retrieval = "authorized_retrieval"
    integration = "industrial_integration"
    workcell = "workcell"
    coding = "coding"
    tools = "tool_agent"
    planned = "structured_plan"
    blocked = "blocked"


class WorkflowSelection(BaseModel):
    selector_version: str = "workflow-selector-v1"
    strategy: WorkflowStrategy
    handler: WorkflowHandler
    rule: str
    reason: str
    domain: str = "general"
    capabilities: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    requires_validated_input: bool = False
    planning_candidate: bool = False
    fallback_reason: str | None = None
    selection_seconds: float = 0


@dataclass(frozen=True)
class WorkflowDecision:
    selection: WorkflowSelection
    chat: ChatModeSelection
    integration: IntegrationRoutePlan | None = None


_TOOL_SIGNALS = (
    "calculate", "summarize_numbers", "convert_units", "convert units", "summarize numbers",
    "read file", "list file", "search file", "knowledge base", "search knowledge",
    "internal document", "ocr", "analyze image", "create report", "generate report",
    "spreadsheet", "presentation", "powerpoint", "word document",
    "read_file", "list_files", "search_files", "get_file_metadata", "write_file", "run_python",
    "knowledge_search", "get_asset_telemetry", "get_asset_history", "ocr_document", "analyze_image",
    "generate_docx", "generate_xlsx", "generate_pptx",
)
_INSPECTION_SIGNALS = ("inspection", "approval note", "maintenance sop")


def requires_tool_agent(request: str, attachments: list[str]) -> bool:
    return bool(attachments) or any(phrase in request.lower() for phrase in _TOOL_SIGNALS)


def select_chat_mode(request: str, requested: ChatMode, attachments: list[str],
                     workcell_id: str | None, tool_scope: str) -> ChatModeSelection:
    if tool_scope == "NUMERIC":
        if attachments or workcell_id or requested not in {ChatMode.automatic, ChatMode.controlled}:
            raise ValueError("Numeric tool scope requires Automatic/Controlled mode without attachments or a Workcell")
        return ChatModeSelection(requested=requested, selected=ChatMode.controlled,
                                 reason="The user restricted this task to the numeric tool catalogue.")
    if tool_scope != "ALL":
        raise ValueError("Unknown tool scope")
    selection = ChatModeSelector().select(requested, request,
        attachment_count=len(attachments), workcell_id=workcell_id)
    if requested == ChatMode.automatic and selection.selected == ChatMode.general and requires_tool_agent(request, attachments):
        return ChatModeSelection(requested=requested, selected=ChatMode.controlled,
                                 reason="The request explicitly asks for a registered tool capability.")
    return selection


class WorkflowSelector:
    """Shared precedence for chat, integration, Workcell and tool dispatch.

    Predictable tool chains use validated plans; investigations remain adaptive.
    """

    def select(self, request: str, *, workspace_root: Path,
               requested_chat_mode: ChatMode = ChatMode.automatic,
               attachments: list[str] | None = None, workcell_id: str | None = None,
               tool_scope: str = "ALL", input_blocked: bool = False) -> WorkflowDecision:
        started = perf_counter()
        attachments = attachments or []
        chat = select_chat_mode(request, requested_chat_mode, attachments, workcell_id, tool_scope)
        normalized = " ".join(request.lower().split())

        def decision(strategy, handler, rule, reason, *, integration=None, **kwargs):
            selection = WorkflowSelection(strategy=strategy, handler=handler, rule=rule,
                reason=reason, selection_seconds=round(perf_counter() - started, 6), **kwargs)
            return WorkflowDecision(selection, chat, integration)

        if input_blocked:
            return decision(WorkflowStrategy.blocked, WorkflowHandler.blocked,
                            "input_policy_block", "Input policy blocked execution before tools or inference.")

        def tool_decision(rule, reason, numeric=False):
            investigation = bool(re.search(r"\b(investigat\w*|diagnos\w*|why|if|depending|until)\b|root cause", normalized))
            chained = bool(re.search(r"\bthen\b|followed by|after that", normalized))
            named_tools = len(set(re.findall(r"\b(calculate|summarize_numbers|convert_units)\b", normalized)))
            capability_groups = (
                ("read file", "read_file"), ("knowledge_search", "search knowledge", "knowledge base"),
                ("analyze image", "analyze_image"), ("ocr",),
                ("generate_docx", "create report", "generate report", "word document"),
                ("generate_xlsx", "spreadsheet"), ("generate_pptx", "presentation", "powerpoint"),
                ("calculate",), ("summarize_numbers", "summarize numbers"), ("convert_units", "convert units"),
            )
            capabilities_requested = sum(any(phrase in normalized for phrase in group) for group in capability_groups)
            candidate = not investigation and (chained or named_tools > 1 or capabilities_requested > 1)
            return decision(WorkflowStrategy.planned if candidate else WorkflowStrategy.adaptive,
                WorkflowHandler.planned if candidate else WorkflowHandler.tools, rule,
                "Generate and validate a dependency plan before executing registered tools." if candidate else reason,
                domain="numeric" if numeric else "industrial_tools",
                capabilities=["numeric_tools"] if numeric else ["registered_tools"],
                planning_candidate=candidate)

        if tool_scope == "NUMERIC":
            return tool_decision("numeric_scope", "Use the restricted numeric catalogue with the bounded tool agent.", numeric=True)

        inspection = bool(attachments) and any(word in normalized for word in _INSPECTION_SIGNALS)
        if requested_chat_mode == ChatMode.automatic and not workcell_id and not inspection:
            integration = IntegrationRoutePlanner().plan(request, attachments, workspace_root)
            if integration.handles_request:
                capabilities = (["authorized_document_retrieval"] if integration.use_graph else [])
                if integration.diagnostic_domain:
                    capabilities.append("time_series_diagnostics")
                capabilities.append("controlplane_release")
                strategy = WorkflowStrategy.workflow if integration.diagnostic_domain else WorkflowStrategy.retrieval
                return decision(strategy, WorkflowHandler.integration,
                    "industrial_service_plan", integration.reason, integration=integration,
                    domain=integration.diagnostic_domain or "organizational_knowledge",
                    capabilities=capabilities, services=integration.services,
                    requires_validated_input=integration.missing_diagnostic_input)

        if chat.selected == ChatMode.controlled and (workcell_id or inspection):
            return decision(WorkflowStrategy.workflow, WorkflowHandler.workcell,
                "explicit_workcell" if workcell_id else "inspection_workcell",
                "Execute the selected Workcell through its existing validation and authorization checks.",
                domain="workcell" if workcell_id else "inspection", capabilities=["validated_workcell"])
        if (chat.selected == ChatMode.controlled and attachments
                and attachments[0].lower().endswith(".csv")
                and any(word in normalized for word in ("code", "python", "anomal"))):
            return decision(WorkflowStrategy.workflow, WorkflowHandler.coding,
                "csv_coding_workflow", "Use the existing bounded coding and sandbox verification workflow.",
                domain="data_analysis", capabilities=["code_generation", "sandbox_execution"])
        if chat.selected == ChatMode.controlled and requires_tool_agent(request, attachments):
            return tool_decision("registered_tool_request", "Select registered tools adaptively within the existing execution bounds.")
        if chat.selected == ChatMode.authorized or (
                chat.selected == ChatMode.controlled and extract_asset_references(request)):
            return decision(WorkflowStrategy.retrieval, WorkflowHandler.retrieval,
                "authorized_evidence", "Use authorized evidence and asset context before forming the response.",
                domain="industrial_assets" if extract_asset_references(request) else "organizational_knowledge",
                capabilities=["authorized_retrieval", "evidence_verification"])
        return decision(WorkflowStrategy.direct, WorkflowHandler.direct,
            "explicit_general" if requested_chat_mode == ChatMode.general else "conversational_response",
            "Answer within the selected chat mode without opening files or invoking tools.",
            capabilities=["answer_generation"])
