from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.workcells.handlers import WorkcellHandlerContext, WorkcellHandlerRegistry
from app.workcells.loader import WorkcellLoader
from app.workcells.registry import WorkcellRegistry
from app.workcells.validator import WorkcellValidator


async def validate_pump_inspection_inputs(context: WorkcellHandlerContext, step, inputs) -> dict[str, Any]:
    attachments = context.inputs.get("attachments", [])
    if not attachments:
        raise ValueError("WORKCELL_INPUT_INVALID: Pump Inspection requires at least one attachment")
    return {"validated_inputs": {"attachment_count": len(attachments)}}


async def execute_existing_pump_inspection(context: WorkcellHandlerContext, step, inputs) -> dict[str, Any]:
    runner = context.services.get("pump_inspection_runner")
    if runner is None:
        raise RuntimeError("Trusted Pump Inspection service is unavailable")
    state = await runner()
    return {"task_state": state}


async def dispatch_registered_artifact_handler(context: WorkcellHandlerContext, step, inputs) -> dict[str, Any]:
    dispatcher = context.services.get("artifact_dispatcher")
    if dispatcher is None:
        raise RuntimeError("Artifact handler may only run through the registered ArtifactService adapter")
    return await dispatcher(step.handler, inputs)


def create_workcell_handler_registry() -> WorkcellHandlerRegistry:
    registry = WorkcellHandlerRegistry()
    registry.register("validate_pump_inspection_inputs", validate_pump_inspection_inputs)
    registry.register("execute_existing_pump_inspection", execute_existing_pump_inspection)
    registry.register("generate_docx", dispatch_registered_artifact_handler)
    registry.register("generate_xlsx", dispatch_registered_artifact_handler)
    registry.register("generate_pptx", dispatch_registered_artifact_handler)
    return registry


def configured_workcell_registry(settings: Settings) -> WorkcellRegistry:
    handlers = create_workcell_handler_registry()
    loader = WorkcellLoader(settings.workcells_root)
    validator = WorkcellValidator(
        handlers,
        settings.tools_config,
        unsigned_workcells_allowed=settings.unsigned_workcells_allowed,
    )
    registry = WorkcellRegistry(settings.workcells_root, loader, validator)
    registry.discover()
    return registry
