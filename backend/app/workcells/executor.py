from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.workcells.handlers import WorkcellHandlerContext, WorkcellHandlerRegistry
from app.workcells.models import WorkcellDefinition, WorkcellExecutionState


WorkcellEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class WorkcellExecutionError(RuntimeError):
    pass


class WorkcellExecutor:
    def __init__(self, handlers: WorkcellHandlerRegistry) -> None:
        self.handlers = handlers

    @staticmethod
    def _ordered_steps(definition: WorkcellDefinition):
        steps = {step.id: step for step in definition.workflow.steps}
        ordered = []
        complete: set[str] = set()
        while len(ordered) < len(steps):
            ready = sorted(
                (step for step in steps.values() if step.id not in complete and set(step.depends_on) <= complete),
                key=lambda step: step.id,
            )
            if not ready:
                raise WorkcellExecutionError("WORKCELL_INVALID: workflow is cyclic or has missing dependencies")
            for step in ready:
                ordered.append(step)
                complete.add(step.id)
        return ordered

    async def execute(
        self,
        context: WorkcellHandlerContext,
        *,
        event_callback: WorkcellEventCallback | None = None,
    ) -> WorkcellExecutionState:
        definition = context.definition
        self._validate_input(context.inputs, definition.input_schema)
        state = WorkcellExecutionState(
            workcell_id=definition.manifest.id,
            workcell_version=definition.manifest.version,
            workcell_hash=definition.content_hash,
            workflow_version=definition.workflow.version,
        )

        async def emit(name: str, payload: dict[str, Any]) -> None:
            if event_callback:
                await event_callback(name, payload)

        await emit("workcell_loaded", {"workcell_id": state.workcell_id, "version": state.workcell_version, "hash": state.workcell_hash})
        for step in self._ordered_steps(definition):
            state.current_step = step.id
            if step.condition == "input_present" and not context.inputs.get(step.condition_input or ""):
                state.completed_steps.append(step.id)
                state.step_outputs[step.id] = {"skipped": True}
                continue
            resolved_inputs = {name: context.accumulated.get(reference, context.inputs.get(reference)) for name, reference in step.inputs.items()}
            state.step_inputs[step.id] = resolved_inputs
            await emit("workcell_step_started", {"step_id": step.id, "handler": step.handler})
            try:
                output = await self.handlers.get(step.handler)(context, step, resolved_inputs)
                if not isinstance(output, dict):
                    raise TypeError("Workcell handlers must return dictionaries")
                state.step_outputs[step.id] = output
                for name in step.outputs:
                    if name not in output:
                        raise WorkcellExecutionError(f"Handler {step.handler} omitted declared output {name}")
                    context.accumulated[name] = output[name]
                state.completed_steps.append(step.id)
                await emit("workcell_step_completed", {"step_id": step.id, "outputs": sorted(output)})
            except Exception as exc:
                state.failed_steps.append(step.id)
                await emit("workcell_step_failed", {"step_id": step.id, "error": str(exc)})
                if step.failure_behavior == "stop":
                    raise WorkcellExecutionError(f"WORKCELL_EXECUTION_FAILED at {step.id}: {exc}") from exc
        state.current_step = None
        await emit("workcell_completed", {"workcell_id": state.workcell_id, "completed_steps": state.completed_steps})
        return state

    @staticmethod
    def _validate_input(value: dict[str, Any], schema: dict[str, Any]) -> None:
        if schema.get("type") != "object":
            raise WorkcellExecutionError("WORKCELL_INPUT_INVALID: input schema must describe an object")
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise WorkcellExecutionError(f"WORKCELL_INPUT_INVALID: missing {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise WorkcellExecutionError(f"WORKCELL_INPUT_INVALID: unknown fields {sorted(unknown)}")
        types = {
            "string": str, "array": list, "object": dict,
            "integer": int, "number": (int, float), "boolean": bool,
        }
        for name, item in properties.items():
            if name not in value or not isinstance(item, dict) or "type" not in item:
                continue
            expected = types.get(item["type"])
            if expected and (not isinstance(value[name], expected) or (item["type"] in {"integer", "number"} and isinstance(value[name], bool))):
                raise WorkcellExecutionError(f"WORKCELL_INPUT_INVALID: {name} must be {item['type']}")
