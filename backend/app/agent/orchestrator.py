from __future__ import annotations

from datetime import datetime, timezone

from app.agent.executor import AgentExecutor
from app.agent.planner import AgentPlanner
from app.agent.state import AgentRunState, RunStatus, StepStatus
from app.agent.checkpoints import checkpoint, execution_id, ensure_active
from app.core.config import Settings
from app.llm.factory import configured_local_provider
from app.llm.base import ModelGenerationCancelled
from app.orchestration.execution_mode import ExecutionMode, ExecutionModeSelector
from app.agent.executor import EventCallback
import asyncio
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.orchestration.chat_mode import ChatMode, system_prompt_for_mode


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings.models_config)
        self.router = ModelRouter(self.registry)
        self.planner = AgentPlanner()

    async def run(
        self,
        request: str,
        model_override: str | None = None,
        execution_mode: ExecutionMode = ExecutionMode.automatic,
        attachment_count: int = 0,
        event_callback: EventCallback | None = None,
        cancellation_event: asyncio.Event | None = None,
        *,
        requested_chat_mode: ChatMode = ChatMode.automatic,
        chat_mode: ChatMode = ChatMode.general,
        chat_mode_reason: str = "No organizational evidence or governed action was requested.",
        generation_prompt: str | None = None,
        resume_state: AgentRunState | None = None,
    ) -> AgentRunState:
        routing = resume_state.routing if resume_state else self.router.route(request, model_override)
        selection = ExecutionModeSelector().select(
            execution_mode, request, routing.task_profile, attachment_count
        )
        plan = self.planner.create_plan(request, routing.task_profile, selection.selected)
        state = resume_state or AgentRunState(
            id=execution_id(), request=request, status=RunStatus.running, plan=plan, routing=routing,
            requested_execution_mode=selection.requested.value,
            execution_mode=selection.selected.value,
            execution_mode_reason=selection.reason,
            requested_chat_mode=requested_chat_mode.value,
            chat_mode=chat_mode.value,
            chat_mode_reason=chat_mode_reason,
        )
        await checkpoint(state, "agent_plan_created")
        selected = self.registry.get(routing.model_id)
        runtime = configured_local_provider(
            self.settings,
            model_definition=selected,
            execution_mode=selection.selected.value,
            priority=selection.priority,
        )
        executor = AgentExecutor(
            runtime.provider, runtime.model, event_callback, cancellation_event,
            system_prompt=system_prompt_for_mode(chat_mode),
            generation_prompt=generation_prompt,
        )
        try:
            for step in state.plan.steps:
                if resume_state and step.status == StepStatus.completed:
                    continue
                ensure_active()
                step.status = StepStatus.running
                step.error = None
                await checkpoint(state, "agent_step_started")
                step.observation = await executor.execute_step(state, step)
                step.status = StepStatus.completed
                state.updated_at = datetime.now(timezone.utc)
                await checkpoint(state, "agent_step_completed")
            state.status = RunStatus.completed
        except ModelGenerationCancelled:
            raise
        except Exception as exc:
            step.status = StepStatus.failed
            step.error = str(exc)
            state.status = RunStatus.failed
            state.warnings.append(f"Agent stopped safely: {exc}")
        state.updated_at = datetime.now(timezone.utc)
        return state
