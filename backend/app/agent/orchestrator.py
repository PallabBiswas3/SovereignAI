from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.agent.executor import AgentExecutor
from app.agent.planner import AgentPlanner
from app.agent.state import AgentRunState, RunStatus, StepStatus
from app.core.config import Settings
from app.llm.ollama_provider import OllamaProvider
from app.llm.base import ModelGenerationCancelled
from app.orchestration.execution_mode import ExecutionMode, ExecutionModeSelector
from app.agent.executor import EventCallback
import asyncio
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter


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
    ) -> AgentRunState:
        routing = self.router.route(request, model_override)
        selection = ExecutionModeSelector().select(
            execution_mode, request, routing.task_profile, attachment_count
        )
        plan = self.planner.create_plan(request, routing.task_profile, selection.selected)
        state = AgentRunState(
            id=str(uuid4()), request=request, status=RunStatus.running, plan=plan, routing=routing,
            requested_execution_mode=selection.requested.value,
            execution_mode=selection.selected.value,
            execution_mode_reason=selection.reason,
        )
        selected = self.registry.get(routing.model_id)
        provider = OllamaProvider(
            selected.endpoint,
            self.settings.allow_deterministic_fallback,
            role=selected.role,
            memory_requirement=selected.memory_requirement,
            execution_mode=selection.selected.value,
            priority=selection.priority,
        )
        executor = AgentExecutor(provider, selected.model_tag, event_callback, cancellation_event)
        try:
            for step in state.plan.steps:
                step.status = StepStatus.running
                step.observation = await executor.execute_step(state, step)
                step.status = StepStatus.completed
                state.updated_at = datetime.now(timezone.utc)
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
