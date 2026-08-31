from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.agent.executor import AgentExecutor
from app.agent.planner import AgentPlanner
from app.agent.state import AgentRunState, RunStatus, StepStatus
from app.core.config import Settings
from app.llm.ollama_provider import OllamaProvider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings.models_config)
        self.router = ModelRouter(self.registry)
        self.planner = AgentPlanner()

    async def run(self, request: str, model_override: str | None = None) -> AgentRunState:
        routing = self.router.route(request, model_override)
        plan = self.planner.create_plan(request, routing.task_profile)
        state = AgentRunState(
            id=str(uuid4()), request=request, status=RunStatus.running, plan=plan, routing=routing
        )
        selected = self.registry.get(routing.model_id)
        provider = OllamaProvider(selected.endpoint, self.settings.allow_deterministic_fallback)
        executor = AgentExecutor(provider, selected.model_tag)
        try:
            for step in state.plan.steps:
                step.status = StepStatus.running
                step.observation = await executor.execute_step(state, step)
                step.status = StepStatus.completed
                state.updated_at = datetime.now(timezone.utc)
            state.status = RunStatus.completed
        except Exception as exc:
            step.status = StepStatus.failed
            step.error = str(exc)
            state.status = RunStatus.failed
            state.warnings.append(f"Agent stopped safely: {exc}")
        state.updated_at = datetime.now(timezone.utc)
        return state
