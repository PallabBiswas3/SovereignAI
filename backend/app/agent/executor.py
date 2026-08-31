from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.agent.state import AgentRunState, AgentStep
from app.llm.base import LocalModelProvider


PersistCallback = Callable[[AgentRunState], Awaitable[None]]


class AgentExecutor:
    def __init__(self, provider: LocalModelProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def execute_step(self, run: AgentRunState, step: AgentStep) -> str:
        if step.action == "understand_task":
            return f"Request classified as {run.routing.task_profile.task_type}."
        if step.action == "analyze_input":
            return "Input-analysis capability identified; attached files are handled by registered tools."
        if step.action == "prepare_code":
            return "Code task identified; execution requires the isolated sandbox tool."
        if step.action == "generate_response":
            result = await self.provider.generate(
                run.request,
                self.model,
                "You are a local industrial assistant. Do not invent evidence or claim unavailable tools ran.",
            )
            run.final_response = result.text
            if result.fallback:
                run.warnings.append("Local model unavailable; no synthetic model answer was generated.")
            return f"Response produced by {result.provider}."
        if step.action == "verify_response":
            if not run.final_response:
                raise RuntimeError("No response exists to verify")
            step.verification = "A non-empty response exists; source verification is required for document claims."
            return "Response completeness verified."
        raise ValueError(f"Unsupported agent action: {step.action}")

