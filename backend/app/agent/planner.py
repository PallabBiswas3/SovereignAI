from __future__ import annotations

from app.agent.state import AgentPlan, AgentStep
from app.router.schemas import TaskProfile
from app.orchestration.execution_mode import ExecutionMode


class AgentPlanner:
    """Creates transparent operational steps; it never records private model reasoning."""

    def create_plan(
        self, request: str, profile: TaskProfile, mode: ExecutionMode = ExecutionMode.standard
    ) -> AgentPlan:
        if mode == ExecutionMode.fast:
            return AgentPlan(
                goal=request,
                steps=[AgentStep(id=1, action="generate_response", title="Generate fast local response")],
            )
        steps = [AgentStep(id=1, action="understand_task", title="Classify and validate request")]
        if profile.vision_requirement > 0.35:
            steps.append(AgentStep(id=len(steps) + 1, action="analyze_input", title="Inspect visual input"))
        elif profile.document_requirement > 0.35:
            steps.append(AgentStep(id=len(steps) + 1, action="analyze_input", title="Inspect referenced documents"))
        if profile.coding_requirement > 0.35:
            steps.append(AgentStep(id=len(steps) + 1, action="prepare_code", title="Prepare controlled code task"))
        steps.extend(
            [
                AgentStep(id=len(steps) + 1, action="generate_response", title="Generate local response"),
                AgentStep(id=len(steps) + 2, action="verify_response", title="Verify completion and provenance"),
            ]
        )
        if mode == ExecutionMode.deep:
            steps.insert(
                len(steps) - 1,
                AgentStep(
                    id=0,
                    action="review_response",
                    title="Review response for industrial completeness",
                ),
            )
            for index, step in enumerate(steps, start=1):
                step.id = index
        return AgentPlan(goal=request, steps=steps)
