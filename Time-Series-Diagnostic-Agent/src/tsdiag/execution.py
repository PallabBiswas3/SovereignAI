from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable

from .models import ToolTraceStep, json_safe


class ExecutionTrace(list[ToolTraceStep]):
    """Chronological hierarchical trace with stable named lookup.

    Top-level iteration preserves workflow order. ``flatten()`` walks child steps
    depth-first so callers can query both workflow and nested legacy/tool steps
    without maintaining a second trace in result metadata.
    """

    def __init__(self, steps: Iterable[ToolTraceStep] = ()):
        super().__init__(steps)

    @property
    def steps(self) -> list[ToolTraceStep]:
        return self

    def flatten(self) -> list[ToolTraceStep]:
        rows: list[ToolTraceStep] = []

        def visit(step: ToolTraceStep) -> None:
            rows.append(step)
            for child in step.children:
                visit(child)

        for step in self:
            visit(step)
        return rows

    def get(self, step_id: str, *, occurrence: int = -1, recursive: bool = True) -> ToolTraceStep | None:
        source = self.flatten() if recursive else list(self)
        matches = [step for step in source if step.tool == step_id]
        if not matches:
            return None
        try:
            return matches[occurrence]
        except IndexError:
            return None

    def require(self, step_id: str, *, occurrence: int = -1, recursive: bool = True) -> ToolTraceStep:
        step = self.get(step_id, occurrence=occurrence, recursive=recursive)
        if step is None:
            raise KeyError(f"Required trace step {step_id!r} is absent")
        return step


class StepExecutionError(RuntimeError):
    def __init__(self, step: str, message: str, trace: Iterable[ToolTraceStep]):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.trace = ExecutionTrace(trace)


@dataclass(frozen=True)
class InputField:
    name: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class DomainInputSchema:
    domain: str
    fields: tuple[InputField, ...]

    def validate(self, values: dict[str, Any]) -> None:
        missing = [row.name for row in self.fields if row.required and values.get(row.name) is None]
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "fields": [json_safe(row.__dict__) for row in self.fields]}


@dataclass(frozen=True)
class Step:
    id: str
    execute: Callable[[dict[str, Any]], dict[str, Any]]
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("step id must be non-empty")
        if not callable(self.execute):
            raise TypeError(f"step {self.id!r} execute must be callable")


@dataclass(frozen=True)
class Workflow:
    steps: tuple[Step, ...]
    version: str = "1.0"

    def __post_init__(self) -> None:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("workflow step ids must be unique")
        known: set[str] = set()
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(
                    f"step {step.id!r} depends on steps not declared earlier: {sorted(missing)}"
                )
            known.add(step.id)

    def get(self, step_id: str) -> Step | None:
        return next((step for step in self.steps if step.id == step_id), None)

    def require(self, step_id: str) -> Step:
        step = self.get(step_id)
        if step is None:
            raise KeyError(f"Required workflow step {step_id!r} is absent")
        return step


@dataclass
class DomainToolRegistry:
    implementations: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = field(default_factory=dict)

    def register(self, domain: str, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if not callable(fn):
            raise TypeError(f"Implementation for {domain}.{name} must be callable")
        self.implementations[(domain, name)] = fn

    def get(self, domain: str, name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        try:
            return self.implementations[(domain, name)]
        except KeyError as exc:
            raise KeyError(f"No registered implementation for {domain}.{name}") from exc

    def names(self, domain: str) -> tuple[str, ...]:
        return tuple(name for pack, name in self.implementations if pack == domain)


class StepExecutor:
    """Execute registered tools with isolated errors and auditable timing."""

    def __init__(self, domain: str, registry: DomainToolRegistry):
        self.domain = domain
        self.registry = registry
        self.trace = ExecutionTrace()

    def run(self, name: str, state: dict[str, Any], *, optional: bool = False) -> dict[str, Any]:
        started = perf_counter()
        input_keys = sorted(state.keys())
        try:
            output = self.registry.get(self.domain, name)(state)
            if not isinstance(output, dict):
                raise TypeError("tool must return a dictionary")
            duration = perf_counter() - started
            state.update(output)
            self.trace.append(ToolTraceStep(
                tool=name, status="ok", duration_seconds=duration,
                inputs_summary={"available_keys": input_keys},
                outputs_summary={"output_keys": sorted(output.keys())},
            ))
            return output
        except Exception as exc:
            trace = ToolTraceStep(
                tool=name, status="skipped" if optional else "error",
                duration_seconds=perf_counter() - started,
                details={"error_type": type(exc).__name__, "message": str(exc)},
            )
            self.trace.append(trace)
            if optional:
                return {}
            raise StepExecutionError(name, str(exc), self.trace) from exc


class WorkflowExecutor:
    """Run a declarative workflow while preserving chronological named trace."""

    def run(self, workflow: Workflow, initial_state: dict[str, Any]) -> tuple[dict[str, Any], ExecutionTrace]:
        state = dict(initial_state)
        trace = ExecutionTrace()
        completed: set[str] = set()
        for step in workflow.steps:
            missing = set(step.depends_on) - completed
            if missing:
                raise StepExecutionError(step.id, f"dependencies not completed: {sorted(missing)}", trace)
            started = perf_counter()
            input_keys = sorted(state.keys())
            try:
                output = step.execute(state)
                if not isinstance(output, dict):
                    raise TypeError("workflow step must return a dictionary")
                state.update(output)
                trace.append(ToolTraceStep(
                    tool=step.id,
                    status="ok",
                    duration_seconds=perf_counter() - started,
                    inputs_summary={"available_keys": input_keys, "step_version": step.version},
                    outputs_summary={"output_keys": sorted(output.keys())},
                ))
                completed.add(step.id)
            except Exception as exc:
                trace.append(ToolTraceStep(
                    tool=step.id,
                    status="skipped" if step.optional else "error",
                    duration_seconds=perf_counter() - started,
                    inputs_summary={"available_keys": input_keys, "step_version": step.version},
                    details={"error_type": type(exc).__name__, "message": str(exc)},
                ))
                if step.optional:
                    completed.add(step.id)
                    continue
                raise StepExecutionError(step.id, str(exc), trace) from exc
        return state, trace
