"""Bounded plan data, DAG validation and data-only output bindings (no eval)."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolRegistry
from app.agent.state import RequiredTaskInput


_PATH = r"[A-Za-z][A-Za-z0-9_]{0,31}(?:\.[A-Za-z0-9_]+)*"
_TEXT_REF = re.compile(r"\{\{(" + _PATH + r")\}\}|\{\s*[\"']?\$ref[\"']?\s*:\s*[\"'](" + _PATH + r")[\"']\s*\}")


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
    tool: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any]
    depends_on: list[str] = Field(max_length=6)


class StructuredPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["plan", "adaptive", "clarify"]
    reason_summary: str = Field(min_length=1, max_length=500)
    steps: list[PlanStep] = Field(max_length=6)
    questions: list[RequiredTaskInput] = Field(default_factory=list, max_length=3)


def bindings(value: Any, depth: int = 0) -> set[str]:
    if depth > 16:
        raise ValueError("Plan arguments exceed the nesting limit")
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                raise ValueError("A reference must contain only a string $ref")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}(?:\.[A-Za-z0-9_]+)*", value["$ref"]):
                raise ValueError("Invalid output reference")
            return {value["$ref"].split(".")[0]}
        if "$concat" in value:
            if set(value) != {"$concat"} or not isinstance(value["$concat"], list):
                raise ValueError("$concat requires a list of text and scalar references")
            return bindings(value["$concat"], depth + 1)
        if any(key.startswith("$") for key in value):
            raise ValueError("Unknown binding operator")
        return set().union(*(bindings(item, depth + 1) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(bindings(item, depth + 1) for item in value))
    if isinstance(value, str):
        matches = list(_TEXT_REF.finditer(value))
        remaining = _TEXT_REF.sub("", value)
        if "{{" in remaining or "$ref" in remaining:
            raise ValueError("Malformed text reference; use {{step.field}} or a $ref object")
        return {(match.group(1) or match.group(2)).split(".")[0] for match in matches}
    return set()


def generation_schema(tools: ToolRegistry) -> dict[str, Any]:
    """Constrain generated arguments per tool, including typed binding operators."""
    reference = {"type": "object", "properties": {"$ref": {"type": "string"}},
                 "required": ["$ref"], "additionalProperties": False}
    concat = {"type": "object", "properties": {"$concat": {"type": "array", "items": {
        "anyOf": [{"type": "string"}, {"type": "number"}, reference]}}},
        "required": ["$concat"], "additionalProperties": False}

    def bindable(schema):
        literal = {key: value for key, value in schema.items() if key not in {"title", "description", "default"}}
        if "properties" in literal:
            literal["properties"] = {key: bindable(value) for key, value in literal["properties"].items() if key != "run_id"}
            literal["additionalProperties"] = False
        if "items" in literal:
            literal["items"] = bindable(literal["items"])
        if "anyOf" in literal:
            literal["anyOf"] = [bindable(option) for option in literal["anyOf"]]
        options = [literal, reference]
        if literal.get("type") == "string" or "anyOf" in literal:
            options.append(concat)
        return {"anyOf": options}

    schema = StructuredPlan.model_json_schema()
    base = schema["$defs"]["PlanStep"]
    variants = []
    for tool in tools.discover():
        arguments = tool["input_schema"]
        properties = {key: bindable(value) for key, value in arguments.get("properties", {}).items() if key != "run_id"}
        variants.append({**base, "properties": {**base["properties"],
            "tool": {"type": "string", "enum": [tool["name"]]},
            "arguments": {"type": "object", "properties": properties,
                "required": arguments.get("required", []), "additionalProperties": False}}})
    schema["$defs"]["PlanStep"] = {"oneOf": variants}
    return schema


def resolve(value: Any, outputs: dict[str, Any]) -> Any:
    """Bindings traverse JSON containers only; returned data is never reinterpreted."""
    if isinstance(value, dict):
        if "$ref" in value:
            parts = value["$ref"].split(".")
            try:
                found = outputs[parts[0]]
                for part in parts[1:]:
                    found = found[int(part)] if isinstance(found, list) and part.isdigit() else found[part]
                return found
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"Output reference is unavailable: {value['$ref']}") from exc
        if "$concat" in value:
            items = [resolve(item, outputs) for item in value["$concat"]]
            if any(not isinstance(item, (str, int, float, bool)) for item in items):
                raise ValueError("$concat can join scalar values only")
            return "".join(str(item) for item in items)
        return {key: resolve(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, outputs) for item in value]
    if isinstance(value, str):
        def replace(match):
            observed = resolve({"$ref": match.group(1) or match.group(2)}, outputs)
            if not isinstance(observed, (str, int, float, bool)):
                raise ValueError("Text references require observed scalar values")
            return str(observed)
        return _TEXT_REF.sub(replace, value)
    return value


def validate_value(value: Any, schema: dict, *, deferred: bool = False) -> None:
    """Validate the inline schema subset used by the registered tool catalogue.

    References defer field types until resolution; literal siblings are checked now.
    Unknown schema features fail closed rather than being silently ignored.
    """
    if deferred and isinstance(value, dict) and ("$ref" in value or "$concat" in value):
        return
    supported = {"type", "properties", "required", "additionalProperties", "items", "enum",
                 "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength",
                 "maxLength", "minItems", "maxItems", "title", "description", "default", "anyOf"}
    if set(schema) - supported:
        raise ValueError("Unsupported tool input-schema feature")
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_value(value, option, deferred=deferred)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Tool argument does not match any supported input shape")
    kind = schema.get("type")
    valid = {"string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool),
             "number": isinstance(value, (int, float)) and not isinstance(value, bool),
             "array": isinstance(value, list), "object": isinstance(value, dict),
             "boolean": isinstance(value, bool), "null": value is None}
    if kind and not valid.get(kind, False):
        raise ValueError(f"Tool argument must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Tool argument is outside its allowed values")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Tool arguments must be finite")
        for key, invalid in (("minimum", value < schema.get("minimum", -math.inf)),
                             ("maximum", value > schema.get("maximum", math.inf)),
                             ("exclusiveMinimum", value <= schema.get("exclusiveMinimum", -math.inf)),
                             ("exclusiveMaximum", value >= schema.get("exclusiveMaximum", math.inf))):
            if key in schema and invalid:
                raise ValueError(f"Tool argument violates {key}")
    if isinstance(value, (str, list)):
        lower, upper = ("minLength", "maxLength") if isinstance(value, str) else ("minItems", "maxItems")
        if len(value) < schema.get(lower, 0) or len(value) > schema.get(upper, 200_000):
            raise ValueError("Tool argument length is outside its bounds")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            raise ValueError("Missing required tool argument")
        if "properties" in schema and schema.get("additionalProperties", False) is not True and any(key not in properties for key in value):
            raise ValueError("Unknown tool argument")
        for key, item in value.items():
            validate_value(item, properties.get(key, {}), deferred=deferred)
    if isinstance(value, list):
        for item in value:
            validate_value(item, schema.get("items", {}), deferred=deferred)


def validate_plan(plan: StructuredPlan, tools: ToolRegistry, *, completed: set[str] | None = None,
                  remaining_calls: int = 6, required_tools: set[str] | None = None) -> list[PlanStep]:
    completed = completed or set()
    if plan.action != "plan" or not plan.steps or len(plan.steps) > remaining_calls:
        raise ValueError("A plan must contain steps within the remaining tool-call budget")
    encoded = json.dumps(plan.model_dump(), allow_nan=False)
    if len(encoded) > 64_000:
        raise ValueError("Plan exceeds the argument-size budget")
    steps = {step.id: step for step in plan.steps}
    if len(steps) != len(plan.steps) or set(steps) & completed:
        raise ValueError("Plan step IDs must be unique and must not replace completed steps")
    if (required_tools or set()) - {step.tool for step in plan.steps}:
        raise ValueError("Plan omits an explicitly requested tool")
    for step in plan.steps:
        tool = tools.get(step.tool)
        if "run_id" in step.arguments:
            raise ValueError("Artifact run identity is assigned by the application")
        refs = bindings(step.arguments)
        dependencies = set(step.depends_on)
        if refs - dependencies:
            raise ValueError("Every output reference must declare its dependency")
        if dependencies - (set(steps) | completed) or step.id in dependencies:
            raise ValueError("Plan contains an unknown or self dependency")
        validate_value(step.arguments, tool.input_schema, deferred=True)
    ordered, ready = [], set(completed)
    while len(ordered) < len(steps):
        batch = [step for step in plan.steps if step.id not in ready and set(step.depends_on) <= ready]
        if not batch:
            raise ValueError("Plan dependency graph contains a cycle")
        ordered.extend(batch)
        ready.update(step.id for step in batch)
    return ordered
