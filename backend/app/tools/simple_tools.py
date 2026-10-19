"""Small, deterministic tools for user-supplied numbers. No files or plant access."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import statistics
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.evidence.units import UnitService, UnitServiceError
from app.tools.base import Tool, ToolResult, ToolRisk


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Calculation(_Arguments):
    expression: str = Field(min_length=1, max_length=200)


class Numbers(_Arguments):
    values: list[float] = Field(min_length=1, max_length=256)


class Conversion(_Arguments):
    value: float = Field(ge=-1e12, le=1e12)
    from_unit: str = Field(min_length=1, max_length=40)
    to_unit: str = Field(min_length=1, max_length=40)


def _bounded(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Only finite real numbers are supported")
    result = float(value)
    if not math.isfinite(result) or abs(result) > 1e15:
        raise ValueError("Result exceeds the finite numeric limit (1e15)")
    return result


def calculate(expression: str) -> float:
    """Interpret only numeric AST nodes; never eval, compile, or execute input."""
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex (64 nodes maximum)")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}

    def visit(node):
        if isinstance(node, ast.Constant):
            return _bounded(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _bounded(visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1))
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12:
                    raise ValueError("Exponent magnitude must not exceed 12")
                return _bounded(left ** right)
            operation = operations.get(type(node.op))
            if operation:
                return _bounded(operation(left, right))
        raise ValueError("Use numeric literals, parentheses, + - * / // % or ** only")

    return visit(tree.body)


def _result(name: str, inputs: dict, output: dict, evidence: str) -> ToolResult:
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    return ToolResult(success=True, output={
        **output,
        "validation": "Bounded finite numeric result from user-supplied inputs; no plant data was read.",
        # Reuse the existing tool-evidence collector and final grounding policy.
        "results": [{"text": evidence, "score": 1.0, "source": {
            "file": f"Local tool: {name}", "source_id": f"{name}:{identity}",
            "kind": "local_computation", "basis": "user_supplied_inputs",
        }}],
    })


class CalculateTool(Tool):
    name = "calculate"
    description = "Evaluate bounded arithmetic on user-supplied numbers; no code execution"
    risk = ToolRisk.low
    input_schema = Calculation.model_json_schema()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            values = Calculation.model_validate(args)
            result = calculate(values.expression)
            return _result(self.name, args, {"expression": values.expression, "result": result},
                           f"Expression {values.expression} gives result {result:g} for user-supplied numbers.")
        except ValidationError:
            return ToolResult(success=False, error="Provide only an expression string of 1–200 characters.")
        except (ValueError, SyntaxError, ArithmeticError, TypeError):
            return ToolResult(success=False, error="Invalid or excessive arithmetic; use finite numbers and basic operators (exponent magnitude <=12, result <=1e15).")


class SummarizeNumbersTool(Tool):
    name = "summarize_numbers"
    description = "Compute count, total, minimum, maximum, mean and median of 1–256 supplied numbers"
    risk = ToolRisk.low
    input_schema = Numbers.model_json_schema()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            values = Numbers.model_validate(args).values
            if any(abs(value) > 1e12 for value in values):
                raise ValueError
            output = {"count": len(values), "total": _bounded(math.fsum(values)),
                      "minimum": min(values), "maximum": max(values),
                      "mean": _bounded(statistics.fmean(values)), "median": statistics.median(values)}
            return _result(self.name, args, output,
                           "Summary of user-supplied numbers: " + "; ".join(f"{key} = {value:g}" for key, value in output.items()) + ".")
        except (ValidationError, ValueError, ArithmeticError):
            return ToolResult(success=False, error="Provide only values: a list of 1–256 finite numbers with magnitude <=1e12.")


@lru_cache(maxsize=1)
def _units() -> UnitService:
    return UnitService()


class ConvertUnitsTool(Tool):
    name = "convert_units"
    description = "Convert a supplied engineering value using compatible units (e.g. kPa/bar, mm/m, degC/K, W/kW)"
    risk = ToolRisk.low
    input_schema = Conversion.model_json_schema()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            values = Conversion.model_validate(args)
            converted = _units().convert(values.value, values.from_unit, values.to_unit, precision=8)
            _bounded(converted.normalized_value)
            return _result(self.name, args, converted.model_dump(),
                           f"User-supplied {converted.original_value:g} {converted.original_unit} converts to {converted.normalized_value:g} {converted.normalized_unit}.")
        except (ValidationError, UnitServiceError, ValueError, ArithmeticError):
            return ToolResult(success=False, error="Provide a finite value (magnitude <=1e12) and supported, dimensionally compatible engineering units.")


SIMPLE_TOOL_NAMES = ("calculate", "summarize_numbers", "convert_units")
