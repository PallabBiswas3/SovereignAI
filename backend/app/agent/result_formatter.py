"""Render completed factual tool receipts without inventing interpretation."""
from __future__ import annotations

import math
import re
from typing import Any


_INTERPRETATION = re.compile(
    r"\b(explain\w*|interpret\w*|recommend\w*|assess\w*|diagnos\w*|investigat\w*|why|"
    r"compar\w*|analy[sz]\w*|reason\w*|justify|translate|summari[sz]e(?! numbers)|"
    r"how|which|whether|should|safe\w*|risk\w*|compliance|acceptab\w*|conclu\w*|"
    r"suggest\w*|evaluat\w*|decid\w*|cite|citation\w*|sources|proof|verify|verification)\b|\?|"
    r"root cause|in detail|step.by.step", re.I)
_ARTIFACTS = {"generate_docx", "generate_xlsx", "generate_pptx"}


def verified_result(goal: str, observations: list[dict[str, Any]]) -> str | None:
    if not observations or _INTERPRETATION.search(goal):
        return None
    artifacts = any(item.get("tool") in _ARTIFACTS for item in observations)
    lines = []
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Expected a finite observed number")
        return str(value)
    try:
        for item in observations:
            if item.get("success") is not True or item.get("error"):
                return None
            name, output = item["tool"], item.get("output")
            if name == "calculate":
                lines.append(f"calculate: {output['expression']} = {number(output['result'])}.")
            elif name == "summarize_numbers":
                values = "; ".join(f"{key} = {number(output[key])}" for key in ("count", "total", "minimum", "maximum", "mean", "median"))
                lines.append(f"summarize_numbers: {values}.")
            elif name == "convert_units":
                lines.append(f"convert_units: {number(output['original_value'])} {output['original_unit']} = {number(output['normalized_value'])} {output['normalized_unit']}.")
            elif name == "list_files" and isinstance(output, list) and all(isinstance(path, str) for path in output):
                lines.append(f"Workspace files ({len(output)}):\n" + "\n".join(output[:50]))
                if len(output) > 50:
                    lines.append("Only the first 50 paths are shown.")
            elif name == "get_file_metadata":
                lines.append(f"File: {output['name']}; size: {number(output['size'])} bytes; extension: {output['extension']}.")
            elif name == "read_file" and artifacts:
                lines.append(f"Read workspace file: {output['path']}.")
            elif name in _ARTIFACTS:
                artifact = output["artifact"]
                if not re.fullmatch(r"/api/artifacts/[A-Za-z0-9-]+", artifact["url"]):
                    return None
                lines.append(f"Created artifact: {artifact['name']} ({artifact['url']}).")
            else:
                return None
        response = "\n".join(lines)
        return response if len(response) <= 8_000 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
