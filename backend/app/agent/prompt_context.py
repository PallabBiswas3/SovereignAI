"""Compact model context; authoritative schemas and full results stay in the app."""
from __future__ import annotations

import json
import re
from typing import Any

from app.tools.registry import ToolRegistry


OUTPUTS = {
    "calculate": "expression,result", "summarize_numbers": "count,total,minimum,maximum,mean,median",
    "convert_units": "original_value,original_unit,normalized_value,normalized_unit",
    "read_file": "path,text", "list_files": "array of paths", "get_file_metadata": "name,extension,size,modified",
    "knowledge_search": "results:[{text,source}]", "ocr_document": "text",
    "generate_docx": "artifact:{id,name,url}", "generate_xlsx": "artifact:{id,name,url}",
    "generate_pptx": "artifact:{id,name,url}",
}


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def plan_catalogue(goal: str, tools: ToolRegistry, attachments: list[str]) -> ToolRegistry:
    """Shortlist only explicitly named tools; unnamed requests retain the catalogue.

    The model can defer to the full adaptive agent if the shortlist is insufficient.
    Execution still validates against the authoritative, scoped registry.
    """
    names = {str(tool["name"]) for tool in tools.discover()}
    explicit = {name for name in names if re.search(rf"\b{re.escape(name)}\b", goal, re.I)}
    if not explicit or attachments:
        return tools
    aliases = {
        "read_file": ("read file",), "list_files": ("list file",), "search_files": ("search file",),
        "knowledge_search": ("knowledge base", "search knowledge"), "ocr_document": ("ocr",),
        "analyze_image": ("analyze image",), "generate_docx": ("create report", "word report"),
        "generate_xlsx": ("spreadsheet",), "generate_pptx": ("presentation", "powerpoint"),
        "calculate": ("calculate",), "summarize_numbers": ("summarize numbers",),
        "convert_units": ("convert units",),
    }
    normalized = goal.lower()
    explicit.update(name for name, phrases in aliases.items() if any(phrase in normalized for phrase in phrases))
    selected = ToolRegistry()
    for name in sorted(explicit & names):
        selected.register(tools.get(name))
    return selected


def catalogue(tools: ToolRegistry) -> str:
    def shape(schema):
        if "anyOf" in schema:
            return {"anyOf": [shape(option) for option in schema["anyOf"]]}
        kind = schema.get("type", "json")
        if kind == "array":
            return [shape(schema.get("items", {}))]
        if kind == "object":
            return {key: shape(value) for key, value in schema.get("properties", {}).items()}
        return kind
    return dumps([{"tool": tool["name"], "purpose": str(tool["description"])[:110],
        "args": {key: shape(value) for key, value in tool["input_schema"].get("properties", {}).items() if key != "run_id"},
        "required": tool["input_schema"].get("required", []),
        **({"output": OUTPUTS[tool["name"]]} if tool["name"] in OUTPUTS else {})}
        for tool in tools.discover()])


def compact_observations(observations: list[dict[str, Any]]) -> str:
    """Remove redundant computation evidence; preserve retrieval text and citations."""
    compact = []
    for observation in observations[-6:]:
        item = dict(observation)
        output = item.get("output")
        if item.get("tool") in {"calculate", "summarize_numbers", "convert_units"} and isinstance(output, dict):
            item["output"] = {key: value for key, value in output.items() if key not in {"validation", "results"}}
            citations = [{key: source[key] for key in ("source_id", "file", "page", "section", "kind", "basis") if key in source}
                         for result in output.get("results", []) if isinstance(result, dict)
                         and isinstance(source := result.get("source"), dict)]
            if citations:
                item["output"]["sources"] = citations
        compact.append(item)
    return json.dumps(compact, ensure_ascii=False, default=str)[:30_000]
