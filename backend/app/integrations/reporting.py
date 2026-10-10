"""Bounded factual-report generation; shape validation is not factual verification."""
from __future__ import annotations

import re
from typing import Any


def reporting_limit(query: str) -> int | None:
    """Opt in only for explicit fact-only requests, not ordinary assessments."""
    if not re.search(r"\b(?:report|state|list|provide|return)\s+only\b|\bonly\s+(?:one|two|three|four|five|six|[1-6])\s+facts?\b", query, re.IGNORECASE):
        return None
    counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\b(one|two|three|four|five|six|[1-6])\s+(?:facts?|sentences?)(?!\s+(?:for|per|each)\b)", query, re.IGNORECASE)
    if match:
        count = match.group(1).lower()
        return counts[count] if count in counts else int(count)
    return 6


def reporting_schema(limit: int) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["facts"],
            "properties": {"facts": {"type": "array", "minItems": 1, "maxItems": limit,
                "items": {"type": "object", "additionalProperties": False,
                    "required": ["statement", "citations"], "properties": {
                        "statement": {"type": "string", "minLength": 1, "maxLength": 350},
                        "citations": {"type": "array", "minItems": 1, "maxItems": 6,
                                      "uniqueItems": True, "items": {"type": "string"}},
                    }}}}}


def render_reported_facts(data: Any, limit: int, allowed_ids: set[str]) -> str:
    """Reject overflow/malformed output; never silently truncate a generated answer."""
    if not isinstance(data, dict) or set(data) != {"facts"}:
        raise ValueError("factual report must contain only facts")
    facts = data["facts"]
    if not isinstance(facts, list) or not 1 <= len(facts) <= limit:
        raise ValueError("factual report exceeds the requested fact count or contains no facts")
    lines = []
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != {"statement", "citations"}:
            raise ValueError("invalid factual report item")
        text, ids = fact["statement"], fact["citations"]
        if not isinstance(text, str) or not text.strip() or len(text) > 350:
            raise ValueError("invalid factual statement")
        text = text.strip()
        if re.search(r"[.!?]\s+\S|[\n\r;\[\]]", text):
            raise ValueError("a factual report item must contain one sentence and no embedded citations")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 6
                or any(not isinstance(item, str) or item not in allowed_ids for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("factual report must cite supplied evidence identifiers")
        lines.append(text.rstrip(".!?") + " " + " ".join(f"[{item}]" for item in ids) + ".")
    return "\n".join(lines)
