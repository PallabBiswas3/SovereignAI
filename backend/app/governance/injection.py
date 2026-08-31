from __future__ import annotations

import re


class PromptInjectionScanner:
    PATTERNS = {
        "instruction_override": re.compile(r"ignore\s+(?:all\s+)?(?:(?:previous|prior)(?:\s+system)?|system)\s+instructions?", re.IGNORECASE),
        "policy_bypass": re.compile(r"(?:bypass|disable|override)\s+(?:the\s+)?(?:policy|safety|guardrails?)", re.IGNORECASE),
        "data_exfiltration": re.compile(r"(?:upload|send|exfiltrate)\s+(?:all\s+)?(?:files|secrets|documents|data)", re.IGNORECASE),
        "role_spoofing": re.compile(r"(?:system|developer)\s*(?:message|prompt)\s*:", re.IGNORECASE),
    }

    def scan(self, text: str) -> list[dict[str, object]]:
        findings: list[dict[str, object]] = []
        for kind, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({"kind": kind, "start": match.start(), "end": match.end(), "severity": "high"})
        return findings
