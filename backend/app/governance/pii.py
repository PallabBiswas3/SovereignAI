from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    masked: str


class PIIDetector:
    PATTERNS = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"),
        "aadhaar_like": re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
        "pan_like": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "account_identifier": re.compile(r"\b(?:account|a/c)\s*(?:no\.?|number)?\s*[:#-]?\s*\d{8,18}\b", re.IGNORECASE),
        "address_like": re.compile(r"\b\d{1,5}\s+[A-Za-z][A-Za-z .'-]{2,40}\s+(?:road|rd|street|st|lane|ln|avenue|ave)\b", re.IGNORECASE),
    }

    def detect(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for kind, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                matches.append(PIIMatch(kind, match.start(), match.end(), self._mask(value)))
        return sorted(matches, key=lambda item: item.start)

    @staticmethod
    def _mask(value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

