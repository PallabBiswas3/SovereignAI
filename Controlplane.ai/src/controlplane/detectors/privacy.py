from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Pattern

from controlplane.detectors.base import Detector
from controlplane.schema import (
    DetectorResult,
    Finding,
    FindingStatus,
    Interaction,
    RiskCategory,
    Severity,
)


@dataclass(frozen=True)
class PrivacyPattern:
    subtype: str
    pattern: Pattern[str]
    severity: Severity
    confidence: float
    replacement: str


_PATTERNS = [
    PrivacyPattern(
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        Severity.CRITICAL,
        0.99,
        "[REDACTED PRIVATE KEY]",
    ),
    PrivacyPattern(
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.I),
        Severity.CRITICAL,
        0.98,
        "Bearer [REDACTED TOKEN]",
    ),
    PrivacyPattern(
        "api_key",
        re.compile(
            r"\b(?:"
            r"sk-[A-Za-z0-9_-]{16,}|"
            r"(?:sk|rk)_(?:test|live)_[A-Za-z0-9_-]{8,}|"
            r"AKIA[0-9A-Z]{16}|"
            r"(?:api[_ -]?key|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./+-]{12,}"
            r")",
            re.I,
        ),
        Severity.CRITICAL,
        0.96,
        "[REDACTED API KEY]",
    ),
    PrivacyPattern(
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
        Severity.MEDIUM,
        0.98,
        "[REDACTED EMAIL]",
    ),
    PrivacyPattern(
        "us_ssn",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        Severity.HIGH,
        0.99,
        "[REDACTED SSN]",
    ),
    PrivacyPattern(
        "aadhaar_number",
        re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)"),
        Severity.HIGH,
        0.94,
        "[REDACTED ID]",
    ),
]

_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\w)\+?(?:\d[\s().-]?){9,14}\d(?!\w)")


class PrivacyDetector(Detector):
    name = "privacy"

    def detect(self, interaction: Interaction, settings: dict[str, Any]) -> DetectorResult:
        findings = self._scan(interaction.response, location="response", actionable=True)
        if bool(settings.get("scan_prompt", True)):
            findings.extend(self._scan(interaction.prompt, location="prompt", actionable=False))
        if bool(settings.get("scan_conversation", False)):
            for index, turn in enumerate(interaction.conversation):
                findings.extend(
                    self._scan(
                        turn.content,
                        location=f"conversation:{index}:{turn.role}",
                        actionable=False,
                    )
                )
        return DetectorResult(
            detector=self.name,
            findings=findings,
            metadata={"scanned_prompt": bool(settings.get("scan_prompt", True))},
        )

    def _scan(self, text: str, *, location: str, actionable: bool) -> list[Finding]:
        findings: list[Finding] = []
        occupied: set[tuple[int, int, str]] = set()
        for spec in _PATTERNS:
            for match in spec.pattern.finditer(text or ""):
                key = (match.start(), match.end(), spec.subtype)
                if key in occupied:
                    continue
                occupied.add(key)
                subtype = spec.subtype if actionable else f"input_{spec.subtype}"
                findings.append(
                    Finding(
                        category=RiskCategory.PRIVACY,
                        subtype=subtype,
                        severity=spec.severity,
                        confidence=spec.confidence,
                        status=FindingStatus.DETECTED,
                        message=f"Potential {spec.subtype.replace('_', ' ')} found in {location}.",
                        detector=self.name,
                        span=match.span() if actionable else None,
                        replacement=spec.replacement if actionable else None,
                        metadata={"location": location},
                    )
                )

        for match in _PHONE_RE.finditer(text or ""):
            raw = match.group(0).strip()
            digits = re.sub(r"\D", "", raw)
            # Bare 12-digit values are commonly Aadhaar identifiers and bare
            # 13-19 digit values may be payment cards. International phone
            # numbers in that range must carry an explicit '+' prefix.
            valid_length = 10 <= len(digits) <= 15
            ambiguous_bare_identifier = not raw.startswith("+") and len(digits) != 10
            overlaps = any(
                max(match.start(), start) < min(match.end(), end)
                for start, end, _ in occupied
            )
            if not valid_length or ambiguous_bare_identifier or overlaps:
                continue
            subtype = "phone_number" if actionable else "input_phone_number"
            findings.append(
                Finding(
                    category=RiskCategory.PRIVACY,
                    subtype=subtype,
                    severity=Severity.MEDIUM,
                    confidence=0.90,
                    status=FindingStatus.DETECTED,
                    message=f"Potential phone number found in {location}.",
                    detector=self.name,
                    span=match.span() if actionable else None,
                    replacement="[REDACTED PHONE]" if actionable else None,
                    metadata={"location": location},
                )
            )
            occupied.add((match.start(), match.end(), subtype))

        for match in _CARD_RE.finditer(text or ""):
            digits = re.sub(r"\D", "", match.group(0))
            overlaps = any(
                max(match.start(), start) < min(match.end(), end)
                for start, end, _ in occupied
            )
            if overlaps or not 13 <= len(digits) <= 19 or not self._luhn_valid(digits):
                continue
            findings.append(
                Finding(
                    category=RiskCategory.PRIVACY,
                    subtype="payment_card" if actionable else "input_payment_card",
                    severity=Severity.HIGH,
                    confidence=0.99,
                    status=FindingStatus.DETECTED,
                    message=f"Potential payment card number found in {location}.",
                    detector=self.name,
                    span=match.span() if actionable else None,
                    replacement="[REDACTED CARD]" if actionable else None,
                    metadata={"location": location},
                )
            )
        return findings

    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        total = 0
        parity = len(digits) % 2
        for index, char in enumerate(digits):
            value = int(char)
            if index % 2 == parity:
                value *= 2
                if value > 9:
                    value -= 9
            total += value
        return total % 10 == 0


def mask_privacy_values(text: str) -> str:
    """Blank actionable privacy spans while preserving response offsets.

    Factuality verification should reason about the surrounding claim, not use
    an email address, phone number, credential, identifier, or payment card as
    an entity/evidence feature. Spaces preserve all original character spans.
    """

    value = text or ""
    findings = PrivacyDetector()._scan(value, location="response", actionable=True)
    if not findings:
        return value
    characters = list(value)
    for finding in findings:
        if finding.span is None:
            continue
        start, end = finding.span
        characters[start:end] = " " * (end - start)
    return "".join(characters)
