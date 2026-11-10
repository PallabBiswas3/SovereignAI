from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Pattern

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
    # Extra acceptance check for a match (e.g. reject an already-masked value).
    validator: Callable[["re.Match[str]", str], bool] | None = None
    # Numeric identifiers: skipped inside dates/times and technical identifiers.
    numeric: bool = False


# Industrial text is full of numbers that are not personal data. These spans are
# reserved before numeric identifier patterns run, so a timestamp such as
# "2026-10-02 14:30:55" can no longer be read as a phone number.
_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b|"
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|"
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
)
# A number introduced as an equipment/document identifier is not a person's ID.
_TECHNICAL_ID_CONTEXT = re.compile(
    r"(?:serial(?:\s+(?:no\.?|number))?|s/n|\bsn|tag(?:\s+no\.?)?|part(?:\s+(?:no\.?|number))?|p/n|"
    r"model(?:\s+no\.?)?|batch|lot|work\s+order|\bwo|purchase\s+order|\bpo|invoice|asset(?:\s+id)?|"
    r"firmware|version|\brev(?:ision)?|\bfw|\bv)\s*[#:.=-]?\s*$",
    re.IGNORECASE,
)
_MASKED_SECRET = re.compile(r"^(?:\*+|x+|•+|\[?redacted\]?.*|<[^>]*>|\.\.\.|none|not\s+set|required|changed|reset|expired|unknown)$", re.I)


def _real_password(match: "re.Match[str]", text: str) -> bool:
    return not _MASKED_SECRET.match(match.group("value").strip("'\"").rstrip(".,;"))


def _not_version_number(match: "re.Match[str]", text: str) -> bool:
    return not _TECHNICAL_ID_CONTEXT.search(text[max(0, match.start() - 24):match.start()])


# Verhoeff check digit (used by Aadhaar). A random 12-digit serial or reading
# passes with probability ~1/10, so this removes most false positives without
# guessing from surrounding words.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(char)]]
    return checksum == 0


def _aadhaar_checksum(match: "re.Match[str]", text: str) -> bool:
    return verhoeff_valid(re.sub(r"\D", "", match.group(0)))


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
        "password",
        re.compile(
            r"\b(?:password|passwd|pwd|passcode|pass\s*phrase)\b\s*(?:is|was|[:=])\s*"
            r"(?P<value>['\"][^'\"\n]{4,}['\"]|[^\s'\"]{4,})",
            re.I,
        ),
        Severity.CRITICAL,
        0.95,
        "password: [REDACTED PASSWORD]",
        validator=_real_password,
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
        numeric=True,
    ),
    PrivacyPattern(
        "aadhaar_number",
        re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)"),
        Severity.HIGH,
        0.97,
        "[REDACTED ID]",
        validator=_aadhaar_checksum,
        numeric=True,
    ),
    PrivacyPattern(
        # Indian PAN: 5 letters (4th = holder type), 4 digits, 1 letter.
        "pan_number",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z](?![A-Za-z0-9])"),
        Severity.HIGH,
        0.93,
        "[REDACTED ID]",
    ),
    PrivacyPattern(
        # Plant/OT network addresses (PLC, HMI, historian) are attack-surface data.
        "ip_address",
        re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.]|\.\d)"),
        Severity.MEDIUM,
        0.90,
        "[REDACTED IP]",
        validator=_not_version_number,
    ),
]

_OT_ADDRESS = PrivacyPattern(
    "ot_network_address", _PATTERNS[-1].pattern, Severity.HIGH, 0.97, "[REDACTED OT ADDRESS]",
    validator=_not_version_number,
)

_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")  # ends on a digit: keeps the following space
# Do not start a phone number in the middle of a date, decimal, ratio or tag.
_PHONE_RE = re.compile(r"(?<![\w.:/-])\+?(?:\d[\s().-]?){9,14}\d(?!\w)")


class PrivacyRecognizerConfig:
    """Organization-specific recognizers, loaded from YAML instead of code.

    Example (set CONTROLPLANE_PRIVACY_RECOGNIZERS to its path)::

        ot_networks: ["10.20.0.0/16"]          # PLC/HMI/historian ranges
        identifiers:
          - subtype: employee_id
            pattern: "\\\\bEMP-\\\\d{6}\\\\b"
            severity: medium                 # low|medium|high|critical
            replacement: "[REDACTED EMPLOYEE ID]"

    An invalid file raises at startup: a silently ignored recognizer would
    look like protection that is not there.
    """

    def __init__(self, ot_networks: list[str] | None = None, identifiers: list[dict] | None = None) -> None:
        import ipaddress

        self.ot_networks = [ipaddress.ip_network(value, strict=False) for value in (ot_networks or [])]
        self.patterns: list[PrivacyPattern] = []
        for item in identifiers or []:
            subtype = str(item.get("subtype", ""))
            if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", subtype):
                raise ValueError(f"privacy identifier subtype must be snake_case: {subtype!r}")
            self.patterns.append(PrivacyPattern(
                subtype, re.compile(str(item["pattern"])), Severity(str(item.get("severity", "medium")).lower()),
                float(item.get("confidence", 0.95)), str(item.get("replacement", f"[REDACTED {subtype.upper()}]")),
            ))

    @classmethod
    def from_file(cls, path: str) -> "PrivacyRecognizerConfig":
        from pathlib import Path

        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - {"ot_networks", "identifiers"}
        if unknown:
            raise ValueError(f"unknown privacy recognizer keys: {sorted(unknown)}")
        return cls(data.get("ot_networks"), data.get("identifiers"))

    @classmethod
    def from_env(cls) -> "PrivacyRecognizerConfig":
        import os

        path = os.getenv("CONTROLPLANE_PRIVACY_RECOGNIZERS", "").strip()
        return cls.from_file(path) if path else cls()

    def in_ot_network(self, address: str) -> bool:
        import ipaddress

        value = ipaddress.ip_address(address)
        return any(value in network for network in self.ot_networks)


class PrivacyDetector(Detector):
    name = "privacy"

    def __init__(self, recognizers: PrivacyRecognizerConfig | None = None) -> None:
        self.recognizers = recognizers if recognizers is not None else PrivacyRecognizerConfig.from_env()
        self.patterns = [*_PATTERNS, *self.recognizers.patterns]

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

    @staticmethod
    def _technical_number(text: str, start: int, end: int, reserved: list[tuple[int, int]]) -> bool:
        """A date/time, or a number introduced as an equipment/document ID."""
        if any(max(start, r_start) < min(end, r_end) for r_start, r_end in reserved):
            return True
        return bool(_TECHNICAL_ID_CONTEXT.search(text[max(0, start - 24):start]))

    def _scan(self, text: str, *, location: str, actionable: bool) -> list[Finding]:
        findings: list[Finding] = []
        occupied: set[tuple[int, int, str]] = set()
        text = text or ""
        reserved = [match.span() for match in _DATETIME_RE.finditer(text)]
        for pattern_spec in self.patterns:
            for match in pattern_spec.pattern.finditer(text):
                spec = pattern_spec
                # An address inside a configured plant/OT range is control-
                # network attack surface: report it at higher severity.
                if spec.subtype == "ip_address" and self.recognizers.in_ot_network(match.group(0)):
                    spec = _OT_ADDRESS
                key = (match.start(), match.end(), spec.subtype)
                if key in occupied:
                    continue
                if spec.validator is not None and not spec.validator(match, text):
                    continue
                if spec.numeric and self._technical_number(text, match.start(), match.end(), reserved):
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

        for match in _PHONE_RE.finditer(text):
            if self._technical_number(text, match.start(), match.end(), reserved):
                continue
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

        for match in _CARD_RE.finditer(text):
            if self._technical_number(text, match.start(), match.end(), reserved):
                continue
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


_DEFAULT_DETECTOR: PrivacyDetector | None = None


def _default_detector() -> PrivacyDetector:
    """Shared detector with the process's configured recognizers (loaded once)."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = PrivacyDetector()
    return _DEFAULT_DETECTOR


def mask_privacy_values(text: str) -> str:
    """Blank actionable privacy spans while preserving response offsets.

    Factuality verification should reason about the surrounding claim, not use
    an email address, phone number, credential, identifier, or payment card as
    an entity/evidence feature. Spaces preserve all original character spans.
    """

    value = text or ""
    findings = _default_detector()._scan(value, location="response", actionable=True)
    if not findings:
        return value
    characters = list(value)
    for finding in findings:
        if finding.span is None:
            continue
        start, end = finding.span
        characters[start:end] = " " * (end - start)
    return "".join(characters)
