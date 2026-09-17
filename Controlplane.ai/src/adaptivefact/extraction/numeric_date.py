from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

try:
    import dateparser
except ImportError:  # optional at runtime; simple formats still work without it
    dateparser = None

from datetime import datetime

_NUMBER_TOKEN_RE = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:%|percent|bn|billion|mn|million|m|k|thousand|crore|lakh)?(?!\w)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_MONTH_NAME = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RE = re.compile(
    rf"\b(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+(?:18|19|20|21)\d{{2}}\b|"
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAME})\s+(?:18|19|20|21)\d{{2}}\b|"
    r"\b(?:18|19|20|21)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d{1,2}[-/]\d{1,2}[-/](?:18|19|20|21)?\d{2}\b",
    re.IGNORECASE,
)

_MULTIPLIERS = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "mn": Decimal("1000000"),
    "million": Decimal("1000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "lakh": Decimal("100000"),
    "crore": Decimal("10000000"),
}


@dataclass(frozen=True)
class NumberValue:
    raw: str
    normalized: str
    kind: str
    unit: str | None


@dataclass(frozen=True)
class DateValue:
    raw: str
    normalized: str


def _decimal_string(value: Decimal) -> str:
    value = value.normalize()
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalize_number(raw: str) -> NumberValue | None:
    original = raw.strip()
    if not original:
        return None

    lowered = original.lower().replace(",", "").strip()
    currency = None
    if lowered and lowered[0] in "$€£₹":
        currency = lowered[0]
        lowered = lowered[1:].strip()

    percent = lowered.endswith("%") or lowered.endswith("percent")
    lowered = re.sub(r"\s*percent$", "", lowered).rstrip("%").strip()

    multiplier = Decimal("1")
    suffix = None
    match = re.search(r"\s*(bn|billion|mn|million|m|k|thousand|crore|lakh)$", lowered)
    if match:
        suffix = match.group(1)
        multiplier = _MULTIPLIERS[suffix]
        lowered = lowered[: match.start()].strip()

    try:
        value = Decimal(lowered) * multiplier
    except InvalidOperation:
        return None

    kind = "percent" if percent else "currency" if currency else "number"
    unit = "%" if percent else currency
    normalized = _decimal_string(value)
    return NumberValue(raw=original, normalized=normalized, kind=kind, unit=unit)


def extract_numbers(text: str) -> list[NumberValue]:
    values: list[NumberValue] = []
    seen: set[tuple[str, str, str | None]] = set()
    for match in _NUMBER_TOKEN_RE.finditer(text or ""):
        parsed = normalize_number(match.group(0))
        if parsed is None:
            continue
        key = (parsed.normalized, parsed.kind, parsed.unit)
        if key not in seen:
            values.append(parsed)
            seen.add(key)
    return values


def extract_dates(text: str) -> list[DateValue]:
    values: list[DateValue] = []
    occupied: list[tuple[int, int]] = []

    for match in _DATE_RE.finditer(text or ""):
        raw = match.group(0)
        parsed = dateparser.parse(raw, settings={"STRICT_PARSING": True}) if dateparser is not None else None
        if parsed is None:
            cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", raw, flags=re.IGNORECASE).replace(",", "").strip()
            for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    continue
        normalized = parsed.date().isoformat() if parsed else raw.lower().replace(",", "").strip()
        values.append(DateValue(raw=raw, normalized=normalized))
        occupied.append(match.span())

    for match in _YEAR_RE.finditer(text or ""):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        raw = match.group(0)
        values.append(DateValue(raw=raw, normalized=raw))

    deduped: list[DateValue] = []
    seen = set()
    for value in values:
        if value.normalized not in seen:
            deduped.append(value)
            seen.add(value.normalized)
    return deduped
