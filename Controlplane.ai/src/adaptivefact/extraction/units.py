"""Unit-aware physical quantities for release-gate checks.

The general number extractor in ``numeric_date`` is unit-blind (``4 bar`` and
``4 psi`` are both ``4``) and reads ``6 m`` as six million. This module parses
industrial measurements with their units, converts them within one physical
dimension, and compares values with a tolerance derived from the precision the
text actually states. It is deliberately conservative:

- quantities in different dimensions never match (``6 bar`` vs ``6 mm``);
- gauge/absolute pressure and RMS/peak vibration qualifiers must agree when both
  texts state them;
- a converted match only counts when it agrees to the precision written in the
  claim (``58 psi`` matches ``4 bar`` = 58.02 psi; ``4.0 psi`` does not).

It never decides that a claim is true. Callers use it to *block* support for a
quantity that is absent from the supporting evidence, and to present an equal
quantity to NLI in the source's unit so that a correct conversion is not
mistaken for a contradiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

# unit spelling -> (dimension, factor to base, offset to base, canonical symbol)
# Single-letter or case-significant units are matched case-sensitively below.
_UNITS: dict[str, tuple[str, Decimal, Decimal, str]] = {}


def _add(dimension: str, factor: str, symbol: str, *spellings: str, offset: str = "0") -> None:
    for spelling in (symbol, *spellings):
        _UNITS[spelling] = (dimension, Decimal(factor), Decimal(offset), symbol)


# Pressure (base Pa). Gauge and plain units share a dimension; absolute units
# are kept separate because they differ by one atmosphere.
_add("pressure", "100000", "bar", "barg", "bar(g)")
_add("pressure", "100", "mbar", "mbarg")
_add("pressure", "1", "Pa")
_add("pressure", "1000", "kPa", "kpa", "kPag")
_add("pressure", "1000000", "MPa", "mpa", "MPag")
_add("pressure", "6894.757293168", "psi", "psig", "PSI", "PSIG")
_add("pressure", "98066.5", "kg/cm2", "kg/cm²", "kgf/cm2", "kgf/cm²")
_add("pressure_abs", "100000", "bara", "bar(a)")
_add("pressure_abs", "6894.757293168", "psia", "PSIA")
_add("pressure_abs", "1000", "kPaa", "kPa(a)")
_add("pressure_abs", "101325", "atm")
# Length (base m).
_add("length", "0.000001", "µm", "um", "micron", "microns")
_add("length", "0.001", "mm")
_add("length", "0.01", "cm")
_add("length", "1", "m", "metre", "metres", "meter", "meters")
_add("length", "1000", "km")
# Bare "in" is omitted: "rose 4 in the last hour" is not 4 inches.
_add("length", "0.0254", "inch", "inches")
_add("length", "0.3048", "ft", "feet", "foot")
# Velocity, including vibration velocity (base m/s).
_add("velocity", "0.001", "mm/s")
_add("velocity", "0.01", "cm/s")
_add("velocity", "0.0254", "in/s", "ips")
_add("velocity", "1", "m/s")
# Acceleration (base m/s²).
_add("acceleration", "1", "m/s2", "m/s²")
_add("acceleration", "9.80665", "g", "G")
# Temperature (base K, affine).
_add("temperature", "1", "°C", "degC", "deg C", "C", "℃", "celsius", "Celsius", offset="273.15")
_add("temperature", "0.5555555555555555555555555556", "°F", "degF", "deg F", "F", "℉", "fahrenheit", "Fahrenheit",
     offset="255.3722222222222222222222222")
_add("temperature", "1", "K", "kelvin")
# Rotational speed / frequency (base Hz).
_add("frequency", "1", "Hz", "hz")
_add("frequency", "1000", "kHz", "khz")
_add("frequency", "0.01666666666666666666666666667", "rpm", "RPM", "r/min", "cpm", "CPM")
# Volumetric flow (base m³/s).
_add("flow", "0.0002777777777777777777777777778", "m3/h", "m³/h", "m3/hr", "m³/hr")
_add("flow", "0.00001666666666666666666666666667", "L/min", "l/min", "lpm", "LPM")
_add("flow", "0.001", "L/s", "l/s")
_add("flow", "0.0000630901964", "gpm", "GPM", "USgpm")
# Power (base W).
_add("power", "1", "W")
_add("power", "1000", "kW", "kw")
_add("power", "1000000", "MW")
_add("power", "745.6998715822702", "hp", "HP", "bhp")
# Electrical (base A / V).
_add("current", "1", "A", "amp", "amps", "ampere", "amperes")
_add("current", "0.001", "mA")
_add("voltage", "1", "V", "volt", "volts")
_add("voltage", "1000", "kV", "kv")
# Time (base s).
_add("time", "1", "s", "sec", "secs", "second", "seconds")
_add("time", "60", "min", "mins", "minute", "minutes")
_add("time", "3600", "h", "hr", "hrs", "hour", "hours")
_add("time", "86400", "day", "days")
# Dimensionless.
_add("percent", "1", "%", "percent", "pct")

# Spellings whose case carries meaning (m vs M, C vs c, A vs a, ...).
_CASE_SENSITIVE = {spelling for spelling in _UNITS if len(spelling) <= 2 and spelling.isalpha()} | {"Pa", "PSI"}
_INSENSITIVE = {spelling.casefold(): spelling for spelling in _UNITS if spelling not in _CASE_SENSITIVE}

_UNIT_ALTERNATION = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
_NUMBER = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
_QUALIFIER = r"(?:\s*\(?\s*(?P<qual>rms|RMS|pk-pk|peak-to-peak|peak|pk|p-p|0-pk)\s*\)?)?"
_QUANTITY = re.compile(
    rf"(?<![\w.-])(?P<between>between\s+)?(?P<low>{_NUMBER})"
    # A unit after the first number is only allowed when a range separator
    # follows; otherwise "600 cm" could split into "c" + "m".
    rf"(?:(?:\s*(?P<low_unit>{_UNIT_ALTERNATION})(?![\w/²]))?\s*(?P<sep>to|and|[-–—])\s*(?P<high>{_NUMBER}))?"
    rf"\s*(?P<unit>{_UNIT_ALTERNATION})(?![\w/²])" + _QUALIFIER,
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    """One measured value or range, converted to its dimension's base unit."""

    text: str
    start: int
    end: int
    dimension: str
    unit: str
    low: Decimal  # base units
    high: Decimal  # equal to ``low`` for a scalar
    low_tolerance: Decimal  # base-unit half-step of the stated precision
    high_tolerance: Decimal
    qualifier: str | None
    is_range: bool

    @property
    def values(self) -> tuple[tuple[Decimal, Decimal], ...]:
        """Base values with tolerances (one for a scalar, two for a range)."""
        if self.is_range:
            return ((self.low, self.low_tolerance), (self.high, self.high_tolerance))
        return ((self.low, self.low_tolerance),)


def _resolve(spelling: str | None) -> tuple[str, Decimal, Decimal, str] | None:
    if not spelling:
        return None
    if spelling in _UNITS:
        return _UNITS[spelling]
    canonical = _INSENSITIVE.get(spelling.casefold())
    return _UNITS[canonical] if canonical else None


def _half_step(raw: str) -> Decimal:
    digits = raw.split(".", 1)[1] if "." in raw else ""
    return Decimal(5) * Decimal(10) ** (-(len(digits) + 1))


def _qualifier(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.casefold()
    if raw == "rms":
        return "rms"
    if raw in {"pk-pk", "peak-to-peak", "p-p"}:
        return "peak-to-peak"
    return "peak"


def _quantity(text: str, start: int, end: int, unit, low_raw: str, high_raw: str | None,
              low_unit=None, qualifier: str | None = None) -> Quantity:
    dimension, factor, offset, symbol = unit
    low_factor, low_offset = (low_unit or unit)[1], (low_unit or unit)[2]
    low_raw = low_raw.replace(",", "")
    high_raw = (high_raw or low_raw).replace(",", "")
    low = Decimal(low_raw) * low_factor + low_offset
    high = Decimal(high_raw) * factor + offset
    low_tol, high_tol = _half_step(low_raw) * low_factor, _half_step(high_raw) * factor
    if low > high:
        low, high, low_tol, high_tol = high, low, high_tol, low_tol
    return Quantity(text=text[start:end].strip(), start=start, end=end, dimension=dimension, unit=symbol,
                    low=low, high=high, low_tolerance=low_tol, high_tolerance=high_tol,
                    qualifier=qualifier, is_range=low != high or high_raw != low_raw)


def extract_quantities(text: str) -> list[Quantity]:
    """Return unit-bearing scalars and ranges, in order of appearance."""
    text = text or ""
    output: list[Quantity] = []
    for match in _QUANTITY.finditer(text):
        unit = _resolve(match.group("unit"))
        if unit is None:
            continue  # e.g. "5 a" / "3 c": case-sensitive spelling did not match
        qualifier = _qualifier(match.group("qual"))
        low_unit = _resolve(match.group("low_unit")) if match.group("low_unit") else None
        if match.group("high") is None:
            output.append(_quantity(text, match.start(), match.end(), unit, match.group("low"), None,
                                    qualifier=qualifier))
            continue
        joined_by_and = match.group("sep").casefold() == "and" and not match.group("between")
        if joined_by_and or (low_unit is not None and low_unit[0] != unit[0]):
            # "5.1 bar and 80 °C" / "pressure 5.1 and 80 °C" are separate values,
            # not one range. Keep the first only if it carried its own unit.
            if low_unit is not None:
                output.append(_quantity(text, match.start(), match.end("low_unit"), low_unit, match.group("low"), None))
            output.append(_quantity(text, match.start("high"), match.end(), unit, match.group("high"), None,
                                    qualifier=qualifier))
            continue
        output.append(_quantity(text, match.start(), match.end(), unit, match.group("low"), match.group("high"),
                                low_unit=low_unit, qualifier=qualifier))
    return output


def _value_matches(claim_value: Decimal, claim_tol: Decimal, source_value: Decimal, source_tol: Decimal) -> bool:
    # Agree to the coarser of the two stated precisions, plus a tiny absolute
    # epsilon for decimal conversion factors.
    return abs(claim_value - source_value) <= max(claim_tol, source_tol) + Decimal("1e-9") * max(abs(source_value), 1)


def _compatible(claim: Quantity, source: Quantity) -> bool:
    if claim.dimension != source.dimension:
        return False
    return claim.qualifier is None or source.qualifier is None or claim.qualifier == source.qualifier


def quantity_witnessed(claim: Quantity, sources: list[Quantity]) -> bool:
    """True when every value the claim states also appears in the sources.

    A claimed scalar may equal a source scalar or either end of a source range
    ("upper limit 6 bar" vs "4 to 6 bar"). A claimed range needs both ends.
    """
    candidates = [source for source in sources if _compatible(claim, source)]
    source_values = [value for source in candidates for value in source.values]
    if claim.is_range:
        exact_ranges = [source for source in candidates if source.is_range]
        if any(_value_matches(claim.low, claim.low_tolerance, s.low, s.low_tolerance)
               and _value_matches(claim.high, claim.high_tolerance, s.high, s.high_tolerance) for s in exact_ranges):
            return True
    return all(any(_value_matches(value, tol, other, other_tol) for other, other_tol in source_values)
               for value, tol in claim.values)


def unwitnessed_quantities(claim_text: str, evidence_texts: list[str]) -> list[Quantity]:
    """Claim quantities that no evidence text states, after unit conversion."""
    claim_quantities = extract_quantities(claim_text)
    if not claim_quantities:
        return []
    sources = [quantity for text in evidence_texts for quantity in extract_quantities(text)]
    return [quantity for quantity in claim_quantities if not quantity_witnessed(quantity, sources)]


def scalar_outside_source_ranges(claim_text: str, evidence_text: str) -> bool:
    """True when a claimed reading lies outside a same-dimension source range.

    ``"normal at 6.8 bar"`` against ``"normal range 4.0 to 6.0 bar"`` is True;
    ``"recorded 5.1 bar"`` against the same range is False. Values are compared
    after unit conversion, so ``510 kPa`` is inside ``4 to 6 bar``.
    """
    ranges = [quantity for quantity in extract_quantities(evidence_text) if quantity.is_range]
    for claim in extract_quantities(claim_text):
        if claim.is_range:
            continue
        same = [source for source in ranges if _compatible(claim, source)]
        if same and not any(
            source.low - source.low_tolerance <= claim.low <= source.high + source.high_tolerance for source in same
        ):
            return True
    return False


def align_units_to_premise(hypothesis: str, premise: str) -> str:
    """Rewrite claim quantities into the premise's unit when they are equal.

    ``600 cm`` becomes ``6 m`` for a premise that states ``6 m``. Nothing is
    rewritten unless the converted value matches a premise value, so a wrong
    quantity is still presented to NLI exactly as the claim wrote it.
    """
    premise_quantities = extract_quantities(premise)
    if not premise_quantities:
        return hypothesis
    replacements: list[tuple[int, int, str]] = []
    for claim in extract_quantities(hypothesis):
        for source in premise_quantities:
            if source.unit == claim.unit or not _compatible(claim, source) or source.is_range != claim.is_range:
                continue
            if not quantity_witnessed(claim, [source]):
                continue
            replacements.append((claim.start, claim.end, source.text))
            break
    for start, end, text in sorted(replacements, reverse=True):
        hypothesis = hypothesis[:start] + text + hypothesis[end:]
    return hypothesis
