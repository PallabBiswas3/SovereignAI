from __future__ import annotations

import re
from decimal import Decimal

import pint
from pydantic import BaseModel


class UnitServiceError(ValueError):
    code = "UNIT_ERROR"


class UnitAmbiguousError(UnitServiceError):
    code = "UNIT_AMBIGUOUS"


class IncompatibleUnitsError(UnitServiceError):
    code = "INCOMPATIBLE_UNITS"


class NormalizedQuantity(BaseModel):
    original_value: float
    original_unit: str
    normalized_value: float
    normalized_unit: str
    dimension: str


class UnitService:
    """Pint-backed local engineering unit conversion with explicit dimensions."""

    UNIT_ALIASES = {
        "c": "degC",
        "°c": "degC",
        "celsius": "degC",
        "degc": "degC",
        "k": "kelvin",
        "kelvin": "kelvin",
        "pa": "pascal",
        "pascal": "pascal",
        "kpa": "kilopascal",
        "kilopascal": "kilopascal",
        "mpa": "megapascal",
        "megapascal": "megapascal",
        "bar": "bar",
        "mm": "millimeter",
        "cm": "centimeter",
        "m": "meter",
        "mm/s": "millimeter / second",
        "mm/s rms": "millimeter / second",
        "millimeter / second": "millimeter / second",
        "m/s": "meter / second",
        "meter / second": "meter / second",
        "w": "watt",
        "kw": "kilowatt",
        "kilowatt": "kilowatt",
        "mw": "megawatt",
        "megawatt": "megawatt",
        "watt": "watt",
        "n": "newton",
        "kn": "kilonewton",
        "kilonewton": "kilonewton",
        "newton": "newton",
        "g": "gram",
        "kg": "kilogram",
        "kilogram": "kilogram",
        "gram": "gram",
        "hz": "hertz",
        "hertz": "hertz",
        "rpm": "engineering_rpm",
        "engineering_rpm": "engineering_rpm",
        "meter": "meter",
        "millimeter": "millimeter",
        "centimeter": "centimeter",
    }
    CANONICAL = {
        "[mass] / [length] / [time] ** 2": ("bar", "pressure"),
        "[temperature]": ("degC", "temperature"),
        "[length]": ("meter", "length"),
        "[length] / [time]": ("millimeter / second", "velocity"),
        "[mass] * [length] ** 2 / [time] ** 3": ("kilowatt", "power"),
        "[mass] * [length] / [time] ** 2": ("newton", "force"),
        "[mass]": ("kilogram", "mass"),
        "1 / [time]": ("hertz", "frequency"),
    }
    DISPLAY = {
        "degC": "°C",
        "kelvin": "K",
        "millimeter / second": "mm/s",
        "meter / second": "m/s",
        "kilopascal": "kPa",
        "megapascal": "MPa",
        "pascal": "Pa",
        "kilowatt": "kW",
        "megawatt": "MW",
        "watt": "W",
        "kilonewton": "kN",
        "newton": "N",
        "kilogram": "kg",
        "gram": "g",
        "hertz": "Hz",
        "engineering_rpm": "rpm",
        "meter": "m",
        "millimeter": "mm",
        "centimeter": "cm",
        "bar": "bar",
    }

    def __init__(self) -> None:
        self.registry = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
        self.registry.define("engineering_rpm = count / minute")

    def _unit(self, unit: str | None) -> str:
        if not unit or not unit.strip():
            raise UnitAmbiguousError("UNIT_AMBIGUOUS: a unit is required and cannot be inferred safely")
        normalized = re.sub(r"\s+", " ", unit.strip().lower().replace("r.m.s.", "rms"))
        mapped = self.UNIT_ALIASES.get(normalized)
        if not mapped:
            raise UnitAmbiguousError(f"UNIT_AMBIGUOUS: unsupported or ambiguous unit '{unit}'")
        return mapped

    def compatible(self, left_unit: str | None, right_unit: str | None) -> bool:
        left = self._unit(left_unit)
        right = self._unit(right_unit)
        try:
            (1 * self.registry(left)).to(right)
            return True
        except pint.DimensionalityError:
            return False

    def convert(
        self,
        value: float,
        from_unit: str | None,
        to_unit: str | None,
        *,
        precision: int | None = None,
    ) -> NormalizedQuantity:
        source = self._unit(from_unit)
        target = self._unit(to_unit)
        try:
            quantity = self.registry.Quantity(value, source).to(target)
        except pint.DimensionalityError as exc:
            raise IncompatibleUnitsError(
                f"INCOMPATIBLE_UNITS: cannot compare '{from_unit}' with '{to_unit}'"
            ) from exc
        converted = self._round(float(quantity.magnitude), value, precision)
        dimension = self._dimension(source)
        return NormalizedQuantity(
            original_value=value,
            original_unit=str(from_unit),
            normalized_value=converted,
            normalized_unit=self.DISPLAY.get(target, str(to_unit)),
            dimension=dimension,
        )

    def normalize(self, value: float, unit: str | None) -> NormalizedQuantity:
        source = self._unit(unit)
        dimension_key = str(self.registry.get_dimensionality(source))
        target_info = self.CANONICAL.get(dimension_key)
        if target_info is None:
            raise UnitAmbiguousError(f"UNIT_AMBIGUOUS: no canonical engineering unit for '{unit}'")
        return self.convert(value, unit, target_info[0])

    def _dimension(self, unit: str) -> str:
        dimension_key = str(self.registry.get_dimensionality(unit))
        return self.CANONICAL.get(dimension_key, (unit, dimension_key))[1]

    @staticmethod
    def _round(value: float, original: float, precision: int | None) -> float:
        if precision is None:
            decimal = Decimal(str(original))
            decimals = max(1, min(6, -decimal.as_tuple().exponent + 2))
        else:
            decimals = max(0, min(8, precision))
        return round(value, decimals)
