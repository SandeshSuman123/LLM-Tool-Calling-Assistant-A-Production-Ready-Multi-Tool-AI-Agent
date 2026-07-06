"""
tools/converter.py
====================

WHY THIS FILE EXISTS
---------------------
Unit conversion is deterministic and easy to get subtly wrong if left to
an LLM's "mental math" (e.g. confusing miles vs nautical miles). Offloading
it to a tool guarantees numerically correct answers every time.

RESPONSIBILITY
---------------
Convert a numeric value between supported units across three categories:
length, weight/mass, and temperature.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas (enum-constrained "from_unit"/"to_unit")
✓ Error Handling (incompatible unit categories)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from utils.errors import ToolExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)


class Unit(str, Enum):
    """All units supported by this tool, across categories."""

    # Length
    METERS = "meters"
    KILOMETERS = "kilometers"
    MILES = "miles"
    FEET = "feet"
    INCHES = "inches"
    # Weight / mass
    KILOGRAMS = "kilograms"
    GRAMS = "grams"
    POUNDS = "pounds"
    OUNCES = "ounces"
    # Temperature
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


# Conversion factors TO a base unit for each category.
_LENGTH_TO_METERS = {
    Unit.METERS: 1.0,
    Unit.KILOMETERS: 1000.0,
    Unit.MILES: 1609.344,
    Unit.FEET: 0.3048,
    Unit.INCHES: 0.0254,
}

_WEIGHT_TO_GRAMS = {
    Unit.KILOGRAMS: 1000.0,
    Unit.GRAMS: 1.0,
    Unit.POUNDS: 453.592,
    Unit.OUNCES: 28.3495,
}

_TEMPERATURE_UNITS = {Unit.CELSIUS, Unit.FAHRENHEIT, Unit.KELVIN}


class ConverterArgs(BaseModel):
    """Arguments for the unit conversion tool."""

    value: float = Field(..., description="The numeric value to convert.")
    from_unit: Unit = Field(..., description="The unit `value` is currently in.")
    to_unit: Unit = Field(..., description="The unit to convert `value` into.")


def _convert_temperature(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """Convert between Celsius, Fahrenheit, and Kelvin via Celsius as a pivot."""
    # Normalize input to Celsius first.
    if from_unit == Unit.CELSIUS:
        celsius = value
    elif from_unit == Unit.FAHRENHEIT:
        celsius = (value - 32) * 5.0 / 9.0
    elif from_unit == Unit.KELVIN:
        celsius = value - 273.15
    else:  # pragma: no cover
        raise ToolExecutionError(f"Not a temperature unit: {from_unit}")

    # Convert from Celsius to the target unit.
    if to_unit == Unit.CELSIUS:
        return celsius
    if to_unit == Unit.FAHRENHEIT:
        return celsius * 9.0 / 5.0 + 32
    if to_unit == Unit.KELVIN:
        return celsius + 273.15
    raise ToolExecutionError(f"Not a temperature unit: {to_unit}")  # pragma: no cover


def run(args: ConverterArgs) -> ToolResult:
    """
    Convert `args.value` from `args.from_unit` to `args.to_unit`.
    """
    logger.info(
        "Converter tool invoked: %s %s -> %s", args.value, args.from_unit, args.to_unit
    )
    from_unit, to_unit = args.from_unit, args.to_unit

    try:
        if from_unit in _TEMPERATURE_UNITS or to_unit in _TEMPERATURE_UNITS:
            if from_unit not in _TEMPERATURE_UNITS or to_unit not in _TEMPERATURE_UNITS:
                raise ToolExecutionError(
                    "Cannot convert between a temperature unit and a non-temperature unit."
                )
            result = _convert_temperature(args.value, from_unit, to_unit)

        elif from_unit in _LENGTH_TO_METERS and to_unit in _LENGTH_TO_METERS:
            meters = args.value * _LENGTH_TO_METERS[from_unit]
            result = meters / _LENGTH_TO_METERS[to_unit]

        elif from_unit in _WEIGHT_TO_GRAMS and to_unit in _WEIGHT_TO_GRAMS:
            grams = args.value * _WEIGHT_TO_GRAMS[from_unit]
            result = grams / _WEIGHT_TO_GRAMS[to_unit]

        else:
            raise ToolExecutionError(
                f"Cannot convert between incompatible units: '{from_unit}' and '{to_unit}'."
            )

        return ToolResult(
            success=True,
            data={
                "input": f"{args.value} {from_unit.value}",
                "output": f"{round(result, 6)} {to_unit.value}",
                "value": round(result, 6),
            },
        )

    except ToolExecutionError as exc:
        return ToolResult(success=False, error=str(exc))


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="unit_converter",
    description=(
        "Convert a numeric value between units of length, weight/mass, or "
        "temperature (e.g. miles to kilometers, Fahrenheit to Celsius)."
    ),
    args_model=ConverterArgs,
)
