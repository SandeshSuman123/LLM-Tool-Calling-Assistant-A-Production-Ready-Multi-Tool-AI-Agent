"""
tools/weather.py
==================

WHY THIS FILE EXISTS
---------------------
Weather is a classic tool-calling demo because it requires REAL-TIME data
the LLM cannot possibly know from training alone. This tool demonstrates
calling an external HTTP API and translating its response into a compact,
LLM-friendly summary.

RESPONSIBILITY
---------------
Given a city name, geocode it, fetch current weather from Open-Meteo
(a free, no-API-key-required weather service), and return a structured
summary (temperature, wind speed, conditions).

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas
✓ Error Handling (network failures, unknown city names)
"""

from __future__ import annotations

from typing import Any, Dict

import requests
from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from config import settings
from utils.errors import ToolExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> human-readable text.
# https://open-meteo.com/en/docs (see "WMO Weather interpretation codes")
_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    95: "Thunderstorm",
}


class WeatherArgs(BaseModel):
    """Arguments for the weather tool."""

    city: str = Field(..., description="City name, e.g. 'Kathmandu' or 'Jodhpur'.")


def _geocode(city: str) -> Dict[str, Any]:
    """Resolve a city name to latitude/longitude using Open-Meteo's geocoder."""
    response = requests.get(
        _GEOCODE_URL,
        params={"name": city, "count": 1},
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not results:
        raise ToolExecutionError(f"Could not find a location named '{city}'.")
    return results[0]


def run(args: WeatherArgs) -> ToolResult:
    """
    Fetch current weather conditions for `args.city`.

    WHY THIS FUNCTION EXISTS: demonstrates a two-step external API workflow
    (geocode -> forecast) wrapped in the same `ToolResult` contract every
    other tool uses, so the executor doesn't need to know these internals.
    """
    logger.info("Weather tool invoked for city=%r", args.city)
    try:
        location = _geocode(args.city)
        response = requests.get(
            _WEATHER_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current_weather": "true",
            },
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        current = response.json().get("current_weather", {})

        weather_code = current.get("weathercode")
        summary = {
            "city": location.get("name", args.city),
            "country": location.get("country"),
            "temperature_celsius": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
            "conditions": _WEATHER_CODES.get(weather_code, "Unknown"),
        }
        return ToolResult(success=True, data=summary)

    except ToolExecutionError as exc:
        return ToolResult(success=False, error=str(exc))
    except requests.exceptions.RequestException as exc:
        logger.exception("Weather API request failed")
        return ToolResult(success=False, error=f"Weather service unavailable: {exc}")


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="get_weather",
    description=(
        "Get the current real-time weather (temperature, wind, conditions) "
        "for a given city. Use this whenever the user asks about current "
        "weather — never guess weather from memory."
    ),
    args_model=WeatherArgs,
)
