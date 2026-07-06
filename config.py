"""
config.py
=========

WHY THIS FILE EXISTS
---------------------
Every real project needs ONE place where configuration lives. Hard-coding API
keys, model names, or timeouts directly inside business logic is a classic
beginner mistake — it makes the code impossible to reconfigure without
editing source, and it risks leaking secrets into version control.

RESPONSIBILITY
---------------
- Load environment variables from a `.env` file (via python-dotenv).
- Validate that required secrets (like the OpenRouter API key) are present.
- Expose a single `Settings` object that the rest of the app imports from,
  instead of every module calling `os.getenv(...)` directly.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- `agent/llm.py` reads `Settings.OPENROUTER_API_KEY`, `Settings.MODEL_NAME`,
  and `Settings.BASE_URL` to construct the OpenAI-compatible client.
- `main.py` imports `Settings` at startup to fail fast if configuration
  is missing, rather than crashing deep inside a tool call later.

CONCEPTS DEMONSTRATED
-----------------------
✓ Environment Variables
✓ Configuration Management
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a .env file in the project root into os.environ.
# This must happen BEFORE we read any os.getenv() calls below.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration object.

    Using a frozen dataclass (rather than a plain dict or scattered
    os.getenv calls) gives us:
      - Autocomplete / type hints in editors.
      - A single source of truth.
      - Immutability, so nothing accidentally mutates config at runtime.
    """

    # --- OpenRouter / LLM configuration -----------------------------------
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Free model on OpenRouter that supports tool calling well.
    # You can swap this for any other OpenRouter model ID.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen/qwen-2.5-72b-instruct:free")

    # --- Optional tool API keys --------------------------------------------
    # Weather tool uses Open-Meteo (no key required) by default, but we allow
    # an OpenWeatherMap key to be plugged in if the user prefers it.
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")

    # Web search tool uses a lightweight DuckDuckGo HTML scrape by default,
    # but a Tavily/Serper key can be provided for higher-quality results.
    SEARCH_API_KEY: str = os.getenv("SEARCH_API_KEY", "")

    # --- Runtime behaviour ---------------------------------------------------
    MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

    # HTTP headers OpenRouter recommends for attribution / rankings.
    APP_NAME: str = os.getenv("APP_NAME", "ai-personal-assistant")
    APP_URL: str = os.getenv("APP_URL", "https://github.com/your-username/ai-personal-assistant")

    def validate(self) -> None:
        """
        Fail fast if required configuration is missing.

        WHY THIS EXISTS: it is far better for the program to refuse to start
        with a clear error message than to crash three tool-calls deep with a
        cryptic 401 Unauthorized error.
        """
        if not self.OPENROUTER_API_KEY:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and "
                "add your OpenRouter API key (https://openrouter.ai/keys)."
            )


# A single shared instance imported everywhere else in the project.
settings = Settings()
