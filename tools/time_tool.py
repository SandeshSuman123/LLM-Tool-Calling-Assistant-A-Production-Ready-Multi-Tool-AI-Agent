"""
tools/time_tool.py
====================

WHY THIS FILE EXISTS
---------------------
LLMs do not have a real-time clock — they only know "now" as whatever
their training data implied. Any question like "what time is it in Tokyo
right now?" REQUIRES a tool call. This is one of the simplest possible
tools and a great first example for beginners.

RESPONSIBILITY
---------------
Return the current date/time, optionally for a specific IANA timezone
(e.g. "Asia/Kathmandu", "America/New_York").

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas (optional parameter with a default)
✓ Error Handling (invalid timezone names)
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeArgs(BaseModel):
    """Arguments for the current-time tool."""

    timezone: str = Field(
        "UTC",
        description=(
            "IANA timezone name, e.g. 'Asia/Kathmandu', 'America/New_York', "
            "'UTC'. Defaults to UTC if not specified."
        ),
    )


def run(args: TimeArgs) -> ToolResult:
    """
    Return the current date and time for the requested timezone.
    """
    logger.info("Time tool invoked for timezone=%r", args.timezone)
    try:
        tz = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError:
        return ToolResult(success=False, error=f"Unknown timezone: '{args.timezone}'.")

    now = datetime.now(tz)
    return ToolResult(
        success=True,
        data={
            "timezone": args.timezone,
            "iso_format": now.isoformat(),
            "human_readable": now.strftime("%A, %d %B %Y, %I:%M %p"),
        },
    )


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="get_current_time",
    description=(
        "Get the current date and time, optionally for a specific IANA "
        "timezone. Use this whenever the user asks what time or date it is."
    ),
    args_model=TimeArgs,
)
