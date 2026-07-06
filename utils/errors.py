"""
utils/errors.py
================

WHY THIS FILE EXISTS
---------------------
When an LLM calls a tool, MANY things can go wrong: bad arguments from the
model, a network timeout, a tool that doesn't exist, a division by zero
inside the calculator. If we let raw Python exceptions bubble straight into
the LLM conversation, we get two problems:

  1. The program crashes instead of continuing the conversation.
  2. The model never learns *why* the call failed, so it can't retry
     intelligently.

RESPONSIBILITY
---------------
Define a small hierarchy of custom exceptions that tools and the executor
raise deliberately. The tool executor catches these and converts them into
a structured error message that gets sent back to the LLM as a normal tool
result — so the model can see the error and decide what to do next
(e.g. ask the user for clarification, or retry with corrected arguments).

CONCEPTS DEMONSTRATED
-----------------------
✓ Error Handling
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for all tool-related errors in this project."""


class ToolNotFoundError(ToolError):
    """Raised when the LLM requests a tool name that isn't registered."""


class ToolArgumentError(ToolError):
    """
    Raised when a tool receives arguments that fail validation
    (e.g. a Pydantic ValidationError), or arguments the model
    hallucinated that don't match the tool's schema.
    """


class ToolExecutionError(ToolError):
    """
    Raised when a tool's underlying logic fails at runtime
    (e.g. a network request times out, an API returns a 500,
    a math expression divides by zero).
    """


class LLMRequestError(Exception):
    """Raised when the call to the OpenRouter/LLM API itself fails."""
