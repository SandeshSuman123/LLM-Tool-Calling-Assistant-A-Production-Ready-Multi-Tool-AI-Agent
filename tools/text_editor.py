"""
tools/text_editor.py
======================

WHY THIS FILE EXISTS
---------------------
Demonstrates a tool with an ENUM-constrained argument (an "operation" the
model must pick from a fixed list) rather than free-form text — a very
common real-world tool-schema pattern (think "action" parameters in
Slack/Jira/Notion tool integrations).

RESPONSIBILITY
---------------
Apply simple, deterministic text transformations (uppercase, lowercase,
reverse, word count, title case, strip whitespace) to a piece of text
supplied by the model.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas (enum-constrained parameter)
✓ Error Handling
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from utils.logger import get_logger

logger = get_logger(__name__)


class TextOperation(str, Enum):
    """
    Fixed set of supported operations.

    WHY AN ENUM: Pydantic converts `Enum` fields into a JSON Schema `enum`
    list automatically, which constrains the LLM to only ever pass one of
    these exact values — eliminating an entire class of "invalid argument"
    errors compared to a free-text `operation: str` field.
    """

    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    TITLE_CASE = "title_case"
    REVERSE = "reverse"
    WORD_COUNT = "word_count"
    STRIP_WHITESPACE = "strip_whitespace"


class TextEditorArgs(BaseModel):
    """Arguments for the text editor tool."""

    text: str = Field(..., description="The input text to transform.")
    operation: TextOperation = Field(..., description="Which transformation to apply.")


def run(args: TextEditorArgs) -> ToolResult:
    """
    Apply `args.operation` to `args.text` and return the transformed value.
    """
    logger.info("Text editor tool invoked: operation=%s", args.operation)

    if args.operation == TextOperation.UPPERCASE:
        result = args.text.upper()
    elif args.operation == TextOperation.LOWERCASE:
        result = args.text.lower()
    elif args.operation == TextOperation.TITLE_CASE:
        result = args.text.title()
    elif args.operation == TextOperation.REVERSE:
        result = args.text[::-1]
    elif args.operation == TextOperation.WORD_COUNT:
        result = len(args.text.split())
    elif args.operation == TextOperation.STRIP_WHITESPACE:
        result = args.text.strip()
    else:  # pragma: no cover - unreachable due to Enum validation
        return ToolResult(success=False, error=f"Unsupported operation: {args.operation}")

    return ToolResult(success=True, data=result)


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="text_editor",
    description=(
        "Apply a text transformation (uppercase, lowercase, title_case, "
        "reverse, word_count, strip_whitespace) to a piece of text."
    ),
    args_model=TextEditorArgs,
)
