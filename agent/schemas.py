"""
agent/schemas.py
=================

WHY THIS FILE EXISTS
---------------------
Tool calling only works reliably if the LLM knows EXACTLY what arguments
each tool expects — their names, types, and which are required. That
description is called a "Tool Schema" (JSON Schema format). Rather than
hand-writing JSON Schema dictionaries for every tool (error-prone, easy to
get out of sync with the actual Python function), we define each tool's
arguments as a Pydantic model and auto-generate the JSON Schema from it.

RESPONSIBILITY
---------------
- Define one Pydantic `BaseModel` per tool, describing its arguments.
- Provide a helper, `pydantic_to_tool_schema`, that converts a Pydantic
  model + tool name/description into the exact dict shape the OpenAI /
  OpenRouter "tools" API expects.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- Each file in `tools/` defines its own Pydantic argument model (importing
  `BaseModel` directly) and a `TOOL_SCHEMA` built with the helper here.
- `agent/tool_registry.py` collects all these schemas into the final list
  that gets sent to the LLM on every request.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Schemas (JSON Schema)
✓ Modular Architecture
"""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel


def pydantic_to_tool_schema(
    name: str,
    description: str,
    args_model: Type[BaseModel],
) -> Dict[str, Any]:
    """
    Convert a Pydantic model into an OpenAI/OpenRouter-compatible "tool"
    definition.

    WHY THIS FUNCTION EXISTS: the OpenAI-style function-calling API expects
    a very specific dict shape:

        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": {<JSON Schema>}
            }
        }

    Pydantic already knows how to generate JSON Schema from a model via
    `.model_json_schema()`. This helper just wraps that output in the
    shape the LLM API expects, so tool authors never write raw JSON Schema
    by hand — they just write a normal Pydantic model.
    """
    schema = args_model.model_json_schema()

    # Pydantic emits a "title" key for the model itself; the LLM API doesn't
    # need it and it can confuse smaller models, so we strip it.
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


class ToolResult(BaseModel):
    """
    Standard shape returned by every tool's `run()` function.

    WHY THIS EXISTS: without a consistent return shape, the tool executor
    would need special-case handling for every tool. By making every tool
    return a `ToolResult`, the executor can serialize it to JSON uniformly
    before sending it back to the LLM as a tool result message.
    """

    success: bool
    data: Any = None
    error: str | None = None

    def to_content_string(self) -> str:
        """
        Serialize this result to a string suitable for a tool-role message.

        Tool result messages in the OpenAI/OpenRouter chat format must have
        string `content` — so even structured data gets JSON-encoded here.
        """
        import json

        if self.success:
            return json.dumps({"success": True, "result": self.data})
        return json.dumps({"success": False, "error": self.error})
