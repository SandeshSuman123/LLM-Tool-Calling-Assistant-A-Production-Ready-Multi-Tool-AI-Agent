"""tools package — every individual tool the assistant can call.

Each module here follows the exact same three-part contract:

    1. `ArgsModel`  — a Pydantic model describing the tool's arguments.
    2. `run(args)`  — the actual Python function that does the work.
    3. `TOOL_SCHEMA` — the JSON-Schema tool definition sent to the LLM.

This consistent contract is what makes the project's `tool_registry.py`
able to auto-discover and register every tool with almost no boilerplate,
and is what makes adding a brand-new tool a 5-minute job (see the
"How to Add a New Tool" section in README.md).
"""
