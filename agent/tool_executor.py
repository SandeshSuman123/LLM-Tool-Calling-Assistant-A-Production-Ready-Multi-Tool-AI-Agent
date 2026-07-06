"""
agent/tool_executor.py
=========================

WHY THIS FILE EXISTS
---------------------
When the LLM decides to call a tool, it doesn't run any code itself — it
just emits a "tool_call" object containing a tool name and a JSON string
of arguments. Something on OUR side has to:

    1. Parse that JSON.
    2. Validate it against the tool's expected schema.
    3. Actually invoke the Python function.
    4. Catch any errors and turn them into a message the LLM can read.

That "something" is the Tool Executor — the bridge between "the model
said to call calculator" and "calculator.run() actually executed".

RESPONSIBILITY
---------------
- Execute a SINGLE tool call (`execute_tool_call`).
- Execute MULTIPLE tool calls from one LLM turn, one after another
  (`execute_tool_calls`) — this is what "Multiple Tool Calling" and
  "Batch Tool Calling" mean in practice: a single assistant turn can
  request several tools at once (e.g. weather AND time), and we must
  execute all of them and return one tool-result message per call.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- Called by `agent/conversation.py` after the LLM response contains
  `tool_calls`.
- Uses `agent/tool_registry.py` to resolve tool name -> Python function.
- Uses `agent/message_handler.py`'s `build_tool_result_message` to format
  each tool's output back into the OpenAI chat-message shape.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Calling
✓ Sending Tool Results Back
✓ Multiple Tool Calling
✓ Batch Tool Calling
✓ Error Handling
✓ Logging
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import ValidationError

from agent.message_handler import build_tool_result_message
from agent.schemas import ToolResult
from agent.tool_registry import get_tool
from utils.errors import ToolArgumentError, ToolError, ToolNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


def execute_tool_call(tool_call: Any) -> Dict[str, str]:
    """
    Execute ONE tool call object as returned by the OpenAI/OpenRouter API.

    A `tool_call` object looks like:

        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression": "2 + 2"}'
            }
        }

    WHY THIS FUNCTION EXISTS: this is the innermost, single-responsibility
    unit of tool execution — parse arguments, validate, run, format result.
    Keeping it separate from the "loop over many tool calls" logic below
    makes each piece independently testable.

    Returns a dict already shaped as a `tool` role message, ready to be
    appended to the conversation.
    """
    call_id = tool_call.id
    tool_name = tool_call.function.name
    raw_arguments = tool_call.function.arguments

    logger.info("Executing tool call id=%s name=%s args=%s", call_id, tool_name, raw_arguments)

    try:
        registered_tool = get_tool(tool_name)

        try:
            parsed_arguments: Dict[str, Any] = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(
                f"Tool '{tool_name}' received malformed JSON arguments: {exc}"
            ) from exc

        try:
            validated_args = registered_tool.args_model(**parsed_arguments)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"Arguments for tool '{tool_name}' failed validation: {exc}"
            ) from exc

        result: ToolResult = registered_tool.run_fn(validated_args)

    except ToolNotFoundError as exc:
        logger.warning("Tool not found: %s", exc)
        result = ToolResult(success=False, error=str(exc))
    except ToolArgumentError as exc:
        logger.warning("Tool argument error: %s", exc)
        result = ToolResult(success=False, error=str(exc))
    except ToolError as exc:
        logger.error("Tool execution error: %s", exc)
        result = ToolResult(success=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        # WHY A BROAD CATCH HERE: a bug inside any individual tool must
        # NEVER crash the whole assistant. We log the full traceback for
        # developers but return a clean error string to the LLM so the
        # conversation can continue gracefully.
        logger.exception("Unexpected error while executing tool '%s'", tool_name)
        result = ToolResult(success=False, error=f"Unexpected internal error: {exc}")

    return build_tool_result_message(tool_call_id=call_id, content=result.to_content_string())


def execute_tool_calls(tool_calls: List[Any]) -> List[Dict[str, str]]:
    """
    Execute MULTIPLE tool calls requested within a single assistant turn.

    WHY THIS FUNCTION EXISTS: models frequently request more than one tool
    in a single turn — e.g. "What's the weather in Kathmandu and what's
    23 * 4?" triggers both `get_weather` and `calculator` simultaneously.
    This is "Multiple Tool Calling". When the SAME tool is called several
    times with different arguments in one turn (e.g. weather for three
    different cities), that's "Batch Tool Calling" — this function handles
    both cases identically, since each call is independent.

    Every tool call MUST get exactly one corresponding tool-result message
    with a matching `tool_call_id`, or the next LLM request will be
    rejected by the API as malformed.
    """
    logger.info("Executing %d tool call(s) from this turn", len(tool_calls))
    return [execute_tool_call(tool_call) for tool_call in tool_calls]
