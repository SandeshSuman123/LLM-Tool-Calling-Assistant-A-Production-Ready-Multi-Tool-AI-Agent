"""
agent/message_handler.py
===========================

WHY THIS FILE EXISTS
---------------------
The OpenAI/OpenRouter chat API represents a conversation as a LIST OF
MESSAGE DICTS, each with a `role` ("system" / "user" / "assistant" /
"tool") and some form of `content`. Assistant messages that call tools
also carry a `tool_calls` field, and tool-result messages must carry a
`tool_call_id` linking them back to the specific call they answer.

Rather than scattering `{"role": "...", "content": "..."}` dict literals
throughout the codebase (typo-prone, inconsistent), this module is the
ONE place that knows the exact shape of every message type.

RESPONSIBILITY
---------------
Provide small, well-named builder functions for each message role:
`build_system_message`, `build_user_message`, `build_assistant_message`,
`build_tool_result_message`.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- `agent/conversation.py` uses these builders to grow the conversation
  history turn by turn.
- `agent/tool_executor.py` uses `build_tool_result_message` to format
  each tool's output before it goes back to the LLM.

CONCEPTS DEMONSTRATED
-----------------------
✓ Message Content Blocks
✓ Sending Tool Results Back
✓ Multi-turn Conversations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_system_message(content: str) -> Dict[str, str]:
    """
    Build the system message that seeds every conversation.

    WHY THIS EXISTS AS ITS OWN FUNCTION: the system prompt is only ever
    set once, at conversation start — giving it a dedicated builder makes
    that "exactly once" intent explicit in `conversation.py`.
    """
    return {"role": "system", "content": content}


def build_user_message(content: str) -> Dict[str, str]:
    """Build a standard user turn."""
    return {"role": "user", "content": content}


def build_assistant_message(
    content: Optional[str],
    tool_calls: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Build an assistant message, optionally carrying tool call requests.

    WHY `tool_calls` MATTERS: when the model decides to call one or more
    tools instead of replying in plain text, `content` is often `None`
    and the actual "message content" is a list of tool_call objects
    (this is what "Message Content Blocks" refers to in the OpenAI/
    Anthropic tool-calling model — a message is not just a string, it is
    made of typed content blocks). We must echo this exact assistant
    message back into the conversation history BEFORE appending tool
    results, or the API will reject the next request as out of order.
    """
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        # Convert SDK tool_call objects to plain dicts so this message can
        # be safely appended to the message list sent on the NEXT request.
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return message


def build_tool_result_message(tool_call_id: str, content: str) -> Dict[str, str]:
    """
    Build a `tool` role message carrying one tool's result back to the LLM.

    WHY `tool_call_id` MATTERS: when multiple tools are called in one turn
    ("Multiple Tool Calling" / "Batch Tool Calling"), the model needs to
    know WHICH result belongs to WHICH call. The `tool_call_id` is the
    only thing that links a tool result back to its originating request.
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
