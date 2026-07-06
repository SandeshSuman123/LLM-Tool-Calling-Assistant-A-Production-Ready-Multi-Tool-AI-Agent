"""
agent/conversation.py
========================

WHY THIS FILE EXISTS
---------------------
This is the ORCHESTRATOR — the file that ties the LLM client, the tool
registry, the tool executor, the message handler, and the streaming
handler together into one coherent "agent loop". Every other module in
`agent/` does ONE job; this module decides the ORDER in which those jobs
happen for a full user turn.

THE CORE LOOP (this is what "agentic tool calling" actually is):

    User
      │
      ▼
    Send full message history + tool schemas to the LLM
      │
      ▼
    Does the response contain tool_calls?
      │
      ├── NO  ──> It's a final answer. Show it. Done.
      │
      └── YES ──> Execute every tool call (agent/tool_executor.py)
                    │
                    ▼
                  Append the tool results to message history
                    │
                    ▼
                  Send the UPDATED history back to the LLM
                    │
                    ▼
                  (loop back to "Does the response contain tool_calls?")

This loop can repeat several times in a row (e.g. the model calls
`get_weather`, reads the result, then decides it ALSO needs `unit_converter`
before it can answer) — that's why we cap iterations with
`MAX_TOOL_ITERATIONS` to guard against infinite loops with a misbehaving
model.

RESPONSIBILITY
---------------
- Own the conversation's message history (a simple Python list).
- Run the agent loop described above for each user turn.
- Decide whether to use streaming or non-streaming based on config.

CONCEPTS DEMONSTRATED
-----------------------
✓ Multi-turn Conversations
✓ Tool Calling
✓ Multiple Tool Calling / Batch Tool Calling
✓ Fine-grained Tool Calling / Streaming
✓ Sending Tool Results Back
✓ Modular Architecture
✓ Error Handling
✓ Logging
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from agent.llm import chat_completion, stream_chat_completion
from agent.message_handler import (
    build_assistant_message,
    build_system_message,
    build_user_message,
)
from agent.streaming import stream_and_collect
from agent.tool_executor import execute_tool_calls
from agent.tool_registry import get_all_schemas
from config import settings
from utils.errors import LLMRequestError
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful, precise AI personal assistant with access to tools:
a calculator, a weather lookup, a web search, a text editor, a current-time
lookup, and a unit converter.

Rules:
- If a question requires real-time data, precise arithmetic, or a
  deterministic transformation, ALWAYS use the appropriate tool instead
  of guessing.
- If multiple tools are needed to answer one question, call all of them.
- If a tool returns an error, explain the problem to the user in plain
  language instead of retrying blindly.
- Otherwise, answer directly and conversationally.
"""


class ConversationManager:
    """
    Owns one conversation's message history and drives the agent loop.

    WHY A CLASS (rather than free functions + a global list): a real
    application might run multiple independent conversations at once
    (e.g. a FastAPI server handling many users). Wrapping state in a
    class makes each conversation's history independent and testable.
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = [build_system_message(_SYSTEM_PROMPT)]
        self.tool_schemas = get_all_schemas()
        logger.info("ConversationManager initialized with %d tools", len(self.tool_schemas))

    def send_user_message(self, user_text: str) -> str:
        """
        Process one full user turn and return the assistant's final reply.

        WHY THIS FUNCTION EXISTS: this is the ONE public method the rest
        of the app (main.py) needs to call. Everything about tool-calling,
        streaming, and multi-turn looping is an internal implementation
        detail hidden behind this simple `str -> str` interface.
        """
        self.messages.append(build_user_message(user_text))

        for iteration in range(1, settings.MAX_TOOL_ITERATIONS + 1):
            logger.info("Agent loop iteration %d/%d", iteration, settings.MAX_TOOL_ITERATIONS)

            assistant_reply = self._get_assistant_reply()

            # Echo the assistant's turn (text and/or tool_calls) into
            # history BEFORE processing tool calls — the API requires the
            # assistant message that REQUESTED the tools to be present in
            # history before the corresponding tool-result messages.
            self.messages.append(
                build_assistant_message(
                    content=assistant_reply["content"],
                    tool_calls=assistant_reply["tool_calls"],
                )
            )

            if not assistant_reply["tool_calls"]:
                # No tool calls => the model has given its final answer.
                return assistant_reply["content"] or ""

            # The model wants to call one or more tools. Execute them all,
            # then loop back and send the results to the model.
            tool_result_messages = execute_tool_calls(assistant_reply["tool_calls"])
            self.messages.extend(tool_result_messages)

        logger.warning("Max tool iterations (%d) reached without a final answer", settings.MAX_TOOL_ITERATIONS)
        return (
            "I attempted several tool calls but couldn't reach a final answer. "
            "Could you rephrase your question?"
        )

    def _get_assistant_reply(self) -> Dict[str, Any]:
        """
        Get one assistant turn from the LLM, in streaming or non-streaming
        mode depending on configuration, normalized to the same shape:
        `{"content": str | None, "tool_calls": list | None}`.

        WHY NORMALIZE HERE: the rest of the agent loop (`send_user_message`)
        should not need to know or care whether streaming is on — this is
        the one seam where that difference is absorbed.
        """
        try:
            if settings.ENABLE_STREAMING:
                chunk_iterator = stream_chat_completion(self.messages, tools=self.tool_schemas)
                return stream_and_collect(chunk_iterator)

            response = chat_completion(self.messages, tools=self.tool_schemas)
            message = response.choices[0].message
            return {"content": message.content, "tool_calls": message.tool_calls}

        except LLMRequestError as exc:
            logger.error("LLM request failed: %s", exc)
            # Return a graceful, non-crashing "final answer" so the CLI
            # loop in main.py can continue instead of dying entirely.
            return SimpleNamespace(content=f"[Error contacting the model: {exc}]", tool_calls=None).__dict__
