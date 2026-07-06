"""
agent/streaming.py
=====================

WHY THIS FILE EXISTS
---------------------
When streaming is enabled, the API does NOT send one complete message —
it sends dozens/hundreds of small "chunks", each containing either a
fragment of assistant text OR a fragment of a tool call's arguments
(literally partial JSON, e.g. one chunk might contain `{"expr` and the
next `ession": "2+2"}`). This is exactly what "Fine-grained Tool Calling"
means: tool call construction happens incrementally, and something has to
buffer + reassemble the fragments before they're usable.

RESPONSIBILITY
---------------
- Consume the raw chunk iterator from `agent/llm.py`.
- Print assistant text to the console as it arrives (live streaming UX).
- Accumulate tool_call fragments (id, name, arguments) across chunks into
  complete, well-formed tool_call objects.
- Return a final, "flattened" result equivalent to what a non-streaming
  response would have given us, so the rest of the pipeline
  (`tool_executor.py`, `conversation.py`) doesn't need two separate code
  paths for streaming vs non-streaming.

CONCEPTS DEMONSTRATED
-----------------------
✓ Fine-grained Tool Calling / Streaming
✓ Message Content Blocks
✓ Error Handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

from rich.console import Console

from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class _ToolCallBuffer:
    """
    Accumulates fragments of ONE tool call across many streamed chunks.

    WHY THIS EXISTS: the API sends tool call arguments as a string that is
    built up incrementally, e.g. chunk 1 -> '{"exp', chunk 2 -> 'ression"',
    chunk 3 -> ': "2+2"}'. We must concatenate these fragments in order
    before the JSON is valid and can be parsed by the tool executor.
    """

    id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = field(default="")

    def to_namespace(self) -> SimpleNamespace:
        """
        Convert this buffer into an object with the same `.id`,
        `.type`, `.function.name`, `.function.arguments` shape that
        `agent/tool_executor.py` expects from a non-streaming response,
        so downstream code stays identical regardless of streaming mode.
        """
        return SimpleNamespace(
            id=self.id,
            type="function",
            function=SimpleNamespace(name=self.name, arguments=self.arguments),
        )


def stream_and_collect(chunk_iterator: Iterator[Any]) -> Dict[str, Any]:
    """
    Consume a stream of chunks, print text live, and return the final
    reassembled result.

    WHY THIS FUNCTION EXISTS: this is the single reassembly point that
    turns "many small deltas" into "one complete assistant turn" —
    exactly the transformation needed to bridge fine-grained streaming
    chunks back into the same message shape the rest of the app already
    understands (see `agent/message_handler.build_assistant_message`).

    Returns a dict: {"content": str | None, "tool_calls": list[SimpleNamespace]}
    """
    full_text = ""
    tool_call_buffers: Dict[int, _ToolCallBuffer] = {}

    for chunk in chunk_iterator:
        choice = chunk.choices[0]
        delta = choice.delta

        # --- Case 1: a fragment of normal assistant text -----------------
        if delta.content:
            console.print(delta.content, end="")
            full_text += delta.content

        # --- Case 2: a fragment of one or more tool calls -----------------
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index
                buffer = tool_call_buffers.setdefault(index, _ToolCallBuffer())

                if tool_call_delta.id:
                    buffer.id = tool_call_delta.id
                if tool_call_delta.function and tool_call_delta.function.name:
                    buffer.name = tool_call_delta.function.name
                if tool_call_delta.function and tool_call_delta.function.arguments:
                    buffer.arguments += tool_call_delta.function.arguments

    if full_text:
        console.print()  # newline after streamed text

    tool_calls: List[SimpleNamespace] = [
        buffer.to_namespace() for buffer in tool_call_buffers.values()
    ]

    logger.debug(
        "Stream collection complete: %d chars of text, %d tool call(s)",
        len(full_text),
        len(tool_calls),
    )

    return {"content": full_text or None, "tool_calls": tool_calls or None}
