"""
agent/llm.py
==============

WHY THIS FILE EXISTS
---------------------
Every other module in this project should be able to say "send these
messages to the model" without knowing or caring which SDK, base URL, or
HTTP details are involved. This file is the ONLY place that touches the
OpenAI SDK / OpenRouter directly.

WHY OPENROUTER + THE OPENAI SDK: OpenRouter exposes an API that is
byte-for-byte compatible with OpenAI's `/chat/completions` endpoint,
including function/tool calling. That means we can use the official
`openai` Python SDK, just pointed at a different `base_url`, and get
access to dozens of free and paid models (Qwen, DeepSeek, Llama, Gemma,
etc.) without writing our own HTTP client.

RESPONSIBILITY
---------------
- Construct a configured OpenAI-SDK client pointed at OpenRouter.
- Provide `chat_completion()` for a normal, single-shot request.
- Provide `stream_chat_completion()` for token-by-token streaming.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- `agent/conversation.py` calls `chat_completion()` (or the streaming
  variant) with the full message history + tool schemas from
  `agent/tool_registry.py`.
- `agent/streaming.py` consumes the generator from `stream_chat_completion()`
  to render partial output as it arrives.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Calling (the `tools=` parameter)
✓ Fine-grained Tool Calling / Streaming
✓ Error Handling
✓ Environment Variables / Configuration Management
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from openai import APIError, APITimeoutError, OpenAI

from config import settings
from utils.errors import LLMRequestError
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_client() -> OpenAI:
    """
    Construct the OpenAI SDK client, pointed at OpenRouter's base URL.

    WHY A SEPARATE FUNCTION: keeps client construction testable in
    isolation (e.g. you can monkeypatch this in unit tests) and documents,
    in one place, the exact headers OpenRouter recommends for attribution.
    """
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.BASE_URL,
        default_headers={
            # OpenRouter uses these optional headers to attribute usage to
            # your app on their public leaderboards / rankings.
            "HTTP-Referer": settings.APP_URL,
            "X-Title": settings.APP_NAME,
        },
    )


# A single shared client instance, built once at import time.
client = _build_client()


def chat_completion(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Send a normal (non-streaming) chat completion request.

    WHY THIS FUNCTION EXISTS: this is the core request/response primitive
    used for every non-streaming turn. Passing `tools=tools` is what
    enables "Tool Calling" — it tells the model which functions it is
    allowed to invoke and with what argument shapes (see
    `agent/tool_registry.get_all_schemas()`).

    Returns the raw SDK response object; `agent/conversation.py` inspects
    `.choices[0].message` to decide whether the model wants to call a
    tool or has produced a final answer.
    """
    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        return response
    except APITimeoutError as exc:
        logger.exception("LLM request timed out")
        raise LLMRequestError(f"The model took too long to respond: {exc}") from exc
    except APIError as exc:
        logger.exception("LLM API returned an error")
        raise LLMRequestError(f"The model API returned an error: {exc}") from exc


def stream_chat_completion(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Any]:
    """
    Send a STREAMING chat completion request and yield raw chunk objects.

    WHY THIS FUNCTION EXISTS: streaming lets the user see the assistant's
    text appear token-by-token instead of waiting for the full response —
    much better perceived latency. It also demonstrates "Fine-grained Tool
    Calling", where even tool_call arguments arrive incrementally as
    partial JSON fragments across multiple chunks, and must be
    reassembled before they can be parsed (see `agent/streaming.py`).

    This function is a thin, error-handled wrapper around the SDK's
    streaming iterator — all chunk-assembly logic lives in
    `agent/streaming.py` to keep this file focused purely on "talking to
    the API".
    """
    try:
        stream = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        yield from stream
    except APITimeoutError as exc:
        logger.exception("Streaming LLM request timed out")
        raise LLMRequestError(f"The model took too long to respond: {exc}") from exc
    except APIError as exc:
        logger.exception("Streaming LLM API returned an error")
        raise LLMRequestError(f"The model API returned an error: {exc}") from exc
