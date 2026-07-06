
# AI Personal Assistant — A Complete Tool-Calling Reference Project
=======
# LLM-Tool-Calling-Assistant-A-Production-Ready-Multi-Tool-AI-Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Made with Pydantic](https://img.shields.io/badge/Validation-Pydantic-e92063?logo=pydantic&logoColor=white)
>>>>>>> 57b0b7b318c32a64fdb3133f9a47335426b95409

A production-quality, fully working Python implementation of an LLM-powered
personal assistant that demonstrates **every major concept of LLM tool
calling** — built on [OpenRouter](https://openrouter.ai) so it runs entirely
on **free models** (Qwen, Llama, Gemma, DeepSeek, etc.).

This project is designed to be **studied line by line**, extended, put on
GitHub, and referenced in interviews.


=======
> 💡 Built as a hands-on deep dive into agentic tool-calling: schemas,
> registries, dispatchers, multi-turn state, and streaming — the same
> primitives used under the hood by LangChain, the OpenAI Assistants API,
> and Anthropic's tool-use API.

>>>>>>> 57b0b7b318c32a64fdb3133f9a47335426b95409
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Environment Variables](#environment-variables)
5. [How Tool Calling Works](#how-tool-calling-works)
6. [How to Run](#how-to-run)
7. [Example Prompts](#example-prompts)
8. [Example Tool Calls](#example-tool-calls)
9. [How to Add a New Tool](#how-to-add-a-new-tool)
10. [Folder Structure](#folder-structure)
11. [Concepts Demonstrated (Checklist)](#concepts-demonstrated-checklist)
12. [Future Improvements](#future-improvements)

---

## Project Overview

The assistant is a command-line chat program that automatically decides,
turn by turn, whether it can answer directly or whether it needs to call
one of six tools:

| Tool              | Purpose                                             | External dependency        |
|-------------------|------------------------------------------------------|-----------------------------|
| `calculator`      | Exact arithmetic evaluation                          | None (safe AST evaluator)  |
| `get_weather`     | Real-time weather for a city                         | Open-Meteo (free, no key)  |
| `web_search`      | Current-events / general web search                  | DuckDuckGo HTML (free)     |
| `text_editor`     | Deterministic text transforms (upper/lower/reverse…) | None                        |
| `get_current_time`| Current date/time in any IANA timezone               | None (stdlib `zoneinfo`)   |
| `unit_converter`  | Length / weight / temperature conversion              | None                        |

The model itself decides — per user message — which of these tools (zero,
one, or several at once) it needs before producing a final answer.

---

## Architecture

The codebase is split into clearly separated layers, each with a single
responsibility:

```
                       ┌───────────────────────┐
                       │        main.py        │   CLI entry point / REPL
                       └───────────┬───────────┘
                                   │
                       ┌───────────▼───────────┐
                       │ agent/conversation.py  │   Orchestrates the agent loop
                       └───────────┬───────────┘
                 ┌─────────────────┼──────────────────┐
                 │                 │                  │
       ┌─────────▼───────┐ ┌───────▼────────┐ ┌───────▼─────────┐
       │  agent/llm.py    │ │ agent/message_  │ │ agent/streaming.py│
       │ (talks to        │ │  handler.py     │ │ (reassembles      │
       │  OpenRouter)     │ │ (builds message │ │  streamed chunks) │
       │                  │ │  dicts)         │ │                   │
       └─────────┬────────┘ └────────────────┘ └───────────────────┘
                 │
       ┌─────────▼─────────────┐
       │ agent/tool_executor.py│  Validates args, runs tool, formats result
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │ agent/tool_registry.py│  Maps tool name -> schema + Python function
       └─────────┬─────────────┘
                 │
       ┌─────────▼─────────────┐
       │        tools/         │  calculator, weather, search, text_editor,
       │                       │  time_tool, converter
       └────────────────────────┘
```

### The Core Agent Loop

```
User
  │
  ▼
Send full message history + tool schemas to the LLM
  │
  ▼
Does the response contain tool_calls?
  │
  ├── NO  ──▶ It's a final answer. Print it. Done.
  │
  └── YES ──▶ Execute every requested tool call
                │
                ▼
              Append tool results to message history
                │
                ▼
              Send the UPDATED history back to the LLM
                │
                ▼
              (loop back to "Does the response contain tool_calls?")
```

This loop is capped at `MAX_TOOL_ITERATIONS` (default `6`) to protect
against a misbehaving model looping forever.

### Multiple Tool Calling

A single user message can trigger more than one tool in the same turn:

```
"What's the weather in Kathmandu, and what's 23 * 4?"
                    │
                    ▼
        LLM emits TWO tool_calls in one response:
        [ get_weather(city="Kathmandu"), calculator(expression="23*4") ]
                    │
                    ▼
        tool_executor.execute_tool_calls() runs BOTH
                    │
                    ▼
        TWO tool-result messages appended (matched by tool_call_id)
                    │
                    ▼
        LLM combines both results into one final natural-language answer
```

### Batch Tool Calling

The **same** tool called several times in one turn (e.g. weather for three
cities) is handled by the exact same code path above — each call is
independent and gets its own `tool_call_id` and its own result message.

### Streaming / Fine-grained Tool Calling

```
LLM response arrives as many small chunks:

 chunk 1: delta.content = "The "
 chunk 2: delta.content = "weather "
 chunk 3: delta.tool_calls = [{index:0, function:{name:"get_weather"}}]
 chunk 4: delta.tool_calls = [{index:0, function:{arguments:'{"city":'}}]
 chunk 5: delta.tool_calls = [{index:0, function:{arguments:'"Kathmandu"}'}}]
                    │
                    ▼
        agent/streaming.py buffers and concatenates these fragments
                    │
                    ▼
        Once the stream ends, a complete tool_call object is reconstructed
        and handed to the exact same tool_executor used for non-streaming
```

### Conversation Flow (Multi-turn)

```
[system prompt]
[user: "What's the weather in Jodhpur?"]
[assistant: tool_calls=[get_weather(city="Jodhpur")]]
[tool: {"success": true, "result": {...}}]
[assistant: "It's currently 34°C and clear in Jodhpur."]
[user: "Convert that to Fahrenheit"]
[assistant: tool_calls=[unit_converter(value=34, from_unit=celsius, to_unit=fahrenheit)]]
[tool: {"success": true, "result": {...}}]
[assistant: "That's 93.2°F."]
```

Notice the SECOND question ("Convert that to Fahrenheit") only makes sense
because the full message history — including the earlier tool result — is
sent on every request. This is what "Multi-turn Conversations" means in
practice.

---

## Installation

```bash
git clone https://github.com/your-username/ai-personal-assistant.git
cd ai-personal-assistant

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your OPENROUTER_API_KEY
```

Get a free OpenRouter API key at **https://openrouter.ai/keys**.

---

## Environment Variables

All configuration lives in `.env` (see `.env.example` for the full list
with comments). The most important ones:

| Variable               | Required | Default                                  | Purpose                                   |
|-------------------------|----------|-------------------------------------------|--------------------------------------------|
| `OPENROUTER_API_KEY`   | ✅ Yes   | —                                          | Your OpenRouter API key                    |
| `MODEL_NAME`           | No       | `qwen/qwen-2.5-72b-instruct:free`         | Any OpenRouter model ID that supports tools|
| `ENABLE_STREAMING`     | No       | `true`                                     | Toggle streaming vs single-shot responses  |
| `MAX_TOOL_ITERATIONS`  | No       | `6`                                        | Safety cap on the agent loop               |
| `LOG_LEVEL`            | No       | `INFO`                                     | `DEBUG` / `INFO` / `WARNING` / `ERROR`     |

`config.py` reads and validates all of these — the program refuses to
start with a clear error message if `OPENROUTER_API_KEY` is missing.

---

## How Tool Calling Works

1. **Define each tool's arguments as a Pydantic model** (`tools/*.py`).
   Pydantic auto-generates JSON Schema from the model, so the tool's
   contract is defined once and used both for LLM-facing schema AND for
   validating whatever arguments the model actually sends back.

2. **Register every tool** in `agent/tool_registry.py`, which builds a
   dict of `tool name -> {schema, Python function, Pydantic model}`.

3. **Attach all tool schemas to every LLM request** (`agent/llm.py`,
   `tools=get_all_schemas()`). This is what tells the model "here are
   your available actions."

4. **Inspect the response.** If `message.tool_calls` is non-empty, the
   model wants to invoke one or more tools instead of answering directly.

5. **Execute each tool call** (`agent/tool_executor.py`): parse the JSON
   arguments, validate them against the Pydantic model, run the Python
   function, and catch any errors so a bad tool call never crashes the
   whole assistant.

6. **Send tool results back** as `role: "tool"` messages, each tagged with
   the `tool_call_id` of the call it answers (`agent/message_handler.py`).

7. **Loop.** Send the updated history back to the model. It may call more
   tools, or produce a final natural-language answer.

---

## How to Run

```bash
python main.py
```

Example session:

```
╭────────────── AI Personal Assistant ──────────────╮
│ Model: qwen/qwen-2.5-72b-instruct:free            │
│ Tools: calculator, get_weather, web_search, ...    │
│ Type your question, or 'exit' to quit.             │
╰─────────────────────────────────────────────────────╯

You: What's 18% of 2450, and what's the weather in Kathmandu?
Assistant: 18% of 2450 is 441. In Kathmandu right now it's 27°C with
partly cloudy skies and light wind.

You: exit
Goodbye!
```

---

## Example Prompts

- `"What's 23 * (17 + 4) / 2?"` → calculator
- `"What's the weather like in Jodhpur right now?"` → get_weather
- `"What time is it in Tokyo?"` → get_current_time
- `"Convert 5 miles to kilometers"` → unit_converter
- `"Reverse this text: Hello World"` → text_editor
- `"What are the latest updates on the Claude API?"` → web_search
- `"What's the weather in Kathmandu AND what's 100°F in Celsius?"` →
  **multiple tool calling** (get_weather + unit_converter in one turn)

---

## Example Tool Calls

A raw tool call as emitted by the model (before execution):

```json
{
  "id": "call_9f89s",
  "type": "function",
  "function": {
    "name": "unit_converter",
    "arguments": "{\"value\": 100, \"from_unit\": \"fahrenheit\", \"to_unit\": \"celsius\"}"
  }
}
```

The tool result sent back to the model:

```json
{
  "role": "tool",
  "tool_call_id": "call_9f89s",
  "content": "{\"success\": true, \"result\": {\"input\": \"100.0 fahrenheit\", \"output\": \"37.777778 celsius\", \"value\": 37.777778}}"
}
```

---

## How to Add a New Tool

Adding a new tool takes three steps and never requires touching the
executor, conversation manager, or LLM client:

1. **Create `tools/your_tool.py`** following the same contract every
   existing tool uses:

   ```python
   from pydantic import BaseModel, Field
   from agent.schemas import ToolResult, pydantic_to_tool_schema

   class YourToolArgs(BaseModel):
       some_input: str = Field(..., description="Describe this clearly for the LLM.")

   def run(args: YourToolArgs) -> ToolResult:
       # your logic here
       return ToolResult(success=True, data="...")

   TOOL_SCHEMA = pydantic_to_tool_schema(
       name="your_tool",
       description="One clear sentence describing when to use this tool.",
       args_model=YourToolArgs,
   )
   ```

2. **Register it** in `agent/tool_registry.py`:
   ```python
   from tools import calculator, converter, search, text_editor, time_tool, weather, your_tool
   modules = [calculator, weather, search, text_editor, time_tool, converter, your_tool]
   ```

3. **Done.** The registry automatically picks up its schema and Python
   function — no other file needs to change.

---

## Folder Structure

```
project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                  # CLI entry point / REPL
├── config.py                # Environment variables & settings
├── agent/
│   ├── __init__.py
│   ├── conversation.py       # Orchestrates the full agent loop
│   ├── llm.py                # OpenRouter/OpenAI SDK client wrapper
│   ├── tool_executor.py      # Validates + runs tool calls
│   ├── tool_registry.py      # Catalog of all tools
│   ├── message_handler.py    # Builds chat message dicts
│   ├── streaming.py          # Reassembles streamed chunks
│   └── schemas.py            # Pydantic -> JSON Schema helpers
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── weather.py
│   ├── search.py
│   ├── text_editor.py
│   ├── time_tool.py
│   └── converter.py
└── utils/
    ├── __init__.py
    ├── logger.py
    └── errors.py
```

---

## Concepts Demonstrated (Checklist)

| Concept                              | Where implemented                                   |
|----------------------------------------|--------------------------------------------------------|
| Tool Functions                        | `tools/*.py` (`run()` in each file)                   |
| Tool Schemas (JSON Schema)             | `agent/schemas.py`, `tools/*.py` (`TOOL_SCHEMA`)      |
| Tool Calling                           | `agent/llm.py` (`tools=` param), `agent/conversation.py` |
| Message Content Blocks                 | `agent/message_handler.py` (`tool_calls` on assistant messages) |
| Sending Tool Results Back              | `agent/message_handler.build_tool_result_message`      |
| Multi-turn Conversations                | `agent/conversation.ConversationManager`               |
| Multiple Tool Calling                   | `agent/tool_executor.execute_tool_calls`               |
| Batch Tool Calling                      | Same function — arbitrary number of calls per turn     |
| Fine-grained Tool Calling / Streaming   | `agent/streaming.py`                                    |
| Tool Registry                          | `agent/tool_registry.py`                                |
| Tool Dispatcher                        | `agent/tool_executor.py`                                 |
| Modular Architecture                   | Whole `agent/` + `tools/` split                          |
| Error Handling                         | `utils/errors.py`, try/except in every tool + executor  |
| Logging                                | `utils/logger.py`, used throughout                       |
| Environment Variables                  | `config.py`, `.env.example`                              |
| Configuration Management               | `config.Settings`                                        |

### What You Learned

- **Why this module exists:** tool calling turns an LLM from a "text
  predictor" into an **agent** that can take real actions and fetch real
  data — the model decides *when* to act, your code decides *how*.
- **Common interview questions:** "How do you validate LLM-generated tool
  arguments?" (Pydantic), "How do you handle a model calling multiple
  tools at once?" (loop + match by `tool_call_id`), "How does streaming
  interact with tool calls?" (arguments arrive as partial JSON fragments
  that must be reassembled).
- **Common beginner mistakes:** using raw `eval()` for a calculator tool
  (security risk — this project uses a whitelisted AST evaluator
  instead); forgetting to append the assistant's tool-call message to
  history before appending tool results (the API will reject the next
  request); not capping the tool-calling loop (risk of infinite loops).
- **Real-world production usage:** this exact pattern — registry, schema
  validation, executor, loop — is how frameworks like LangChain, the
  OpenAI Assistants API, and Anthropic's tool-use API all work internally,
  just with more abstraction layers on top.
- **Anthropic implementation:** Claude's Messages API uses `tools=[...]`
  with a similar JSON-Schema `input_schema`, and returns `tool_use`
  content blocks; results are sent back as `tool_result` content blocks
  in a `user` message.
- **OpenAI implementation:** Chat Completions uses `tools=[...]` with
  `function.parameters` JSON Schema, returns `tool_calls` on the assistant
  message; results go back as `role: "tool"` messages (the exact shape
  this project uses).
- **OpenRouter implementation:** OpenRouter proxies the OpenAI-compatible
  format across many different model providers, so this same code works
  unmodified with Qwen, Llama, Gemma, DeepSeek, Mistral, and others —
  just change `MODEL_NAME`.

---

## Future Improvements

- Add a persistent conversation store (SQLite/Redis) so history survives
  restarts.
- Add a FastAPI wrapper to expose the assistant as an HTTP/WebSocket API.
- Add automatic retries with exponential backoff for transient network
  errors in `agent/llm.py`.
- Add unit tests (`pytest`) for every tool and for the tool executor's
  error-handling paths.
- Add a `memory` tool that lets the assistant persist facts about the
  user across sessions.
- Support parallel (concurrent) execution of independent tool calls in
<<<<<<< HEAD
  `execute_tool_calls` using `asyncio` for lower latency on batch calls.
=======
  `execute_tool_calls` using `asyncio` for lower latency on batch calls

