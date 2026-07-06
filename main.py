"""
main.py
=========

WHY THIS FILE EXISTS
---------------------
Every application needs a single, obvious entry point. This file wires
together configuration validation, the conversation manager, and a simple
command-line REPL (read-eval-print loop) — it deliberately contains almost
no "logic" of its own; it just orchestrates calls into `agent/`.

RESPONSIBILITY
---------------
- Validate configuration at startup (fail fast on missing API keys).
- Print a friendly welcome banner listing available tools.
- Run a REPL: read user input, hand it to `ConversationManager`, print
  the reply, repeat until the user exits.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
Imports `config.settings` (validated here) and `agent.conversation.
ConversationManager` (the whole system's public interface).

CONCEPTS DEMONSTRATED
-----------------------
✓ Multi-turn Conversations (the REPL keeps one ConversationManager alive
  across many turns)
✓ Configuration Management
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from agent.conversation import ConversationManager
from agent.tool_registry import list_tool_names
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()

_EXIT_COMMANDS = {"exit", "quit", ":q"}


def _print_welcome_banner() -> None:
    """Print a short banner describing the assistant and its tools."""
    tools = ", ".join(list_tool_names())
    console.print(
        Panel.fit(
            f"[bold cyan]AI Personal Assistant[/bold cyan]\n"
            f"Model: [green]{settings.MODEL_NAME}[/green]\n"
            f"Tools: [yellow]{tools}[/yellow]\n\n"
            f"Type your question, or '[bold]exit[/bold]' to quit.",
            border_style="cyan",
        )
    )


def main() -> None:
    """
    Entry point: validate config, print the banner, run the REPL.

    WHY VALIDATION HAPPENS FIRST: `settings.validate()` raises immediately
    if `OPENROUTER_API_KEY` is missing, so the user gets one clear error
    message instead of a confusing crash the first time a tool needs
    network access.
    """
    settings.validate()
    _print_welcome_banner()

    conversation = ConversationManager()

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in _EXIT_COMMANDS:
            console.print("[dim]Goodbye![/dim]")
            break

        console.print("[bold magenta]Assistant:[/bold magenta] ", end="")
        try:
            reply = conversation.send_user_message(user_input)
        except Exception as exc:  # noqa: BLE001 - top-level safety net for the REPL
            logger.exception("Unhandled error while processing user message")
            console.print(f"[red]An unexpected error occurred: {exc}[/red]")
            continue

        # When streaming is enabled, the reply text was already printed
        # live token-by-token by agent/streaming.py — only print it here
        # if streaming was OFF (otherwise we'd print it twice).
        if not settings.ENABLE_STREAMING:
            console.print(reply)


if __name__ == "__main__":
    main()
