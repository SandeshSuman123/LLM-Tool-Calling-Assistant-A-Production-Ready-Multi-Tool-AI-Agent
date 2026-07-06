"""
utils/logger.py
================

WHY THIS FILE EXISTS
---------------------
Tool-calling systems are hard to debug with `print()` statements alone,
because a single user turn can trigger multiple LLM calls, multiple tool
calls, and multiple round-trips. A structured logger lets us trace exactly
what happened, in what order, without cluttering the user-facing chat
output.

RESPONSIBILITY
---------------
Configure one shared, project-wide logger (using Rich for pretty console
output when available, falling back to plain logging otherwise).

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
Every module in `agent/` and `tools/` imports `get_logger(__name__)` from
here instead of configuring its own logging — this keeps log formatting
consistent across the whole project.

CONCEPTS DEMONSTRATED
-----------------------
✓ Logging
✓ Configuration Management (LOG_LEVEL is read from config/env)
"""

from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich is optional
    _HAS_RICH = False

from config import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    """
    Set up the root logger exactly once.

    WHY: calling `logging.basicConfig()` multiple times (e.g. once per
    module import) is a common beginner mistake — only the first call has
    any effect, which leads to confusing "why isn't my log level working"
    bugs. We guard with a module-level flag so configuration happens once.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if _HAS_RICH:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Tool call started: %s", tool_name)
    """
    _configure_root_logger()
    return logging.getLogger(name)
