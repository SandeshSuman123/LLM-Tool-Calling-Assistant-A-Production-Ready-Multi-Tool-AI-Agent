"""
agent/tool_registry.py
========================

WHY THIS FILE EXISTS
---------------------
The LLM needs to be told, on every request, exactly which tools exist and
what arguments they take (the "tools" list in the API call). Meanwhile the
executor needs a way to look up "the user picked calculator, which Python
function is that?". The Tool Registry is the single data structure that
answers BOTH questions.

Without a registry, you'd end up with a giant if/elif chain checking tool
names throughout the codebase — brittle and hard to extend. With a
registry, adding tool #7 means writing one new file and one new line here.

RESPONSIBILITY
---------------
- Import every tool module.
- Build a dict mapping tool name -> `RegisteredTool` (schema + Python
  function + Pydantic argument model).
- Expose `get_all_schemas()` for the LLM request, and `get_tool(name)`
  for the executor.

HOW IT INTERACTS WITH OTHER FILES
-----------------------------------
- `agent/llm.py` calls `get_all_schemas()` to attach to every chat request.
- `agent/tool_executor.py` calls `get_tool(name)` to dispatch a tool call.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Registry
✓ Modular Architecture (adding a tool never touches this file's logic,
  only its registration list)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel

from tools import calculator, converter, search, text_editor, time_tool, weather
from utils.errors import ToolNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegisteredTool:
    """
    Bundles everything the executor needs to run one tool call:
    the human-readable name, its Pydantic argument model (for validation),
    its Python function, and the JSON-Schema definition sent to the LLM.
    """

    name: str
    args_model: Type[BaseModel]
    run_fn: Callable[[BaseModel], Any]
    schema: Dict[str, Any]


def _build_registry() -> Dict[str, RegisteredTool]:
    """
    Construct the registry dict at import time.

    WHY A FUNCTION (rather than inlining at module scope): keeps the
    "list of tools" declaration in one clearly readable place, and makes
    it trivial to unit test registry construction in isolation.

    TO ADD A NEW TOOL:
        1. Create `tools/your_tool.py` following the ArgsModel / run() /
           TOOL_SCHEMA contract used by every other tool in this package.
        2. Import it above and add ONE line below.
    """
    modules = [calculator, weather, search, text_editor, time_tool, converter]

    registry: Dict[str, RegisteredTool] = {}
    for module in modules:
        name = module.TOOL_SCHEMA["function"]["name"]
        registry[name] = RegisteredTool(
            name=name,
            args_model=_infer_args_model(module),
            run_fn=module.run,
            schema=module.TOOL_SCHEMA,
        )
        logger.debug("Registered tool: %s", name)

    return registry


def _infer_args_model(module) -> Type[BaseModel]:
    """
    Find the Pydantic `BaseModel` subclass defined in a tool module.

    WHY THIS EXISTS: rather than requiring every tool module to export a
    fixed variable name for its args model, we introspect the module for
    the one class that inherits from `BaseModel`. This keeps each tool
    file's public surface minimal (`run` + `TOOL_SCHEMA`) while still
    letting the registry recover strong typing for validation.
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
            return attr
    raise RuntimeError(f"No Pydantic BaseModel found in tool module: {module.__name__}")


# Built once at import time and reused for the lifetime of the process.
_REGISTRY: Dict[str, RegisteredTool] = _build_registry()


def get_all_schemas() -> List[Dict[str, Any]]:
    """Return the JSON-Schema tool definitions for every registered tool.

    This exact list is passed as the `tools=` parameter on every LLM
    request, telling the model what it is allowed to call.
    """
    return [tool.schema for tool in _REGISTRY.values()]


def get_tool(name: str) -> RegisteredTool:
    """
    Look up a registered tool by name.

    Raises `ToolNotFoundError` if the LLM hallucinates a tool name that
    was never registered — this is a real failure mode with smaller/free
    models, so the executor must handle it gracefully rather than crash.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ToolNotFoundError(f"No tool registered with name '{name}'.") from exc


def list_tool_names() -> List[str]:
    """Convenience helper, mostly used for logging/debugging/CLI help text."""
    return list(_REGISTRY.keys())
