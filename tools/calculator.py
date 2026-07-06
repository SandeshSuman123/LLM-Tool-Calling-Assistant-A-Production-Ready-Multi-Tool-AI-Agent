"""
tools/calculator.py
=====================

WHY THIS FILE EXISTS
---------------------
LLMs are notoriously bad at precise arithmetic on large or complex numbers.
The calculator tool gives the model a reliable way to *delegate* math to
real code instead of guessing — one of the most common and most useful
real-world applications of tool calling.

RESPONSIBILITY
---------------
Safely evaluate a mathematical expression string (e.g. "23 * (17 + 4) / 2")
and return a numeric result.

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas
✓ Error Handling (safe evaluation, no `eval()` on raw untrusted input)
"""

from __future__ import annotations

import ast
import operator
from typing import Union

from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from utils.errors import ToolExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)

Number = Union[int, float]

# Only these AST node types / operators are permitted. This is what makes
# evaluation "safe" — arbitrary code execution (e.g. via raw `eval()`) is
# impossible because the parser only walks a whitelist of math operations.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Number:
    """
    Recursively evaluate an AST node containing only arithmetic.

    WHY THIS EXISTS: naive implementations use Python's built-in `eval()`
    directly on user/LLM-provided strings, which is a serious security
    hole (arbitrary code execution). This function walks a parsed AST and
    only permits numeric literals and a fixed whitelist of operators.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolExecutionError(f"Unsupported constant in expression: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ToolExecutionError(f"Operator not allowed: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ToolExecutionError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))

    raise ToolExecutionError(f"Unsupported expression element: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(
        ...,
        description=(
            "A mathematical expression to evaluate, e.g. '23 * (17 + 4) / 2'. "
            "Supports +, -, *, /, //, %, ** and parentheses."
        ),
    )


def run(args: CalculatorArgs) -> ToolResult:
    """
    Evaluate `args.expression` and return the numeric result.

    WHY THIS FUNCTION EXISTS: this is the actual "tool function" that gets
    invoked once the LLM has decided to call `calculator` and the executor
    has validated its arguments against `CalculatorArgs`.
    """
    logger.info("Calculator tool invoked with expression=%r", args.expression)
    try:
        parsed = ast.parse(args.expression, mode="eval")
        result = _safe_eval(parsed.body)
        return ToolResult(success=True, data=result)
    except ZeroDivisionError:
        return ToolResult(success=False, error="Division by zero.")
    except ToolExecutionError as exc:
        return ToolResult(success=False, error=str(exc))
    except (SyntaxError, ValueError) as exc:
        return ToolResult(success=False, error=f"Invalid expression: {exc}")


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="calculator",
    description=(
        "Evaluate a mathematical expression and return the numeric result. "
        "Use this whenever the user asks for arithmetic, unit-less math, "
        "or numeric computation that must be exact."
    ),
    args_model=CalculatorArgs,
)
